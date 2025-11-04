"""
Trino DACE Model Training Script

Trinoクエリプラン向けのDACEモデル（Transformer-based）のトレーニング。

Usage:
    # ルートディレクトリから実行
    python -m trino_lcm.scripts.train_dace \
        --workload_runs path/to/workload.json \
        --statistics_file path/to/feature_statistics.json \
        --output_dir models/trino_dace \
        --batch_size 32 \
        --epochs 100
"""

import sys
import os
import warnings

# Suppress torchdata deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchdata')

# 環境変数の設定（必須 - import前に実行）
for i in range(11):
    env_key = f'NODE{i:02d}'
    env_value = os.environ.get(env_key)
    if env_value in (None, '', 'None'):
        os.environ[env_key] = '[]'

# スクリプトがsrc/trino_lcm/scripts/にある場合、src/を親パスに追加
from pathlib import Path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse
import json
from typing import Optional, Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models.dace.dace_dataset_trino import create_dace_dataloader
from models.dace.dace_model import DACELora
from classes.classes import DACEModelConfig, DataLoaderOptions
from classes.workload_runs import WorkloadRuns
from training.training.metrics import QError, RMSE
from training.featurizations import DACEFeaturization
from training.preprocessing.feature_statistics import FeatureType
from trino_lcm.scripts.train_zeroshot import load_plans_from_files
from sklearn.preprocessing import RobustScaler
import collections


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数のパーサーを構築"""
    parser = argparse.ArgumentParser(
        description="Train DACE model for Trino query runtime prediction"
    )
    
    # モード選択
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'train_multi_all'],
        default='train',
        help='Training mode: train (single dataset) or train_multi_all (leave-one-out across all datasets)'
    )
    
    # データ関連
    parser.add_argument(
        '--workload_runs',
        type=str,
        nargs='+',
        required=False,
        help='Paths to workload run files (JSON or Trino .txt files) for training (required for train mode)'
    )
    parser.add_argument(
        '--test_workload_runs',
        type=str,
        nargs='+',
        default=None,
        help='Paths to workload run files (JSON or Trino .txt files) for testing'
    )
    parser.add_argument(
        '--statistics_file',
        type=str,
        default=None,
        help='Path to feature statistics JSON file (optional, will be auto-generated if not provided)'
    )
    parser.add_argument(
        '--train_files',
        type=str,
        nargs='+',
        default=None,
        help='Paths to Trino EXPLAIN ANALYZE .txt files for training (used to generate statistics if --statistics_file not provided)'
    )
    parser.add_argument(
        '--max_plans_per_file',
        type=int,
        default=None,
        help='Maximum number of plans to parse per file (for statistics generation)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation split ratio (default: 0.15)'
    )
    parser.add_argument(
        '--plans_dir',
        type=str,
        default='/Users/an/query_engine/explain_analyze_results/',
        help='Directory containing .txt plan files for multiple datasets (required for train_multi_all mode)'
    )
    
    # モデル設定
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='Hidden dimension for transformer (default: 128)'
    )
    parser.add_argument(
        '--node_length',
        type=int,
        default=18,
        help='Length of node feature vector (default: 18)'
    )
    parser.add_argument(
        '--pad_length',
        type=int,
        default=50,
        help='Maximum number of nodes (padding length) (default: 50)'
    )
    parser.add_argument(
        '--max_runtime',
        type=float,
        default=30000.0,
        help='Maximum runtime for normalization (ms) (default: 30000)'
    )
    parser.add_argument(
        '--loss_weight',
        type=float,
        default=0.5,
        help='Loss weight for height-based weighting (not used in Trino mode) (default: 0.5)'
    )
    
    # 訓練設定
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers (default: 4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training (default: cuda if available)'
    )
    parser.add_argument(
        '--cap_training_samples',
        type=int,
        default=None,
        help='Cap number of training samples (default: None)'
    )
    
    # 出力設定
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=10,
        help='Save checkpoint every N epochs (default: 10)'
    )
    parser.add_argument(
        '--log_every',
        type=int,
        default=1,
        help='Log metrics every N epochs (default: 1)'
    )
    
    return parser


def train_epoch(model, train_loader, optimizer, device, epoch):
    """1エポックの訓練"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        seq_encodings, attention_masks, loss_masks, run_times, labels, sample_idxs = batch
        
        # デバイスに移動
        seq_encodings = seq_encodings.to(device)
        attention_masks = attention_masks.to(device)
        loss_masks = loss_masks.to(device)
        run_times = run_times.to(device)
        labels = labels.to(device)
        
        # フォワードパス
        predictions = model((seq_encodings, attention_masks, loss_masks, run_times))
        
        # 損失計算（DaceLossが自動的に使用される）
        loss = model.loss_fxn(predictions, labels)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(sample_idxs)
        total_samples += len(sample_idxs)
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss


def validate(model, val_loader, device):
    """検証"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            seq_encodings, attention_masks, loss_masks, run_times, labels, sample_idxs = batch
            
            seq_encodings = seq_encodings.to(device)
            attention_masks = attention_masks.to(device)
            loss_masks = loss_masks.to(device)
            run_times = run_times.to(device)
            labels = labels.to(device)
            
            predictions = model((seq_encodings, attention_masks, loss_masks, run_times))
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # デバッグ: 予測値とラベルの統計を出力
    print(f"\n🔍 検証データの統計:")
    print(f"   予測値の範囲: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
    print(f"   予測値の平均: {all_predictions.mean():.4f}")
    print(f"   予測値が0以下の数: {(all_predictions <= 0).sum()} / {len(all_predictions)}")
    print(f"   ラベルの範囲: [{all_labels.min():.4f}, {all_labels.max():.4f}]")
    print(f"   ラベルの平均: {all_labels.mean():.4f}")
    print(f"   ラベルが0以下の数: {(all_labels <= 0).sum()} / {len(all_labels)}")
    print()
    
    # 予測値が0以下の場合、最小値を設定（Q-Error計算のため）
    # クエリプランの実行時間は100ms（0.1秒）～30秒の範囲
    # PostgreSQLのQErrorデフォルト値（0.1）に合わせる
    min_val = 0.1  # 0.1秒 = 100ミリ秒
    all_predictions = np.clip(all_predictions, min_val, np.inf)
    all_labels = np.clip(all_labels, min_val, np.inf)
    
    # メトリクス計算
    # QError と RMSE は Metric クラスを継承しているので evaluate_metric を使用
    # QErrorのデフォルトmin_valは0.1なので、それに合わせる
    q_error_metric = QError(min_val=min_val)
    rmse_metric = RMSE()
    
    # evaluate_metric メソッドを呼び出す
    q_error_value = q_error_metric.evaluate_metric(labels=all_labels, preds=all_predictions)
    rmse_value = rmse_metric.evaluate_metric(labels=all_labels, preds=all_predictions)
    
    # Noneの場合のフォールバック
    if q_error_value is None or np.isnan(q_error_value) or np.isinf(q_error_value):
        q_error_value = float('inf')
    if rmse_value is None or np.isnan(rmse_value) or np.isinf(rmse_value):
        rmse_value = float('inf')
    
    return q_error_value, rmse_value


def generate_feature_statistics_from_plans(
    plan_files: list[str],
    output_path: Path,
    plan_features: list[str],
    max_plans_per_file: Optional[int] = None
) -> Path:
    """
    Trinoプランファイルから特徴量統計を生成してJSONファイルとして保存
    
    Args:
        plan_files: Trino EXPLAIN ANALYZE .txtファイルのパスリスト
        output_path: 統計ファイルの保存先パス
        plan_features: 収集する特徴量のリスト（例: ["op_name", "est_card"]）
        max_plans_per_file: ファイルあたりの最大プラン数
    
    Returns:
        保存された統計ファイルのパス
    """
    print("=" * 80)
    print("特徴量統計の生成")
    print("=" * 80)
    print(f"プランファイル: {plan_files}")
    print(f"収集する特徴量: {plan_features}")
    print()
    
    # プランを読み込み
    print("📂 プランの読み込み中...")
    all_plans = load_plans_from_files(plan_files, max_plans_per_file)
    print(f"✅ {len(all_plans)} 個のプランを読み込み完了\n")
    
    # 特徴量の値を収集
    print("📊 特徴量の値を収集中...")
    value_dict = collections.defaultdict(list)
    
    def collect_features_recursively(node):
        """再帰的にノードから特徴量を収集"""
        if hasattr(node, 'plan_parameters'):
            params = node.plan_parameters
            if isinstance(params, dict):
                # dict の場合
                for feat in plan_features:
                    if feat in params:
                        value = params[feat]
                        if value is not None:
                            value_dict[feat].append(value)
            else:
                # SimpleNamespace の場合
                for feat in plan_features:
                    # Trino固有のマッピング
                    if feat == "est_card":
                        # est_card は est_rows から取得
                        value = getattr(params, "est_rows", None)
                        if value is not None:
                            value_dict[feat].append(value)
                    elif feat == "est_cost":
                        # est_cost は est_cpu を優先（Estimatesのcpu値、推定値なのでより適切）
                        # フォールバック: est_cpuがない場合はact_cpu_timeを使用、それもなければact_scheduled_time、それもなければ0.0
                        value = getattr(params, "est_cpu", None)
                        if value is None:
                            value = getattr(params, "act_cpu_time", None)
                        if value is None:
                            value = getattr(params, "act_scheduled_time", None)
                        if value is None:
                            value = 0.0
                        value_dict[feat].append(value)
                    else:
                        # その他の特徴量
                        if hasattr(params, feat):
                            value = getattr(params, feat)
                            if value is not None:
                                value_dict[feat].append(value)
        
        # 子ノードも再帰的に処理
        for child in node.children:
            collect_features_recursively(child)
    
    for plan in tqdm(all_plans, desc="プラン処理"):
        collect_features_recursively(plan)
    
    print()
    
    # 統計を計算
    print("📈 統計を計算中...")
    statistics_dict = {}
    
    for feat_name, values in value_dict.items():
        values = [v for v in values if v is not None]
        if len(values) == 0:
            continue
        
        # 数値型かどうかを判定
        if all([isinstance(v, (int, float)) for v in values]):
            # 数値型: RobustScaler を使用
            scaler = RobustScaler()
            np_values = np.array(values, dtype=np.float32).reshape(-1, 1)
            scaler.fit(np_values)
            
            statistics_dict[feat_name] = {
                "max": float(np_values.max()),
                "scale": float(scaler.scale_.item()),
                "center": float(scaler.center_.item()),
                "type": str(FeatureType.numeric)
            }
        else:
            # カテゴリカル型: value_dict を作成
            unique_values = sorted(set(str(v) for v in values))
            statistics_dict[feat_name] = {
                "value_dict": {v: idx for idx, v in enumerate(unique_values)},
                "no_vals": len(unique_values),
                "type": str(FeatureType.categorical)
            }
    
    # 指定された特徴量で、収集されなかったもの（Trinoには存在しない特徴量）を追加
    for feat_name in plan_features:
        if feat_name not in statistics_dict:
            # 特徴量が存在しない場合は、デフォルト値で統計を追加
            if feat_name == 'est_cost':
                # est_costはTrinoにはないので、デフォルト値0で追加
                statistics_dict[feat_name] = {
                    "max": 0.0,
                    "scale": 1.0,  # スケール1.0で0を中心に
                    "center": 0.0,
                    "type": str(FeatureType.numeric)
                }
                print(f"   ⚠️  {feat_name} がTrinoプランに存在しないため、デフォルト値0で追加しました")
            else:
                # その他の欠損特徴量もデフォルト値で追加
                statistics_dict[feat_name] = {
                    "max": 0.0,
                    "scale": 1.0,
                    "center": 0.0,
                    "type": str(FeatureType.numeric)
                }
                print(f"   ⚠️  {feat_name} が収集されなかったため、デフォルト値0で追加しました")
    
    # JSONファイルとして保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(statistics_dict, f, indent=2)
    
    print(f"✅ 統計ファイルを保存: {output_path}")
    print(f"   特徴量数: {len(statistics_dict)}")
    print()
    
    return output_path


def run(args) -> int:
    """トレーニングを実行"""
    print("=" * 80)
    print("DACE Model Training for Trino")
    print(f"Mode: {args.mode}")
    print("=" * 80)
    print()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # デバイス設定
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print()
    
    # train_multi_allモードの処理
    if args.mode == 'train_multi_all':
        return run_train_multi_all(args, output_dir, device)
    
    # 従来のtrainモード
    if not args.workload_runs:
        raise ValueError("--workload_runs is required for train mode")
    
    # モデル設定（featurizationを先に作成）
    featurization = DACEFeaturization()
    model_config = DACEModelConfig(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        node_length=args.node_length,
        pad_length=args.pad_length,
        max_runtime=args.max_runtime,
        loss_weight=args.loss_weight,
        num_workers=args.num_workers,
        device=device,
        loss_class_name='DaceLoss',
        cap_training_samples=args.cap_training_samples,
        featurization=featurization,
        optimizer_kwargs=dict(lr=args.learning_rate)
    )
    
    # 統計ファイルの処理
    statistics_file = Path(args.statistics_file) if args.statistics_file else None
    
    if statistics_file is None or not statistics_file.exists():
        # 統計ファイルが指定されていない、または存在しない場合は自動生成
        # --train_filesが指定されている場合はそれを使用、なければ--workload_runsから.txtファイルを探す
        stat_files = args.train_files
        if not stat_files:
            # --workload_runsから.txtファイルを抽出
            stat_files = [f for f in args.workload_runs if Path(f).suffix.lower() == '.txt']
        
        if not stat_files:
            raise ValueError(
                "統計ファイルが指定されていません。以下のいずれかを指定してください:\n"
                "  1. --statistics_file: 既存の統計ファイルのパス\n"
                "  2. --train_files: 統計生成用のTrinoプランファイル（.txt）のパス\n"
                "  3. --workload_runs に .txt ファイルを含める（自動的に統計生成に使用されます）"
            )
        
        # 自動生成する統計ファイルのパス
        auto_stats_path = output_dir / 'feature_statistics.json'
        
        # 統計を生成（featurizationで指定された全ての特徴量を含める）
        # Trinoにはない特徴量（est_cost）も統計ファイルに含め、デフォルト値0で処理する
        plan_features = list(featurization.PLAN_FEATURES)
        
        # Trino固有のマッピング: est_card は est_rows から取得
        if 'est_card' not in plan_features and 'est_rows' in plan_features:
            # est_rows があれば est_card に変換して処理
            pass  # 統計生成時に適切にマッピングされる
        
        generate_feature_statistics_from_plans(
            plan_files=stat_files,
            output_path=auto_stats_path,
            plan_features=plan_features,
            max_plans_per_file=args.max_plans_per_file
        )
        
        # generate_feature_statistics_from_plans 内で既に不足している特徴量が追加されているので、
        # ここでは確認のみ
        statistics_file = auto_stats_path
    else:
        print(f"既存の統計ファイルを使用: {statistics_file}")
        print()
    
    # ワークロード設定
    train_workload_runs = [Path(p) for p in args.workload_runs]
    test_workload_runs = [Path(p) for p in args.test_workload_runs] if args.test_workload_runs else []
    
    workload_runs = WorkloadRuns(
        train_workload_runs=train_workload_runs,
        test_workload_runs=test_workload_runs
    )
    
    # データローダー設定
    dataloader_options = DataLoaderOptions(
        shuffle=True,
        val_ratio=args.val_ratio,
        pin_memory=(device.type == 'cuda')
    )
    
    print("Creating dataloaders...")
    feature_statistics, train_loader, val_loader, test_loaders = create_dace_dataloader(
        statistics_file=statistics_file,
        model_config=model_config,
        workload_runs=workload_runs,
        dataloader_options=dataloader_options
    )
    
    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
    if test_loaders:
        print(f"Test loaders: {len(test_loaders)}")
    print()
    
    # モデル作成
    print("Creating DACE model...")
    model = DACELora(config=model_config)
    model.to(device)
    
    # オプティマイザー
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 訓練ループ
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    print()
    
    best_q_error = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # 訓練
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # 検証
        if val_loader and epoch % args.log_every == 0:
            q_error, rmse = validate(model, val_loader, device)
            
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Q-Error: {q_error:.4f}")
            print(f"  Val RMSE: {rmse:.4f}")
            print()
            
            # ベストモデル保存
            if q_error < best_q_error:
                best_q_error = q_error
                checkpoint_path = output_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'q_error': q_error,
                    'rmse': rmse,
                }, checkpoint_path)
                print(f"  ✓ Saved best model (Q-Error: {q_error:.4f})")
                print()
        
        # 定期的なチェックポイント保存
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    # テスト評価
    if test_loaders:
        print("=" * 80)
        print("Testing...")
        print("=" * 80)
        
        for i, test_loader in enumerate(test_loaders):
            q_error, rmse = validate(model, test_loader, device)
            print(f"Test Loader {i+1}:")
            print(f"  Q-Error: {q_error:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print()
    
    print("=" * 80)
    print("Training completed!")
    print(f"Best validation Q-Error: {best_q_error:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 80)
    
    return 0


def run_train_multi_all(args, output_dir: Path, device: torch.device) -> int:
    """20個すべてのデータセットについてleave-one-out validationを実行"""
    # サポートされている20個のデータセット（アルファベット順）
    ALL_DATASETS = [
        'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
        'consumer', 'credit', 'employee', 'fhnk', 'financial', 'geneea',
        'genome', 'hepatitis', 'imdb', 'movielens', 'seznam', 'ssb',
        'tournament', 'tpc_h', 'walmart'
    ]
    
    plans_dir = Path(args.plans_dir)
    
    # 利用可能なデータセットを確認
    txt_files = sorted([p for p in plans_dir.glob('*.txt')])
    available_datasets = set()
    for p in txt_files:
        stem = p.stem  # .txtを除いたファイル名
        parts = stem.split('_')
        # 最長マッチ: ALL_DATASETSから最長の一致を探す（tpc_hなどアンダースコアを含むデータセット名に対応）
        matched_dataset = None
        for i in range(len(parts), 0, -1):
            candidate = '_'.join(parts[:i])
            if candidate in ALL_DATASETS:
                matched_dataset = candidate
                break
        if matched_dataset:
            available_datasets.add(matched_dataset)
    
    available_datasets = sorted(list(available_datasets))
    print(f"\n{'='*80}")
    print(f"Leave-One-Out Validation for All Datasets (DACE)")
    print(f"{'='*80}")
    print(f"利用可能なデータセット: {len(available_datasets)} / {len(ALL_DATASETS)}")
    print(f"データセット: {', '.join(available_datasets)}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"{'='*80}\n")
    
    # モデル設定
    featurization = DACEFeaturization()
    plan_features = list(featurization.PLAN_FEATURES)
    
    # 各データセットについて訓練・テストを実行
    results_summary = []
    
    for idx, test_dataset in enumerate(available_datasets, 1):
        print(f"\n{'#'*80}")
        print(f"# [{idx}/{len(available_datasets)}] Testing dataset: {test_dataset}")
        print(f"{'#'*80}\n")
        
        try:
            # データファイルを準備
            train_files = []
            test_files = []
            
            for p in txt_files:
                stem = p.stem  # .txtを除いたファイル名
                parts = stem.split('_')
                # 最長マッチ: ALL_DATASETSから最長の一致を探す
                matched_dataset = None
                for i in range(len(parts), 0, -1):
                    candidate = '_'.join(parts[:i])
                    if candidate in ALL_DATASETS:
                        matched_dataset = candidate
                        break
                
                if matched_dataset == test_dataset:
                    test_files.append(p)
                elif matched_dataset and matched_dataset in available_datasets:
                    train_files.append(p)
            
            if not train_files or not test_files:
                print(f"⚠️  {test_dataset}: 訓練ファイルまたはテストファイルが見つかりません。スキップします。")
                results_summary.append({
                    'test_dataset': test_dataset,
                    'status': 'skipped',
                    'reason': 'missing files'
                })
                continue
            
            # 統計ファイルを生成（全データから）
            all_stat_files = train_files + test_files
            model_dir = output_dir / f'models_{test_dataset}'
            model_dir.mkdir(parents=True, exist_ok=True)
            statistics_file = model_dir / 'feature_statistics.json'
            
            generate_feature_statistics_from_plans(
                plan_files=[str(f) for f in all_stat_files],
                output_path=statistics_file,
                plan_features=plan_features,
                max_plans_per_file=args.max_plans_per_file
            )
            
            # ワークロード設定
            train_workload_runs = [p for p in train_files]
            test_workload_runs = [p for p in test_files]
            
            workload_runs = WorkloadRuns(
                train_workload_runs=train_workload_runs,
                test_workload_runs=test_workload_runs
            )
            
            # モデル設定
            model_config = DACEModelConfig(
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                node_length=args.node_length,
                pad_length=args.pad_length,
                max_runtime=args.max_runtime,
                loss_weight=args.loss_weight,
                num_workers=args.num_workers,
                device=device,
                loss_class_name='DaceLoss',
                cap_training_samples=args.cap_training_samples,
                featurization=featurization,
                optimizer_kwargs=dict(lr=args.learning_rate)
            )
            
            # データローダー設定
            dataloader_options = DataLoaderOptions(
                shuffle=True,
                val_ratio=args.val_ratio,
                pin_memory=(device.type == 'cuda')
            )
            
            print(f"📊 Leave-One-Out Validation [{idx}/{len(available_datasets)}]:")
            print(f"  - Training files: {len(train_files)}")
            print(f"  - Test files: {len(test_files)}")
            print()
            
            print("Creating dataloaders...")
            feature_statistics, train_loader, val_loader, test_loaders = create_dace_dataloader(
                statistics_file=statistics_file,
                model_config=model_config,
                workload_runs=workload_runs,
                dataloader_options=dataloader_options
            )
            
            print(f"Training batches: {len(train_loader)}")
            if val_loader:
                print(f"Validation batches: {len(val_loader)}")
            if test_loaders:
                print(f"Test loaders: {len(test_loaders)}")
            print()
            
            # モデル作成
            print("Creating DACE model...")
            model = DACELora(config=model_config)
            model.to(device)
            
            # オプティマイザー
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            
            # 訓練ループ
            print("=" * 80)
            print("Starting training...")
            print("=" * 80)
            print()
            
            best_q_error = float('inf')
            best_epoch = 0
            
            for epoch in range(1, args.epochs + 1):
                # 訓練
                train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
                
                # 検証
                if val_loader and epoch % args.log_every == 0:
                    q_error, rmse = validate(model, val_loader, device)
                    
                    print(f"Epoch {epoch}/{args.epochs}")
                    print(f"  Train Loss: {train_loss:.4f}")
                    print(f"  Val Q-Error: {q_error:.4f}")
                    print(f"  Val RMSE: {rmse:.4f}")
                    print()
                    
                    # ベストモデル保存
                    if q_error < best_q_error:
                        best_q_error = q_error
                        best_epoch = epoch
                        checkpoint_path = model_dir / 'best_model.pt'
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'q_error': q_error,
                            'rmse': rmse,
                        }, checkpoint_path)
                        print(f"  ✓ Saved best model (Q-Error: {q_error:.4f})")
                        print()
                
                # 定期的なチェックポイント保存
                if epoch % args.save_every == 0:
                    checkpoint_path = model_dir / f'checkpoint_epoch_{epoch}.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
            
            # テスト評価
            test_results = {}
            if test_loaders:
                print("=" * 80)
                print("Testing...")
                print("=" * 80)
                
                all_test_q_errors = []
                all_test_rmses = []
                
                for i, test_loader in enumerate(test_loaders):
                    q_error, rmse = validate(model, test_loader, device)
                    all_test_q_errors.append(q_error)
                    all_test_rmses.append(rmse)
                    print(f"Test Loader {i+1}:")
                    print(f"  Q-Error: {q_error:.4f}")
                    print(f"  RMSE: {rmse:.4f}")
                    print()
                
                test_results = {
                    'test_mean_q_error': float(np.mean(all_test_q_errors)) if all_test_q_errors else None,
                    'test_median_q_error': float(np.median(all_test_q_errors)) if all_test_q_errors else None,
                    'test_mean_rmse': float(np.mean(all_test_rmses)) if all_test_rmses else None,
                    'test_samples': sum(len(loader.dataset) for loader in test_loaders) if test_loaders else 0
                }
                
                # テスト結果を保存
                results_file = model_dir / 'test_results.json'
                with open(results_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                print(f"✅ テスト結果を保存: {results_file}")
                print()
            
            # 結果を保存
            results_summary.append({
                'test_dataset': test_dataset,
                'model_dir': str(model_dir),
                'best_val_q_error': float(best_q_error),
                'best_epoch': int(best_epoch),
                **test_results,
                'status': 'completed'
            })
            
            print(f"✅ [{idx}/{len(available_datasets)}] {test_dataset} の訓練・テスト完了")
            print(f"   モデル保存先: {model_dir}")
            print()
            
        except Exception as e:
            print(f"❌ [{idx}/{len(available_datasets)}] {test_dataset} でエラーが発生:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'test_dataset': test_dataset,
                'status': 'failed',
                'error': str(e)
            })
            continue
    
    # 全体のサマリーを保存
    summary_file = output_dir / 'leave_one_out_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'total_datasets': len(available_datasets),
            'completed': len([r for r in results_summary if r['status'] == 'completed']),
            'failed': len([r for r in results_summary if r['status'] == 'failed']),
            'skipped': len([r for r in results_summary if r.get('status') == 'skipped']),
            'results': results_summary
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("🎉 全データセットでのLeave-One-Out Validation完了！")
    print("=" * 80)
    print(f"完了: {len([r for r in results_summary if r['status'] == 'completed'])}/{len(available_datasets)}")
    print(f"失敗: {len([r for r in results_summary if r['status'] == 'failed'])}/{len(available_datasets)}")
    print(f"スキップ: {len([r for r in results_summary if r.get('status') == 'skipped'])}/{len(available_datasets)}")
    print(f"サマリーファイル: {summary_file}")
    print()
    
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """メイン関数"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

