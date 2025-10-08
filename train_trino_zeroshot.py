#!/usr/bin/env python3
"""
Trino Zero-Shot Model Training Script
統合版: 効率的なバッチ処理とDataLoader対応
"""

import sys
import os
sys.path.append('src')

import argparse
import json
import re
import functools
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.trino.parse_plan import parse_trino_plans
from models.zeroshot.specific_models.trino_zero_shot import TrinoZeroShotModel
from training.featurizations import TrinoTrueCardDetail
from models.zeroshot.trino_plan_batching import trino_plan_collator
from classes.classes import ZeroShotModelConfig
from training.preprocessing.feature_statistics import gather_feature_statistics, FeatureType

# 環境変数の設定（必須）
for i in range(11):
    os.environ.setdefault(f'NODE{i:02d}', '[]')


class TrinoPlanDataset(Dataset):
    """Trinoクエリプランのデータセット"""
    
    def __init__(self, plans):
        """
        Args:
            plans: TrinoPlanOperatorオブジェクトのリスト
        """
        self.plans = plans
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx):
        return idx, self.plans[idx]


class MockQuery:
    """モックのQueryクラス"""
    
    def __init__(self, plan_text):
        self.plan_text = plan_text
        self.timeout = False
        self.analyze_plans = [plan_text]  # parse_trino_plansが期待する形式
        
        # verbose_planはリスト形式で提供（parse_trino_plansが期待する形式）
        self.verbose_plan = plan_text.split('\n')
        
        # 実行時間を抽出
        import re
        execution_time_match = re.search(r'Execution Time: ([\d.]+)ms', plan_text)
        if execution_time_match:
            self.execution_time = float(execution_time_match.group(1))
        else:
            self.execution_time = 1000.0  # デフォルトの実行時間（ミリ秒）


class MockRunStats:
    """モックのRunStatsクラス（テキストファイル用）"""
    
    def __init__(self, plans_text):
        self.plans_text = plans_text
        # parse_trino_plansで必要な属性を追加
        self.query_list = [MockQuery(plan_text) for plan_text in plans_text]
        self.database_stats = {}
    
    def __iter__(self):
        for plan_text in self.plans_text:
            yield plan_text


def split_query_plans(file_path):
    """クエリプランファイルを個別のプランに分割"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # クエリプランを分割
    plans_text = []
    current_plan = []
    
    for line in content.split('\n'):
        if line.startswith('-- ') and 'stmt' in line and current_plan:
            plans_text.append('\n'.join(current_plan))
            current_plan = [line]
        else:
            current_plan.append(line)
    
    if current_plan:
        plans_text.append('\n'.join(current_plan))
    
    return plans_text


def load_plans_from_files(file_paths, max_plans_per_file=None):
    """
    複数のファイルからプランを読み込み
    
    Args:
        file_paths: プランファイルのパスのリスト
        max_plans_per_file: 各ファイルから読み込む最大プラン数
    
    Returns:
        全プランのリスト
    """
    all_plans = []
    
    print(f"📂 {len(file_paths)}個のファイルからプランを読み込み中...")
    
    for file_idx, file_path in enumerate(tqdm(file_paths, desc="ファイル読み込み")):
        # JSONファイルの場合
        if str(file_path).endswith('.json'):
            with open(file_path, 'r') as f:
                run_data = json.load(f)
            
            # parse_trino_plansを使用してプランを解析
            # TODO: JSONからrun_statsオブジェクトを再構築
            pass
        
        # テキストファイルの場合（EXPLAIN ANALYZE出力）
        else:
            # ファイルを読み込んで個別のクエリプランに分割
            with open(file_path, 'r') as f:
                content = f.read()
            
            # クエリプランを分割
            plans_text = []
            current_plan = []
            
            for line in content.split('\n'):
                if line.startswith('-- ') and 'stmt' in line and current_plan:
                    plans_text.append('\n'.join(current_plan))
                    current_plan = [line]
                else:
                    current_plan.append(line)
            
            if current_plan:
                plans_text.append('\n'.join(current_plan))
            
            # 最大プラン数を制限
            if max_plans_per_file:
                plans_text = plans_text[:max_plans_per_file]
            
            print(f"  - {file_path}: {len(plans_text)}個のプランを検出")
            
            # MockRunStatsを作成してパース
            mock_stats = MockRunStats(plans_text)
            
            parsed_runs, _ = parse_trino_plans(
                mock_stats,
                min_runtime=0,
                max_runtime=1000000,
                parse_baseline=False,
                include_zero_card=True
            )
            
            print(f"  - パース結果: {len(parsed_runs['parsed_plans'])}個のプラン")
            
            # データベースIDを設定
            for plan in parsed_runs['parsed_plans']:
                plan.database_id = file_idx
            
            all_plans.extend(parsed_runs['parsed_plans'])
    
    print(f"  - 総プラン数: {len(all_plans)}")
    return all_plans


def create_feature_statistics_from_plans(plans, plan_featurization, output_path=None):
    """
    プランから特徴量統計を動的に収集
    
    Args:
        plans: TrinoPlanOperatorオブジェクトのリスト
        plan_featurization: 特徴量化設定
        output_path: 統計情報の出力パス（オプション）
    
    Returns:
        feature_statistics辞書
    """
    print("📊 プランから特徴量統計を収集中...")
    
    # ダミーの統計情報から開始（動的に拡張）
    feature_statistics = create_dummy_feature_statistics(plan_featurization)
    
    # TODO: 実際のプランから統計を収集して更新
    # この部分は、より正確な統計情報が必要な場合に実装
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(feature_statistics, f, indent=2)
        print(f"  - 特徴量統計を {output_path} に保存")
    
    return feature_statistics


def create_dummy_feature_statistics(plan_featurization):
    """ダミーの特徴量統計情報を作成"""
    feature_statistics = {}
    
    # すべての特徴量を定義
    all_features = set()
    for features in plan_featurization.VARIABLES.values():
        all_features.update(features)
    
    for feat_name in all_features:
        if feat_name == 'op_name':
            operator_dict = {
                'Aggregate': 0, 'LocalExchange': 1, 'RemoteSource': 2, 
                'ScanFilter': 3, 'ScanFilterProject': 4, 'Project': 5, 
                'InnerJoin': 6, 'HashJoin': 7, 'NestedLoopJoin': 8,
                'Sort': 87, 'Limit': 95, 'TopN': 11,
                'TableScan': 13, 'FilterProject': 14, 'Exchange': 15,
                'LeftJoin': 32, 'ScanProject': 60, 'Filter': 61,
            }
            feature_statistics[feat_name] = {
                'type': str(FeatureType.categorical),
                'value_dict': operator_dict,
                'no_vals': 200  # 動的に追加される演算子に対応するため大きめに設定
            }
        elif feat_name == 'operator':
            operator_dict = {
                'Aggregate': 0, 'LocalExchange': 1, 'RemoteSource': 2, 
                'ScanFilter': 3, 'ScanFilterProject': 4, 'Project': 5, 
                'InnerJoin': 6, 'HashJoin': 7, 'NestedLoopJoin': 8,
                'Sort': 87, 'Limit': 95, 'TopN': 11,
                'TableScan': 13, 'FilterProject': 14, 'Exchange': 15,
                'LeftJoin': 32, 'ScanProject': 60, 'Filter': 61,
            }
            feature_statistics[feat_name] = {
                'type': str(FeatureType.categorical),
                'value_dict': operator_dict,
                'no_vals': len(operator_dict)
            }
        elif feat_name == 'aggregation':
            # 集約関数の特徴量統計
            aggregation_dict = {
                'Aggregator.COUNT': 0,
                'Aggregator.SUM': 1,
                'Aggregator.AVG': 2,
                'Aggregator.MIN': 3,
                'Aggregator.MAX': 4,
                None: 5  # 集約なし
            }
            feature_statistics[feat_name] = {
                'type': str(FeatureType.categorical),
                'value_dict': aggregation_dict,
                'no_vals': len(aggregation_dict)
            }
        elif feat_name in ['table_name', 'column_name']:
            feature_statistics[feat_name] = {
                'type': str(FeatureType.categorical),
                'value_dict': {},
                'no_vals': 1000
            }
        elif feat_name in ['rows', 'size', 'cpu', 'memory', 'network']:
            feature_statistics[feat_name] = {
                'type': str(FeatureType.numeric),
                'mean': 0.0,
                'std': 1.0,
                'min': 0.0,
                'max': 1000000.0,
                'center': 0.0,
                'scale': 1.0
            }
        else:
            # その他の特徴量は数値として扱う
            feature_statistics[feat_name] = {
                'type': str(FeatureType.numeric),
                'mean': 0.0,
                'std': 1.0,
                'min': 0.0,
                'max': 100.0,
                'center': 0.0,
                'scale': 1.0
            }
    
    return feature_statistics


def collect_feature_statistics(workload_run_paths, output_path):
    """
    Trinoワークロードから特徴量統計を収集
    
    Args:
        workload_run_paths: ワークロード実行結果のパスのリスト
        output_path: 統計情報の出力パス
    """
    print(f"📊 特徴量統計情報を収集中...")
    
    # gather_feature_statistics関数を使用
    gather_feature_statistics(workload_run_paths, output_path)
    
    print(f"✅ 特徴量統計情報を {output_path} に保存しました")


def train_epoch(model, train_loader, optimizer, device):
    """1エポックのトレーニング"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for graph, features, labels, sample_idxs in tqdm(train_loader, desc="Training"):
        # データをデバイスに転送
        graph = graph.to(device)
        features = {k: v.to(device) for k, v in features.items()}
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device).reshape(-1, 1)
        
        optimizer.zero_grad()
        
        # フォワードパス
        predictions = model((graph, features))
        
        # 損失計算
        loss = model.loss_fxn(predictions, labels_tensor)
        
        # バックプロパゲーション
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def validate(model, val_loader, device):
    """検証"""
    model.eval()
    total_loss = 0
    num_batches = 0
    predictions_all = []
    labels_all = []
    
    with torch.no_grad():
        for graph, features, labels, sample_idxs in tqdm(val_loader, desc="Validation"):
            # データをデバイスに転送
            graph = graph.to(device)
            features = {k: v.to(device) for k, v in features.items()}
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device).reshape(-1, 1)
            
            # フォワードパス
            predictions = model((graph, features))
            
            # 損失計算
            loss = model.loss_fxn(predictions, labels_tensor)
            total_loss += loss.item()
            num_batches += 1
            
            predictions_all.append(predictions.cpu().numpy())
            labels_all.append(labels_tensor.cpu().numpy())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # メトリクス計算
    if len(predictions_all) > 0:
        predictions_all = np.concatenate(predictions_all).flatten()
        labels_all = np.concatenate(labels_all).flatten()
        
        # Q-Error
        q_errors = np.maximum(predictions_all / labels_all, labels_all / predictions_all)
        median_q_error = np.median(q_errors)
        mean_q_error = np.mean(q_errors)
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions_all - labels_all) ** 2))
        
        return avg_loss, median_q_error, mean_q_error, rmse
    
    return avg_loss, None, None, None


def main():
    parser = argparse.ArgumentParser(description='Train Trino Zero-Shot Model (統合版)')
    
    # データ関連の引数
    parser.add_argument('--train_files', type=str, required=True,
                        help='トレーニング用ファイルパス（カンマ区切り）')
    parser.add_argument('--test_file', type=str, required=True,
                        help='テスト用ファイルパス')
    parser.add_argument('--statistics_file', type=str, default=None,
                        help='特徴量統計情報ファイルのパス（オプション）')
    
    # モデル関連の引数
    parser.add_argument('--output_dir', type=str, default='models/trino_zeroshot',
                        help='モデル出力ディレクトリ')
    parser.add_argument('--epochs', type=int, default=100,
                        help='エポック数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学習率')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隠れ層の次元数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='デバイス (cuda/cpu)')
    
    # データ処理関連の引数
    parser.add_argument('--max_plans_per_file', type=int, default=None,
                        help='各ファイルから読み込む最大プラン数')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='検証セットの割合')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoaderのワーカー数')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Trino Zero-Shot Model Training (統合版)")
    print("=" * 80)
    print(f"Train files: {args.train_files}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print()
    
    # 1. プランの読み込み
    print("📂 ステップ1: プランの読み込み")
    train_file_paths = [Path(p.strip()) for p in args.train_files.split(',')]
    test_file_path = Path(args.test_file)
    
    # トレーニングプランの読み込み
    train_plans = load_plans_from_files(train_file_paths, args.max_plans_per_file)
    
    # テストプランの読み込み
    test_plans = load_plans_from_files([test_file_path], args.max_plans_per_file)
    
    print()
    
    # 2. トレーニング/検証セットの分割
    print("📊 ステップ2: トレーニング/検証セットの分割")
    val_size = int(len(train_plans) * args.val_ratio)
    train_size = len(train_plans) - val_size
    
    train_plans_split, val_plans_split = random_split(
        train_plans, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  - トレーニングプラン: {len(train_plans_split)}")
    print(f"  - 検証プラン: {len(val_plans_split)}")
    print(f"  - テストプラン: {len(test_plans)}")
    print()
    
    # 3. 特徴量統計の準備
    print("🔧 ステップ3: 特徴量統計の準備")
    plan_featurization = TrinoTrueCardDetail()
    
    # 特徴量統計情報の読み込みまたは作成
    if args.statistics_file and Path(args.statistics_file).exists():
        with open(args.statistics_file, 'r') as f:
            feature_statistics = json.load(f)
        print(f"  - 既存の統計情報を読み込み: {len(feature_statistics)} features")
    else:
        # トレーニングプランから特徴量統計を収集
        feature_statistics = create_feature_statistics_from_plans(
            [train_plans[i] for i in train_plans_split.indices],
            plan_featurization,
            args.statistics_file
        )
    
    db_statistics = {}  # データベース統計は空（無視）
    print()
    
    # 4. データセットとDataLoaderの作成
    print("📦 ステップ4: データセットとDataLoaderの作成")
    
    # collate_fnを作成（バッチをグラフに変換）
    collate_fn = functools.partial(
        trino_plan_collator,
        feature_statistics=feature_statistics,
        db_statistics=db_statistics,
        plan_featurization=plan_featurization
    )
    
    # DataLoaderの作成
    train_loader = DataLoader(
        TrinoPlanDataset([train_plans[i] for i in train_plans_split.indices]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        TrinoPlanDataset([train_plans[i] for i in val_plans_split.indices]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        TrinoPlanDataset(test_plans),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"  - トレーニングバッチ数: {len(train_loader)}")
    print(f"  - 検証バッチ数: {len(val_loader)}")
    print(f"  - テストバッチ数: {len(test_loader)}")
    print()
    
    # 5. モデル作成
    print("🤖 ステップ5: モデル作成")
    model_config = ZeroShotModelConfig(
        hidden_dim=args.hidden_dim,
        hidden_dim_plan=args.hidden_dim,
        hidden_dim_pred=args.hidden_dim,
        p_dropout=0.1,
        featurization=plan_featurization,
        output_dim=1,
        batch_size=args.batch_size
    )
    
    encoders = [
        ('plan', plan_featurization.PLAN_FEATURES),
        ('logical_pred', plan_featurization.FILTER_FEATURES),
        ('column', plan_featurization.COLUMN_FEATURES),
        ('table', plan_featurization.TABLE_FEATURES),
        ('filter_column', plan_featurization.FILTER_FEATURES + plan_featurization.COLUMN_FEATURES),
        ('output_column', plan_featurization.OUTPUT_COLUMN_FEATURES)
    ]
    
    model = TrinoZeroShotModel(
        model_config=model_config,
        device=args.device,
        feature_statistics=feature_statistics,
        plan_featurization=plan_featurization,
        add_tree_model_types=[],
        encoders=encoders
    )
    
    model = model.to(args.device)
    
    print(f"  - モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 6. オプティマイザとスケジューラ
    print("⚙️  ステップ6: オプティマイザ設定")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    print(f"  - Optimizer: Adam (lr={args.lr})")
    print(f"  - Scheduler: ReduceLROnPlateau")
    print()
    
    # 7. トレーニングループ
    print("🚀 ステップ7: トレーニング開始")
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # トレーニング
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        
        # 検証
        val_result = validate(model, val_loader, args.device)
        val_loss, val_median_q_error, val_mean_q_error, val_rmse = val_result
        
        # 学習率スケジューラ
        scheduler.step(val_loss)
        
        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            # モデルの保存
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        
        # ログ出力
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Q-Error: {val_median_q_error:.4f}, "
                  f"Best: {best_val_loss:.4f} (Epoch {best_epoch})")
    
    print()
    print("✅ トレーニング完了!")
    print()
    
    # 8. テストセットでの評価
    print("📊 ステップ8: テストセットでの最終評価")
    
    # ベストモデルを読み込み
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_result = validate(model, test_loader, args.device)
    test_loss, test_median_q_error, test_mean_q_error, test_rmse = test_result
    
    print(f"【テストセット評価結果】")
    print(f"  - サンプル数: {len(test_plans)}")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - RMSE: {test_rmse:.4f}秒 ({test_rmse*1000:.2f}ms)")
    print(f"  - Median Q-Error: {test_median_q_error:.4f}")
    print(f"  - Mean Q-Error: {test_mean_q_error:.4f}")
    print()
    
    print("=" * 80)
    print("トレーニング完了！")
    print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Test Median Q-Error: {test_median_q_error:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

