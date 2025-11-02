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


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数のパーサーを構築"""
    parser = argparse.ArgumentParser(
        description="Train DACE model for Trino query runtime prediction"
    )
    
    # データ関連
    parser.add_argument(
        '--workload_runs',
        type=str,
        nargs='+',
        required=True,
        help='Paths to workload run JSON files for training'
    )
    parser.add_argument(
        '--test_workload_runs',
        type=str,
        nargs='+',
        default=None,
        help='Paths to workload run JSON files for testing'
    )
    parser.add_argument(
        '--statistics_file',
        type=str,
        required=True,
        help='Path to feature statistics JSON file'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation split ratio (default: 0.15)'
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
    
    # メトリクス計算
    # QError と RMSE は Metric クラスを継承しているので evaluate_metric を使用
    q_error_metric = QError()
    rmse_metric = RMSE()
    
    # evaluate_metric メソッドを呼び出す
    q_error_value = q_error_metric.evaluate_metric(labels=all_labels, preds=all_predictions)
    rmse_value = rmse_metric.evaluate_metric(labels=all_labels, preds=all_predictions)
    
    # Noneの場合のフォールバック
    if q_error_value is None:
        q_error_value = float('inf')
    if rmse_value is None:
        rmse_value = float('inf')
    
    return q_error_value, rmse_value


def run(args) -> int:
    """トレーニングを実行"""
    print("=" * 80)
    print("DACE Model Training for Trino")
    print("=" * 80)
    print()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # デバイス設定
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print()
    
    # ワークロード設定
    train_workload_runs = [Path(p) for p in args.workload_runs]
    test_workload_runs = [Path(p) for p in args.test_workload_runs] if args.test_workload_runs else []
    
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
        featurization=DACEFeaturization(),
        optimizer_kwargs=dict(lr=args.learning_rate)
    )
    
    # データローダー設定
    dataloader_options = DataLoaderOptions(
        shuffle=True,
        val_ratio=args.val_ratio,
        pin_memory=(device.type == 'cuda')
    )
    
    print("Creating dataloaders...")
    feature_statistics, train_loader, val_loader, test_loaders = create_dace_dataloader(
        statistics_file=Path(args.statistics_file),
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """メイン関数"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

