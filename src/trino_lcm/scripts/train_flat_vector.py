"""
Trino Flat-Vector Model Training Script

Trinoクエリプラン向けのFlat-Vectorモデルのトレーニング。
これは、既存のPostgreSQL用Flat-Vectorモデルをtrino向けに再実装したものです。

Usage:
    # ルートディレクトリから実行
    python -m trino_lcm.scripts.train_flat_vector \
        --train_files accidents_valid_verbose.txt \
        --test_file accidents_valid_verbose.txt \
        --output_dir models/trino_flat_vector \
        --epochs 1000
"""

import sys
import os
from pathlib import Path
import argparse
import json
from typing import Optional, Sequence
import numpy as np
import torch

# 環境変数の設定（必須 - import前に実行）
for i in range(11):
    env_key = f'NODE{i:02d}'
    env_value = os.environ.get(env_key)
    if env_value in (None, '', 'None'):
        os.environ[env_key] = '[]'

# スクリプトがsrc/trino_lcm/scripts/にある場合、src/を親パスに追加
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from trino_lcm.models.flat_vector import (
    load_trino_plans_from_files,
    collect_operator_types,
    create_flat_vector_dataset,
    train_flat_vector_model,
    predict_flat_vector_model
)
from training.training.metrics import QError, RMSE, MAPE


def evaluate_with_metrics(bst, X, y, dataset_name="Test"):
    """
    既存のメトリクスを使用してモデルを評価
    
    Args:
        bst: LightGBMモデル
        X: 特徴量
        y: ラベル
        dataset_name: データセット名
    
    Returns:
        評価メトリクス辞書
    """
    # 予測
    y_pred = predict_flat_vector_model(bst, X)
    
    # メトリクスの定義
    metrics = [
        RMSE(),
        MAPE(),
        QError(percentile=50, early_stopping_metric=True),
        QError(percentile=95),
        QError(percentile=99),
        QError(percentile=100)
    ]
    
    # 評価実行
    metrics_dict = {}
    for metric in metrics:
        metric.evaluate(
            metrics_dict=metrics_dict,
            model=None,
            labels=y,
            preds=y_pred,
            probs=None
        )
    
    # 結果を整形
    results = {
        'dataset': dataset_name,
        'num_samples': len(y),
        'rmse': metrics_dict.get('val_mse', 0.0),
        'mape': metrics_dict.get('val_mape', 0.0),
        'median_q_error': metrics_dict.get('val_median_q_error_50', 0.0),
        'p95_q_error': metrics_dict.get('val_median_q_error_95', 0.0),
        'p99_q_error': metrics_dict.get('val_median_q_error_99', 0.0),
        'max_q_error': metrics_dict.get('val_median_q_error_100', 0.0)
    }
    
    print(f"\n📊 【{dataset_name}セット評価結果】")
    print(f"  - サンプル数: {len(y)}")
    print(f"  - RMSE: {results['rmse']:.4f}秒 ({results['rmse']*1000:.2f}ms)")
    print(f"  - MAPE: {results['mape']:.4f}")
    print(f"  - Median Q-Error: {results['median_q_error']:.4f}")
    print(f"  - P95 Q-Error: {results['p95_q_error']:.4f}")
    print(f"  - P99 Q-Error: {results['p99_q_error']:.4f}")
    print(f"  - Max Q-Error: {results['max_q_error']:.4f}")
    
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for Flat-Vector training."""
    parser = argparse.ArgumentParser(
        description='Train Trino Flat-Vector Model (Trino向け再実装版)'
    )
    
    # データ関連の引数
    parser.add_argument('--train_files', type=str, required=True,
                        help='トレーニング用ファイルパス（カンマ区切り）')
    parser.add_argument('--test_file', type=str, required=True,
                        help='テスト用ファイルパス')
    
    # モデル関連の引数
    parser.add_argument('--output_dir', type=str, default='models/trino_flat_vector',
                        help='モデル出力ディレクトリ')
    parser.add_argument('--num_boost_round', type=int, default=1000,
                        help='ブースティングラウンド数')
    parser.add_argument('--early_stopping_rounds', type=int, default=20,
                        help='早期停止ラウンド数')
    parser.add_argument('--seed', type=int, default=42,
                        help='ランダムシード')
    
    # データ処理関連の引数
    parser.add_argument('--max_plans_per_file', type=int, default=None,
                        help='各ファイルから読み込む最大プラン数')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='検証セットの割合')
    parser.add_argument('--use_act_card', action='store_true',
                        help='実際のカーディナリティを使用（デフォルト: 推定カーディナリティ）')
    
    return parser


def run(args) -> int:
    """Run Flat-Vector training with parsed arguments."""
    
    # ランダムシードの設定
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Trino Flat-Vector Model Training")
    print("（PostgreSQL用Flat-Vectorモデルのtrino向け再実装）")
    print("=" * 80)
    print(f"Train files: {args.train_files}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Use actual cardinality: {args.use_act_card}")
    print()
    
    # 1. プランの読み込み
    print("📂 ステップ1: プランの読み込み")
    train_file_paths = [Path(p.strip()) for p in args.train_files.split(',')]
    test_file_path = Path(args.test_file)
    
    # トレーニングプランの読み込み
    train_plans = load_trino_plans_from_files(train_file_paths, args.max_plans_per_file)
    
    # テストプランの読み込み
    test_plans = load_trino_plans_from_files([test_file_path], args.max_plans_per_file)
    
    print()
    
    # 2. 演算子タイプの収集
    print("📊 ステップ2: 演算子タイプの収集")
    op_idx_dict = collect_operator_types(train_plans)
    print()
    
    # 3. トレーニング/検証セットの分割
    print("📊 ステップ3: トレーニング/検証セットの分割")
    val_size = int(len(train_plans) * args.val_ratio)
    train_size = len(train_plans) - val_size
    
    # ランダムシャッフル
    indices = list(range(len(train_plans)))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_plans_split = [train_plans[i] for i in train_indices]
    val_plans_split = [train_plans[i] for i in val_indices]
    
    print(f"  - トレーニングプラン: {len(train_plans_split)}")
    print(f"  - 検証プラン: {len(val_plans_split)}")
    print(f"  - テストプラン: {len(test_plans)}")
    print()
    
    # 4. 特徴量の抽出
    print("🔧 ステップ4: 特徴量の抽出")
    
    print("  - トレーニングセット...")
    X_train, y_train = create_flat_vector_dataset(train_plans_split, op_idx_dict, args.use_act_card)
    
    print("  - 検証セット...")
    X_val, y_val = create_flat_vector_dataset(val_plans_split, op_idx_dict, args.use_act_card, verbose=False)
    
    print("  - テストセット...")
    X_test, y_test = create_flat_vector_dataset(test_plans, op_idx_dict, args.use_act_card, verbose=False)
    
    print(f"\n  - 特徴量次元数: {X_train.shape[1]}")
    print(f"  - トレーニングサンプル: {len(X_train)}")
    print(f"  - 検証サンプル: {len(X_val)}")
    print(f"  - テストサンプル: {len(X_test)}")
    print()
    
    # 5. モデルのトレーニング
    print("🚀 ステップ5: モデルのトレーニング")
    bst = train_flat_vector_model(
        X_train, y_train,
        X_val, y_val,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        verbose=True
    )
    
    # モデルの保存
    model_path = output_dir / f'flat_vector_model_{args.seed}.txt'
    bst.save_model(str(model_path))
    print(f"✅ モデルを保存: {model_path}")
    print()
    
    # 6. モデルの評価
    print("📊 ステップ6: モデルの評価")
    
    # トレーニングセットでの評価
    train_metrics = evaluate_with_metrics(bst, X_train, y_train, "Train")
    
    # 検証セットでの評価
    val_metrics = evaluate_with_metrics(bst, X_val, y_val, "Validation")
    
    # テストセットでの評価
    test_metrics = evaluate_with_metrics(bst, X_test, y_test, "Test")
    
    # 7. 結果の保存
    print("\n💾 ステップ7: 結果の保存")
    
    # 演算子インデックス辞書の保存
    op_idx_path = output_dir / f'op_idx_dict_{args.seed}.json'
    with open(op_idx_path, 'w') as f:
        json.dump(op_idx_dict, f, indent=2)
    print(f"  - 演算子インデックス辞書: {op_idx_path}")
    
    # メトリクスの保存
    metrics_path = output_dir / f'metrics_{args.seed}.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
            'hyperparameters': {
                'num_boost_round': args.num_boost_round,
                'early_stopping_rounds': args.early_stopping_rounds,
                'val_ratio': args.val_ratio,
                'use_act_card': args.use_act_card,
                'seed': args.seed
            }
        }, f, indent=2)
    print(f"  - メトリクス: {metrics_path}")
    
    print()
    print("=" * 80)
    print("トレーニング完了！")
    print(f"Validation Median Q-Error: {val_metrics['median_q_error']:.4f}")
    print(f"Test Median Q-Error: {test_metrics['median_q_error']:.4f}")
    print(f"Model saved to: {model_path}")
    print("=" * 80)
    
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for Flat-Vector training."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    from typing import Optional, Sequence
    import sys
    sys.exit(main())

