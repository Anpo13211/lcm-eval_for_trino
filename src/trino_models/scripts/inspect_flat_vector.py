"""
Trino Flat-Vector Model Inspector

トレーニング済みのFlat-Vectorモデルの詳細情報を表示。
これは、既存のPostgreSQL用Flat-Vectorモデルをtrino向けに再実装したものです。

Usage:
    # ルートディレクトリから実行
    python -m trino_models.scripts.inspect_flat_vector \
        --model_dir models/trino_flat_vector --seed 42
"""

import sys
import os
from pathlib import Path
import argparse
import json

# 環境変数の設定（必須 - import前に実行）
for i in range(11):
    env_key = f'NODE{i:02d}'
    env_value = os.environ.get(env_key)
    if env_value in (None, '', 'None'):
        os.environ[env_key] = '[]'

# スクリプトがsrc/trino_models/scripts/にある場合、src/を親パスに追加
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import lightgbm as lgb


def inspect_model(model_dir: Path, seed: int):
    """
    モデルの詳細情報を表示
    
    Args:
        model_dir: モデルディレクトリ
        seed: ランダムシード
    """
    print("=" * 80)
    print("Trino Flat-Vector Model Inspector")
    print("（PostgreSQL用Flat-Vectorモデルのtrino向け再実装）")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Seed: {seed}")
    print()
    
    # 1. 演算子インデックス辞書の読み込み
    print("📊 演算子タイプ情報")
    print("-" * 80)
    
    op_idx_path = model_dir / f'op_idx_dict_{seed}.json'
    if not op_idx_path.exists():
        print(f"❌ 演算子インデックス辞書が見つかりません: {op_idx_path}")
        return
    
    with open(op_idx_path, 'r') as f:
        op_idx_dict = json.load(f)
    
    print(f"演算子タイプ数: {len(op_idx_dict)}")
    print("\n演算子タイプ一覧:")
    for op_name, idx in sorted(op_idx_dict.items(), key=lambda x: x[1]):
        print(f"  {idx:3d}: {op_name}")
    print()
    
    # 2. モデルの読み込み
    print("🤖 モデル情報")
    print("-" * 80)
    
    model_path = model_dir / f'flat_vector_model_{seed}.txt'
    if not model_path.exists():
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return
    
    bst = lgb.Booster(model_file=str(model_path))
    
    print(f"モデルファイル: {model_path}")
    print(f"ブースティングイテレーション数: {bst.current_iteration()}")
    print(f"ベストイテレーション: {bst.best_iteration}")
    print(f"特徴量数: {bst.num_feature()}")
    print()
    
    # 特徴量重要度
    feature_importance = bst.feature_importance(importance_type='gain')
    feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
    
    # 特徴量名を推測（前半は出現回数、後半はカーディナリティ）
    feature_type_names = []
    for i in range(len(feature_importance)):
        if i < len(op_idx_dict):
            # 演算子の出現回数特徴
            op_name = [name for name, idx in op_idx_dict.items() if idx == i][0]
            feature_type_names.append(f"Count_{op_name}")
        else:
            # 演算子のカーディナリティ特徴
            op_idx = i - len(op_idx_dict)
            op_name = [name for name, idx in op_idx_dict.items() if idx == op_idx][0]
            feature_type_names.append(f"Card_{op_name}")
    
    # 重要度でソート
    feature_importance_sorted = sorted(
        zip(feature_type_names, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("特徴量重要度（Top 10）:")
    for i, (feat_name, importance) in enumerate(feature_importance_sorted[:10], 1):
        print(f"  {i:2d}. {feat_name:30s}: {importance:10.2f}")
    print()
    
    # 3. メトリクスの読み込み
    print("📊 トレーニングメトリクス")
    print("-" * 80)
    
    metrics_path = model_dir / f'metrics_{seed}.json'
    if not metrics_path.exists():
        print(f"⚠️  メトリクスファイルが見つかりません: {metrics_path}")
        return
    
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    # ハイパーパラメータ
    print("ハイパーパラメータ:")
    for key, value in metrics_data.get('hyperparameters', {}).items():
        print(f"  - {key}: {value}")
    print()
    
    # トレーニング/検証/テストメトリクス
    for dataset_name in ['train', 'validation', 'test']:
        if dataset_name not in metrics_data:
            continue
        
        metrics = metrics_data[dataset_name]
        print(f"【{dataset_name.capitalize()}セット】")
        print(f"  - サンプル数: {metrics['num_samples']}")
        print(f"  - RMSE: {metrics['rmse']:.4f}秒 ({metrics['rmse']*1000:.2f}ms)")
        print(f"  - MAPE: {metrics['mape']:.4f}")
        print(f"  - Median Q-Error: {metrics['median_q_error']:.4f}")
        print(f"  - P95 Q-Error: {metrics['p95_q_error']:.4f}")
        print(f"  - P99 Q-Error: {metrics.get('p99_q_error', 0.0):.4f}")
        print(f"  - Max Q-Error: {metrics['max_q_error']:.4f}")
        print()
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Inspect Trino Flat-Vector Model (Trino向け再実装版)'
    )
    
    parser.add_argument('--model_dir', type=str, required=True,
                        help='モデルディレクトリ')
    parser.add_argument('--seed', type=int, default=42,
                        help='ランダムシード（デフォルト: 42）')
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"❌ モデルディレクトリが見つかりません: {model_dir}")
        return
    
    inspect_model(model_dir, args.seed)


if __name__ == "__main__":
    main()

