"""
Trino Flat-Vector Model Prediction Script

トレーニング済みのFlat-Vectorモデルを使用して、新しいクエリの実行時間を予測。
これは、既存のPostgreSQL用Flat-Vectorモデルをtrino向けに再実装したものです。

Usage:
    # ルートディレクトリから実行
    python -m trino_models.scripts.predict_flat_vector \
        --model_dir models/trino_flat_vector \
        --input_file accidents_valid_verbose.txt \
        --output_file predictions.json \
        --seed 42
"""

import sys
import os
from pathlib import Path
import argparse
import json
import lightgbm as lgb

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

from trino_models.models.flat_vector import (
    load_trino_plans_from_files,
    create_flat_vector_dataset,
    predict_flat_vector_model
)
from training.training.metrics import QError, RMSE, MAPE


def evaluate_predictions(y_true, y_pred):
    """
    既存のメトリクスを使用して予測結果を評価
    
    Args:
        y_true: 実際の値
        y_pred: 予測値
    
    Returns:
        評価メトリクス辞書
    """
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
            labels=y_true,
            preds=y_pred,
            probs=None
        )
    
    # 結果を整形
    results = {
        'num_samples': len(y_true),
        'rmse': metrics_dict.get('val_mse', 0.0),
        'mape': metrics_dict.get('val_mape', 0.0),
        'median_q_error': metrics_dict.get('val_median_q_error_50', 0.0),
        'p95_q_error': metrics_dict.get('val_median_q_error_95', 0.0),
        'p99_q_error': metrics_dict.get('val_median_q_error_99', 0.0),
        'max_q_error': metrics_dict.get('val_median_q_error_100', 0.0)
    }
    
    print(f"\n📊 【予測結果評価】")
    print(f"  - サンプル数: {len(y_true)}")
    print(f"  - RMSE: {results['rmse']:.4f}秒 ({results['rmse']*1000:.2f}ms)")
    print(f"  - MAPE: {results['mape']:.4f}")
    print(f"  - Median Q-Error: {results['median_q_error']:.4f}")
    print(f"  - P95 Q-Error: {results['p95_q_error']:.4f}")
    print(f"  - P99 Q-Error: {results['p99_q_error']:.4f}")
    print(f"  - Max Q-Error: {results['max_q_error']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Predict with Trino Flat-Vector Model (Trino向け再実装版)'
    )
    
    # モデル関連の引数
    parser.add_argument('--model_dir', type=str, required=True,
                        help='モデルディレクトリ')
    parser.add_argument('--seed', type=int, default=42,
                        help='ランダムシード（モデルファイル名に使用）')
    
    # データ関連の引数
    parser.add_argument('--input_file', type=str, required=True,
                        help='入力ファイルパス（クエリプラン）')
    parser.add_argument('--output_file', type=str, required=True,
                        help='出力ファイルパス（予測結果JSON）')
    parser.add_argument('--max_plans', type=int, default=None,
                        help='読み込む最大プラン数')
    parser.add_argument('--use_act_card', action='store_true',
                        help='実際のカーディナリティを使用（デフォルト: 推定カーディナリティ）')
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    print("=" * 80)
    print("Trino Flat-Vector Model Prediction")
    print("（PostgreSQL用Flat-Vectorモデルのtrino向け再実装）")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Use actual cardinality: {args.use_act_card}")
    print()
    
    # 1. モデルとメタデータの読み込み
    print("📂 ステップ1: モデルとメタデータの読み込み")
    
    # 演算子インデックス辞書の読み込み
    op_idx_path = model_dir / f'op_idx_dict_{args.seed}.json'
    with open(op_idx_path, 'r') as f:
        op_idx_dict = json.load(f)
    print(f"  - 演算子インデックス辞書: {op_idx_path}")
    print(f"  - 演算子タイプ数: {len(op_idx_dict)}")
    
    # モデルの読み込み
    model_path = model_dir / f'flat_vector_model_{args.seed}.txt'
    bst = lgb.Booster(model_file=str(model_path))
    print(f"  - モデル: {model_path}")
    print()
    
    # 2. プランの読み込み
    print("📂 ステップ2: プランの読み込み")
    plans = load_trino_plans_from_files([input_file], args.max_plans)
    print()
    
    # 3. 特徴量の抽出
    print("🔧 ステップ3: 特徴量の抽出")
    X, y_true = create_flat_vector_dataset(plans, op_idx_dict, args.use_act_card)
    print(f"  - 特徴量次元数: {X.shape[1]}")
    print(f"  - サンプル数: {len(X)}")
    print()
    
    # 4. 予測
    print("🚀 ステップ4: 予測")
    y_pred = predict_flat_vector_model(bst, X)
    print(f"  - 予測完了: {len(y_pred)}サンプル")
    print()
    
    # 5. 評価
    print("📊 ステップ5: 評価")
    metrics = evaluate_predictions(y_true, y_pred)
    print()
    
    # 6. 結果の保存
    print("💾 ステップ6: 結果の保存")
    
    # Q-Errorを計算
    import numpy as np
    q_errors = np.maximum(y_pred / np.maximum(y_true, 1e-6), y_true / np.maximum(y_pred, 1e-6))
    
    # 予測結果をJSON形式で保存
    results = {
        'metadata': {
            'model_dir': str(model_dir),
            'input_file': str(input_file),
            'num_samples': len(y_true),
            'use_act_card': args.use_act_card,
            'seed': args.seed
        },
        'metrics': metrics,
        'predictions': [
            {
                'index': i,
                'true_runtime_s': float(y_true[i]),
                'predicted_runtime_s': float(y_pred[i]),
                'true_runtime_ms': float(y_true[i] * 1000),
                'predicted_runtime_ms': float(y_pred[i] * 1000),
                'q_error': float(q_errors[i])
            }
            for i in range(len(y_true))
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  - 予測結果: {output_file}")
    print()
    
    print("=" * 80)
    print("予測完了！")
    print(f"Median Q-Error: {metrics['median_q_error']:.4f}")
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

