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
import warnings
from pathlib import Path
import argparse
import json
from typing import Optional, Sequence
import numpy as np
import torch

# Suppress torchdata deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchdata')

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
    
    # Q-Error計算のため、min_valを設定（0除算を防ぐ）
    # クエリプランの実行時間は100ms（0.1秒）～30秒の範囲
    min_val = 0.1  # 0.1秒 = 100ミリ秒
    
    # メトリクスの定義
    metrics = [
        RMSE(),
        MAPE(),
        QError(percentile=50, min_val=min_val, early_stopping_metric=True),
        QError(percentile=95, min_val=min_val),
        QError(percentile=99, min_val=min_val),
        QError(percentile=100, min_val=min_val)
    ]
    
    # 評価実行（QErrorクラス内でクリッピングが行われる）
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
    
    # モード選択
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'train_multi_all'],
        default='train',
        help='Training mode: train (single dataset) or train_multi_all (leave-one-out across all datasets)'
    )
    
    # データ関連の引数
    parser.add_argument('--train_files', type=str, required=False,
                        help='トレーニング用ファイルパス（カンマ区切り、trainモードで必須）')
    parser.add_argument('--test_file', type=str, required=False,
                        help='テスト用ファイルパス（trainモードで必須）')
    
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
    parser.add_argument(
        '--plans_dir',
        type=str,
        default='/Users/an/query_engine/explain_analyze_results/',
        help='Directory containing .txt plan files for multiple datasets (required for train_multi_all mode)'
    )
    
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
    print(f"Mode: {args.mode}")
    print("=" * 80)
    
    # train_multi_allモードの処理
    if args.mode == 'train_multi_all':
        return run_train_multi_all(args, output_dir)
    
    # 従来のtrainモード
    if not args.train_files or not args.test_file:
        raise ValueError("--train_files and --test_file are required for train mode")
    
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


def run_train_multi_all(args, output_dir: Path) -> int:
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
    print(f"Leave-One-Out Validation for All Datasets (Flat-Vector)")
    print(f"{'='*80}")
    print(f"利用可能なデータセット: {len(available_datasets)} / {len(ALL_DATASETS)}")
    print(f"データセット: {', '.join(available_datasets)}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"{'='*80}\n")
    
    # 最初に1回だけ全データセットのプランを読み込む
    def load_all_datasets_once_flat_vector(plans_dir: Path, available_datasets: list, max_plans_per_file=None):
        """全データセットのプランを1回だけ読み込む"""
        def infer_dataset_name(p: Path, ALL_DATASETS: list) -> str:
            stem = p.stem
            parts = stem.split('_')
            matched_dataset = None
            for i in range(len(parts), 0, -1):
                candidate = '_'.join(parts[:i])
                if candidate in ALL_DATASETS:
                    matched_dataset = candidate
                    break
            if matched_dataset:
                return matched_dataset
            return stem.split('_')[0]
        
        txt_files = sorted([p for p in plans_dir.glob('*.txt')])
        dataset_to_files = {}
        for p in txt_files:
            ds = infer_dataset_name(p, ALL_DATASETS)
            if ds in available_datasets:
                dataset_to_files.setdefault(ds, []).append(p)
        
        all_plans_by_dataset = {}
        print("=" * 80)
        print("ステップ0: 全データセットのプランを読み込み中...")
        print("=" * 80)
        print()
        
        for ds in available_datasets:
            if ds in dataset_to_files:
                files = dataset_to_files[ds]
                print(f"  読み込み中: {ds} ({len(files)} ファイル)...")
                plans = load_trino_plans_from_files(files, max_plans_per_file)
                all_plans_by_dataset[ds] = plans
                print(f"    ✅ {ds}: {len(plans)} プラン")
        
        print(f"\n✅ 全データセットの読み込み完了")
        print(f"  - 読み込んだデータセット: {len(all_plans_by_dataset)}")
        for ds, plans in all_plans_by_dataset.items():
            print(f"    - {ds}: {len(plans)} プラン")
        print()
        
        return all_plans_by_dataset
    
    all_plans_by_dataset = load_all_datasets_once_flat_vector(
        plans_dir=plans_dir,
        available_datasets=available_datasets,
        max_plans_per_file=args.max_plans_per_file
    )
    
    # 全データセットから演算子タイプを事前に収集（未知の演算子タイプを避けるため）
    print(f"\n{'='*80}")
    print("📊 全データセットから演算子タイプを収集中...")
    print(f"{'='*80}")
    all_plans = []
    for plans in all_plans_by_dataset.values():
        all_plans.extend(plans)
    global_op_idx_dict = collect_operator_types(all_plans)
    print()
    
    # 各データセットについて訓練・テストを実行
    results_summary = []
    
    for idx, test_dataset in enumerate(available_datasets, 1):
        print(f"\n{'#'*80}")
        print(f"# [{idx}/{len(available_datasets)}] Testing dataset: {test_dataset}")
        print(f"{'#'*80}\n")
        
        try:
            # 既に読み込んだプランからtrain/testを分割
            if test_dataset not in all_plans_by_dataset:
                print(f"⚠️  {test_dataset}: プランが見つかりません。スキップします。")
                results_summary.append({
                    'test_dataset': test_dataset,
                    'status': 'skipped',
                    'reason': 'missing plans'
                })
                continue
            
            train_plans = []
            test_plans = all_plans_by_dataset[test_dataset]
            
            for ds, plans in all_plans_by_dataset.items():
                if ds != test_dataset:
                    train_plans.extend(plans)
            
            if not train_plans or not test_plans:
                print(f"⚠️  {test_dataset}: 訓練プランまたはテストプランが見つかりません。スキップします。")
                results_summary.append({
                    'test_dataset': test_dataset,
                    'status': 'skipped',
                    'reason': 'missing plans'
                })
                continue
            
            # モデルディレクトリ
            model_dir = output_dir / f'models_{test_dataset}'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📊 Leave-One-Out Validation [{idx}/{len(available_datasets)}]:")
            print(f"  - Training datasets: {len(all_plans_by_dataset) - 1} datasets")
            print(f"  - Training plans: {len(train_plans)}")
            print(f"  - Test dataset: {test_dataset}")
            print(f"  - Test plans: {len(test_plans)}")
            print()
            
            # 演算子タイプの辞書は全データセットから事前に収集したものを使用
            op_idx_dict = global_op_idx_dict
            
            # トレーニング/検証セットの分割（19個のデータセットをtrain/valに分割）
            val_size = int(len(train_plans) * args.val_ratio)
            train_size = len(train_plans) - val_size
            
            indices = list(range(len(train_plans)))
            np.random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            train_plans_split = [train_plans[i] for i in train_indices]
            val_plans_split = [train_plans[i] for i in val_indices]
            
            print(f"✅ 19個のデータセットから作成:")
            print(f"  - Train plans: {len(train_plans_split)}")
            print(f"  - Val plans (from 19 datasets): {len(val_plans_split)}")
            print()
            
            # 特徴量の抽出
            print("🔧 特徴量の抽出...")
            X_train, y_train = create_flat_vector_dataset(train_plans_split, op_idx_dict, args.use_act_card, verbose=False)
            X_val, y_val = create_flat_vector_dataset(val_plans_split, op_idx_dict, args.use_act_card, verbose=False)
            X_test, y_test = create_flat_vector_dataset(test_plans, op_idx_dict, args.use_act_card, verbose=False)
            
            print(f"  - 特徴量次元数: {X_train.shape[1]}")
            print(f"  - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            print()
            
            # モデルのトレーニング
            print("🚀 モデルのトレーニング...")
            bst = train_flat_vector_model(
                X_train, y_train,
                X_val, y_val,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
                seed=args.seed,
                verbose=False
            )
            
            # モデルの保存
            model_path = model_dir / f'flat_vector_model_{args.seed}.txt'
            bst.save_model(str(model_path))
            
            # 演算子インデックス辞書の保存
            op_idx_path = model_dir / f'op_idx_dict_{args.seed}.json'
            with open(op_idx_path, 'w') as f:
                json.dump(op_idx_dict, f, indent=2)
            
            # モデルの評価
            val_metrics = evaluate_with_metrics(bst, X_val, y_val, "Validation")
            test_metrics = evaluate_with_metrics(bst, X_test, y_test, "Test")
            
            # テスト結果を保存
            test_results = {
                'test_median_q_error': float(test_metrics['median_q_error']),
                'test_mean_q_error': float(np.mean([
                    test_metrics.get('p95_q_error', 0),
                    test_metrics.get('p99_q_error', 0),
                    test_metrics.get('max_q_error', 0)
                ])) if any(k in test_metrics for k in ['p95_q_error', 'p99_q_error', 'max_q_error']) else None,
                'test_rmse': float(test_metrics.get('rmse', 0)),
                'test_samples': len(test_plans)
            }
            
            results_file = model_dir / 'test_results.json'
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            results_summary.append({
                'test_dataset': test_dataset,
                'model_dir': str(model_dir),
                'val_median_q_error': float(val_metrics['median_q_error']),
                **test_results,
                'status': 'completed'
            })
            
            print(f"✅ [{idx}/{len(available_datasets)}] {test_dataset} の訓練・テスト完了")
            print(f"   Validation Median Q-Error: {val_metrics['median_q_error']:.4f}")
            print(f"   Test Median Q-Error: {test_metrics['median_q_error']:.4f}")
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
    print(f"サマリーファイル: {summary_file}")
    print()
    
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

