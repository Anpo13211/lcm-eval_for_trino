"""
Unified Flat Model Training Script with Leave-One-Out Cross-Validation

This script performs leave-one-out cross-validation across 20 datasets.
For each iteration, it trains on 19 datasets and tests on 1 held-out dataset.

Usage:
    python -m training.scripts.train_unified_flat_loo \
        --dbms trino \
        --data_dir /Users/an/query_engine/explain_analyze_results \
        --output_dir models/flat_loo
"""

import sys
import os
import warnings
from itertools import chain
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add src to path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse
import json
import numpy as np
import lightgbm as lgb
from datetime import datetime

# Initialize plugin system
import core.init_plugins

from core.plugins.registry import DBMSRegistry
from models.flat.unified_flat_feature_extractor import (
    extract_batch_features_unified,
    build_operator_vocab,
)
from training.training.metrics import QError, RMSE
from classes.classes import FlatModelConfig
from core.capabilities import check_capabilities


# Dataset names (in order)
DATASET_NAMES = [
    'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
    'consumer', 'credit', 'employee', 'fhnk', 'financial',
    'geneea', 'genome', 'hepatitis', 'imdb', 'movielens',
    'seznam', 'ssb', 'tournament', 'tpc_h', 'walmart'
]


def load_all_plans_once(data_dir: Path, dbms_name: str, max_plans_per_dataset: int = None):
    """
    Load all plans from all datasets once and cache them.
    
    Returns:
        dict: {dataset_name: list_of_plans}
    """
    parser = DBMSRegistry.get_parser(dbms_name)
    all_dataset_plans = {}
    
    print("📂 Loading all datasets (one-time operation)")
    print("="*80)
    
    for dataset_name in DATASET_NAMES:
        file_path = data_dir / f"{dataset_name}_complex_workload_200k_s1_explain_analyze.txt"
        
        if not file_path.exists():
            print(f"⚠️  {dataset_name}: File not found, skipping")
            continue
        
        print(f"  Loading {dataset_name}...")
        
        try:
            parsed_plans, runtimes = parser.parse_explain_analyze_file(
                str(file_path),
                min_runtime=0,
                max_runtime=float('inf')
            )
            
            # Set plan_runtime
            for plan, runtime in zip(parsed_plans, runtimes):
                plan.plan_runtime = runtime
                plan.dataset_name = dataset_name
            
            # Limit if needed
            if max_plans_per_dataset and len(parsed_plans) > max_plans_per_dataset:
                parsed_plans = parsed_plans[:max_plans_per_dataset]
            
            all_dataset_plans[dataset_name] = parsed_plans
            
        except Exception as e:
            print(f"    ⚠️  Failed to load {dataset_name}: {e}")
            continue
    
    # Calculate total plans
    total_plans = sum(len(plans) for plans in all_dataset_plans.values())
    print(f"✓ Loaded {total_plans} plans from {len(all_dataset_plans)} datasets")
    print("="*80)
    print()
    
    return all_dataset_plans


def evaluate_test(model, X_test, y_test, test_dataset_name):
    """Evaluate model on test set with detailed metrics."""
    y_pred = model.predict(X_test)
    
    min_val = 0.1
    safe_preds = np.clip(y_pred, min_val, np.inf)
    safe_labels = np.clip(y_test, min_val, np.inf)
    
    # Calculate metrics
    rmse = RMSE().evaluate_metric(labels=y_test, preds=y_pred)
    median_q_error = QError(percentile=50, min_val=min_val).evaluate_metric(labels=safe_labels, preds=safe_preds)
    mean_q_error = float(np.mean(np.maximum(safe_preds / safe_labels, safe_labels / safe_preds)))
    p95_q_error = QError(percentile=95, min_val=min_val).evaluate_metric(labels=safe_labels, preds=safe_preds)
    p99_q_error = QError(percentile=99, min_val=min_val).evaluate_metric(labels=safe_labels, preds=safe_preds)
    max_q_error = QError(percentile=100, min_val=min_val).evaluate_metric(labels=safe_labels, preds=safe_preds)
    
    print(f"\n  📊 Test Results for {test_dataset_name}")
    print(f"  {'─'*70}")
    print(f"    Samples: {len(y_test)}")
    print(f"    RMSE: {rmse:.4f} sec")
    print(f"    Median Q-Error: {median_q_error:.4f}")
    print(f"    Mean Q-Error: {mean_q_error:.4f}")
    print(f"    P95 Q-Error: {p95_q_error:.4f}")
    print(f"    P99 Q-Error: {p99_q_error:.4f}")
    print(f"    Max Q-Error: {max_q_error:.4f}")
    
    return {
        'dataset': test_dataset_name,
        'rmse': float(rmse) if rmse is not None else None,
        'median_q_error': float(median_q_error) if median_q_error is not None else None,
        'mean_q_error': float(mean_q_error) if mean_q_error is not None else None,
        'p95_q_error': float(p95_q_error) if p95_q_error is not None else None,
        'p99_q_error': float(p99_q_error) if p99_q_error is not None else None,
        'max_q_error': float(max_q_error) if max_q_error is not None else None,
        'num_samples': int(len(y_test))
    }


def run_leave_one_out(
    dbms_name: str,
    data_dir: Path,
    output_dir: Path,
    max_plans_per_dataset: int = None,
    num_boost_round: int = 100,
    learning_rate: float = 0.05
):
    """
    Run leave-one-out cross-validation across all datasets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"LEAVE-ONE-OUT CROSS-VALIDATION - Flat Vector Model ({dbms_name.upper()})")
    print("="*80)
    print(f"Total Datasets: {len(DATASET_NAMES)}")
    print(f"Boosting rounds: {num_boost_round}")
    print("="*80)
    print()

    # Capability check
    try:
        plugin = DBMSRegistry.get_plugin(dbms_name)
        provided_caps = plugin.get_capabilities()
        temp_config = FlatModelConfig()
        required_caps = temp_config.required_capabilities
        missing_caps = check_capabilities(
            required_caps,
            provided_caps,
            temp_config.name.NAME if temp_config.name else "flat_vector",
            dbms_name
        )

        if missing_caps:
            print("="*80)
            print(f"⚠️  WARNING: DBMS '{dbms_name}' is missing capabilities required by Flat model: {missing_caps}")
            print("    Training may fail or produce suboptimal results.")
            print("="*80)
        else:
            print(f"✓ Capability check passed: {dbms_name} provides all required capabilities for Flat model.\n")
    except Exception as e:
        print(f"⚠️  Capability check could not be completed: {e}")
    
    # Step 1: Load all plans once
    all_dataset_plans = load_all_plans_once(data_dir, dbms_name, max_plans_per_dataset)
    
    flat_operator_vocab = build_operator_vocab(
        plans=chain.from_iterable(all_dataset_plans.values()),
        dbms_name=dbms_name
    )
    feature_config = {'flat_vector_operator_vocab': flat_operator_vocab}
    print(f"Detected {len(flat_operator_vocab)} operator slots (including fallback bucket).")
    
    # Step 2: Extract features for all plans (one-time operation)
    print("🔧 Extracting features from all plans (one-time operation)")
    print("="*80)
    
    all_features = {}
    all_labels = {}
    
    for dataset_name, plans in all_dataset_plans.items():
        print(f"  Extracting features for {dataset_name}...")
        X = extract_batch_features_unified(plans, dbms_name, feature_config)
        y = np.array([p.plan_runtime / 1000 for p in plans])  # Convert to seconds
        
        all_features[dataset_name] = X
        all_labels[dataset_name] = y
        print(f"    ✓ Shape: {X.shape}, Labels: {len(y)}")
    
    print(f"\n✓ Feature extraction complete")
    print("="*80)
    print()
    
    # Step 3: Leave-one-out loop
    all_results = []
    
    for fold_idx, test_dataset_name in enumerate(DATASET_NAMES):
        if test_dataset_name not in all_dataset_plans:
            print(f"⚠️  Fold {fold_idx+1}/{len(DATASET_NAMES)}: {test_dataset_name} not found, skipping")
            continue
        
        print("\n" + "="*80)
        print(f"FOLD {fold_idx+1}/{len(DATASET_NAMES)}: Test on {test_dataset_name}")
        print("="*80)
        
        # Prepare train and test data
        X_train_list = []
        y_train_list = []
        
        for dataset_name in all_dataset_plans.keys():
            if dataset_name != test_dataset_name:
                X_train_list.append(all_features[dataset_name])
                y_train_list.append(all_labels[dataset_name])
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        X_test = all_features[test_dataset_name]
        y_test = all_labels[test_dataset_name]
        
        print(f"  Train samples: {len(y_train)} (from {len(all_dataset_plans)-1} datasets)")
        print(f"  Test samples: {len(y_test)} (from {test_dataset_name})")
        print(f"  Feature dimension: {X_train.shape[1]}")
        
        # Train LightGBM
        print(f"\n  🚀 Training...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': learning_rate,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        
        model = lgb.train(params, train_data, num_boost_round=num_boost_round)
        
        # Evaluate on test set
        test_metrics = evaluate_test(model, X_test, y_test, test_dataset_name)
        
        if test_metrics:
            all_results.append(test_metrics)
        
        # Clean up
        del model
    
    # Save all results
    results_file = output_dir / f"{dbms_name}_leave_one_out_results.json"
    
    # Calculate average metrics
    if all_results:
        avg_metrics = {
            'avg_rmse': np.mean([r['rmse'] for r in all_results if r['rmse'] is not None]),
            'avg_median_q_error': np.mean([r['median_q_error'] for r in all_results if r['median_q_error'] is not None]),
            'avg_mean_q_error': np.mean([r['mean_q_error'] for r in all_results if r['mean_q_error'] is not None]),
            'avg_p95_q_error': np.mean([r['p95_q_error'] for r in all_results if r['p95_q_error'] is not None]),
            'avg_p99_q_error': np.mean([r['p99_q_error'] for r in all_results if r['p99_q_error'] is not None]),
        }
        
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'dbms': dbms_name,
            'model': 'flat_vector',
            'num_folds': len(all_results),
            'num_boost_round': num_boost_round,
            'learning_rate': learning_rate,
            'average_metrics': avg_metrics,
            'per_dataset_results': all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("\n" + "="*80)
        print("📊 LEAVE-ONE-OUT CROSS-VALIDATION SUMMARY")
        print("="*80)
        print(f"Completed folds: {len(all_results)}/{len(DATASET_NAMES)}")
        print(f"\nAverage Metrics:")
        print(f"  RMSE: {avg_metrics['avg_rmse']:.4f}")
        print(f"  Median Q-Error: {avg_metrics['avg_median_q_error']:.4f}")
        print(f"  Mean Q-Error: {avg_metrics['avg_mean_q_error']:.4f}")
        print(f"  P95 Q-Error: {avg_metrics['avg_p95_q_error']:.4f}")
        print(f"  P99 Q-Error: {avg_metrics['avg_p99_q_error']:.4f}")
        print(f"\n✓ Results saved to: {results_file}")
        print("="*80)
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Leave-One-Out Cross-Validation for Flat Vector Model')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing all dataset files')
    # Get available DBMS from registry
    available_dbms = DBMSRegistry.get_cli_choices() if DBMSRegistry.list_plugins() else ['trino', 'postgres', 'mysql']
    parser.add_argument('--dbms', type=str, default='trino',
                       choices=available_dbms,
                       help='DBMS name')
    parser.add_argument('--output_dir', type=str, default='models/flat_loo',
                       help='Output directory')
    parser.add_argument('--max_plans', type=int, default=10000,
                       help='Maximum plans per dataset')
    parser.add_argument('--num_boost_round', type=int, default=1000,
                       help='Number of boosting rounds')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    return run_leave_one_out(
        dbms_name=args.dbms,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        max_plans_per_dataset=args.max_plans,
        num_boost_round=args.num_boost_round,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    sys.exit(main())

