"""
Unified Flat Model Training Script

Simple XGBoost/LightGBM model with flat features.
Works for all DBMS.

Usage:
    python -m training.scripts.train_unified_flat \
        --dbms trino \
        --train_files accidents.txt \
        --test_file accidents.txt \
        --schema accidents
"""

import sys
import os
from pathlib import Path

# Add src to path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse
import numpy as np
import lightgbm as lgb

# Initialize plugin system
import core.init_plugins

from core.plugins.registry import DBMSRegistry
from core.statistics.schema import StandardizedStatistics
from models.flat.unified_flat_feature_extractor import (
    extract_batch_features_unified,
    build_operator_vocab,
)
from training.training.metrics import QError, RMSE, MAPE


def load_plans_from_txt(file_paths: list, dbms_name: str, max_plans: int = None):
    """Load plans from txt files."""
    parser = DBMSRegistry.get_parser(dbms_name)
    
    all_plans = []
    for file_path in file_paths:
        print(f"  Loading {Path(file_path).name}...")
        parsed_plans, runtimes = parser.parse_explain_analyze_file(
            str(file_path), min_runtime=0, max_runtime=float('inf')
        )
        
        for plan, runtime in zip(parsed_plans, runtimes):
            plan.plan_runtime = runtime
        
        if max_plans and len(parsed_plans) > max_plans:
            parsed_plans = parsed_plans[:max_plans]
        
        all_plans.extend(parsed_plans)
    
    return all_plans


def run_training(
    dbms_name: str,
    train_files: list,
    test_files: list,
    schema_name: str,
    output_dir: Path,
    max_plans: int = None
):
    """Train Flat model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"Unified Flat Model Training for {dbms_name.upper()}")
    print("="*80)
    print()
    
    # Load plans
    print("📂 Loading plans")
    train_plans = load_plans_from_txt(train_files, dbms_name, max_plans)
    test_plans = load_plans_from_txt(test_files, dbms_name, max_plans)
    print(f"✓ Train: {len(train_plans)}, Test: {len(test_plans)}")
    print()
    
    # Build operator vocabulary from all observed plans
    flat_operator_vocab = build_operator_vocab(train_plans + test_plans, dbms_name)
    feature_config = {'flat_vector_operator_vocab': flat_operator_vocab}
    
    # Extract features
    print("🔧 Extracting features")
    X_train = extract_batch_features_unified(train_plans, dbms_name, feature_config)
    y_train = np.array([p.plan_runtime / 1000 for p in train_plans])  # Convert to sec
    
    X_test = extract_batch_features_unified(test_plans, dbms_name, feature_config)
    y_test = np.array([p.plan_runtime / 1000 for p in test_plans])
    
    print(f"✓ Features: {X_train.shape}")
    print()
    
    # Train
    print("🚀 Training LightGBM")
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    model = lgb.train(params, train_data, num_boost_round=100)
    
    # Evaluate
    print("\n📊 Evaluation")
    y_pred = model.predict(X_test)
    
    rmse = RMSE().evaluate_metric(y_test, y_pred)
    median_q = QError(percentile=50, min_val=0.1).evaluate_metric(
        np.clip(y_test, 0.1, np.inf),
        np.clip(y_pred, 0.1, np.inf)
    )
    
    print(f"  RMSE: {rmse:.4f} sec")
    print(f"  Median Q-Error: {median_q:.4f}")
    
    # Save
    model.save_model(str(output_dir / f"{dbms_name}_flat_model.txt"))
    print(f"\n✓ Model saved to {output_dir}")
    
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--schema', type=str, required=True)
    # Get available DBMS from registry
    available_dbms = DBMSRegistry.get_cli_choices() if DBMSRegistry.list_plugins() else ['trino', 'postgres', 'mysql']
    parser.add_argument('--dbms', type=str, default='trino', choices=available_dbms)
    parser.add_argument('--output_dir', type=str, default='models/unified_flat')
    parser.add_argument('--max_plans', type=int, default=10000)
    
    args = parser.parse_args()
    
    return run_training(
        dbms_name=args.dbms,
        train_files=[Path(p.strip()) for p in args.train_files.split(',')],
        test_files=[Path(args.test_file)],
        schema_name=args.schema,
        output_dir=Path(args.output_dir),
        max_plans=args.max_plans
    )


if __name__ == "__main__":
    sys.exit(main())

