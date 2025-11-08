"""
Unified QueryFormer Training Script with Leave-One-Out Cross-Validation

This script performs leave-one-out cross-validation across 20 datasets.
For each iteration, it trains on 19 datasets and tests on 1 held-out dataset.

Usage:
    python -m training.scripts.train_unified_queryformer_loo \
        --dbms trino \
        --data_dir /Users/an/query_engine/explain_analyze_results \
        --output_dir models/queryformer_loo
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchdata')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add src to path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse
import functools
import torch
import json
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import numpy as np
from datetime import datetime
from types import SimpleNamespace

# Initialize plugin system
import core.init_plugins

from core.plugins.registry import DBMSRegistry
from core.statistics.schema import StandardizedStatistics
from models.query_former.dataloader import query_former_plan_collator
from models.query_former.model import QueryFormer
from classes.classes import QueryFormerModelConfig
from models.workload_driven.dataset.dataset_creation import PlanModelInputDims
from training.training.metrics import QError, RMSE


# Dataset names (in order)
DATASET_NAMES = [
    'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
    'consumer', 'credit', 'employee', 'fhnk', 'financial',
    'geneea', 'genome', 'hepatitis', 'imdb', 'movielens',
    'seznam', 'ssb', 'tournament', 'tpc_h', 'walmart'
]


class UnifiedPlanDataset(Dataset):
    def __init__(self, plans):
        self.plans = plans
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx):
        return idx, self.plans[idx]


def load_table_samples_for_dataset(schema_name: str, no_samples: int = 1000):
    """Load table samples from CSV files for sample_vec generation."""
    import pandas as pd
    from cross_db_benchmark.benchmark_tools.utils import load_schema_json
    
    # Try zero-shot-data location
    data_dirs = [
        f'/Users/an/query_engine/lakehouse/zero-shot_datasets/{schema_name}',
        f'../zero-shot-data/datasets/{schema_name}',
        f'data/{schema_name}'
    ]
    
    try:
        schema = load_schema_json(schema_name, prefer_zero_shot=True)
        table_samples = {}
        
        # Get CSV reading kwargs from schema
        csv_kwargs = vars(schema.csv_kwargs) if hasattr(schema, 'csv_kwargs') else {}
        # Add parameters to suppress warnings
        csv_kwargs.update({
            'low_memory': False,
            'on_bad_lines': 'skip',  # Suppress "Skipping line" warnings
            'encoding_errors': 'ignore'
        })
        
        for table_name in schema.tables:
            df = None
            for data_dir in data_dirs:
                csv_path = Path(data_dir) / f'{table_name}.csv'
                if csv_path.exists():
                    try:
                        # Suppress DtypeWarning during CSV reading
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            df = pd.read_csv(csv_path, **csv_kwargs)
                        if len(df) > no_samples:
                            df = df.sample(random_state=0, n=no_samples)
                        break
                    except Exception:
                        pass
            
            if df is not None:
                table_samples[table_name] = df
        
        return table_samples if table_samples else None
    except Exception:
        return None


def load_all_table_samples_once(no_samples: int = 1000):
    """Load table samples for all datasets once and cache them."""
    print("📦 Loading table samples for all datasets (one-time operation)")
    print("="*80)
    
    all_table_samples = {}
    
    for dataset_name in DATASET_NAMES:
        print(f"  Loading table samples for {dataset_name}...")
        table_samples = load_table_samples_for_dataset(dataset_name, no_samples)
        if table_samples:
            all_table_samples[dataset_name] = table_samples
            print(f"    ✓ Loaded {len(table_samples)} tables")
        else:
            print(f"    ⚠️  No table samples found")
    
    print(f"\n✓ Loaded table samples for {len(all_table_samples)} datasets")
    print("="*80)
    print()
    
    return all_table_samples


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
            
            # Set plan_runtime and database_id
            for plan, runtime in zip(parsed_plans, runtimes):
                plan.plan_runtime = runtime
                plan.database_id = 0
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


def load_all_statistics_once(dbms_name: str, statistics_dir: str = None):
    """
    Load all database statistics once and cache them.
    
    Returns:
        dict: {dataset_name: {'db_stats': StandardizedStatistics, 'column_stats': dict, 'col_stats_list': list}}
    """
    # Use absolute path from project root if relative path provided
    if statistics_dir is None or not Path(statistics_dir).is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        statistics_dir = str(project_root / (statistics_dir or 'datasets_statistics'))
    
    print("📊 Loading all database statistics (one-time operation)")
    print("="*80)
    
    all_stats = {}
    
    for dataset_name in DATASET_NAMES:
        try:
            # Load database statistics
            if dbms_name == "trino":
                from models.zeroshot.utils.statistics_loader import load_database_statistics_for_zeroshot
                raw_stats = load_database_statistics_for_zeroshot(
                    dataset=dataset_name,
                    stats_dir=statistics_dir,
                    prefer_zero_shot=True
                )
            else:
                from models.zeroshot.utils.statistics_loader import load_database_statistics_for_zeroshot
                raw_stats = load_database_statistics_for_zeroshot(
                    dataset=dataset_name,
                    stats_dir=statistics_dir
                )
            
            converter = DBMSRegistry.get_statistics_converter(dbms_name)
            standardized = converter.convert(raw_stats)
            
            # Load column statistics
            col_stats_path = Path(statistics_dir) / f"iceberg_{dataset_name}" / 'column_stats.json'
            column_statistics = {}
            col_stats_list = []
            
            if col_stats_path.exists():
                from cross_db_benchmark.benchmark_tools.utils import load_json
                column_statistics = load_json(str(col_stats_path), namespace=False)
                
                # Convert column_statistics to list format for augment_sample
                import ast
                for key, val in column_statistics.items():
                    # Key can be tuple or string "(table, column)"
                    if isinstance(key, tuple) and len(key) == 2:
                        table_name, col_name = key
                    elif isinstance(key, str) and key.startswith('('):
                        # Parse string like "('table', 'column')"
                        try:
                            table_name, col_name = ast.literal_eval(key)
                        except:
                            continue
                    else:
                        continue
                    
                    val_copy = dict(val)
                    val_copy['tablename'] = table_name
                    val_copy['attname'] = col_name
                    col_stats_list.append(SimpleNamespace(**val_copy))
            
            all_stats[dataset_name] = {
                'db_stats': standardized,
                'column_stats': column_statistics,
                'col_stats_list': col_stats_list
            }
            
            print(f"  ✓ {dataset_name}: {len(standardized.column_stats)} columns, {len(standardized.table_stats)} tables")
            
        except Exception as e:
            print(f"  ⚠️  {dataset_name}: Statistics loading failed - {e}")
            all_stats[dataset_name] = {
                'db_stats': StandardizedStatistics(),
                'column_stats': {},
                'col_stats_list': []
            }
    
    print(f"\n✓ Loaded statistics for {len(all_stats)} datasets")
    print("="*80)
    print()
    
    return all_stats


def build_feature_statistics(plans, db_stats, column_stats, dbms_name: str):
    """Build feature statistics by combining plan-derived and DB statistics."""
    from sklearn.preprocessing import RobustScaler
    from core.features.mapper import FeatureMapper

    mapper = FeatureMapper(dbms_name)

    # Collect categorical vocabularies
    op_names = set()
    filter_ops = set()
    agg_ops = set()
    join_conds = set()

    # Collect numeric values
    numeric_values = {
        'estimated_cardinality': [],
        'estimated_cost': [],
        'estimated_width': [],
    }

    def walk(node):
        if not hasattr(node, 'plan_parameters'):
            return
        params = node.plan_parameters

        # operator names
        op = params.get('op_name') if isinstance(params, dict) else getattr(params, 'op_name', None)
        if op:
            op_names.add(str(op))

        # numeric features via FeatureMapper
        for k in numeric_values.keys():
            try:
                v = mapper.get_feature(k, params)
                if v is not None and not isinstance(v, str):
                    numeric_values[k].append(float(v))
            except Exception:
                pass

        # filter operators
        filt = params.get('filter_columns') if isinstance(params, dict) else getattr(params, 'filter_columns', None)
        if filt:
            def rec(fc):
                op = fc.get('operator') if isinstance(fc, dict) else getattr(fc, 'operator', None)
                if op is not None:
                    filter_ops.add(str(op))
                children = fc.get('children', []) if isinstance(fc, dict) else getattr(fc, 'children', [])
                for c in children:
                    rec(c)
            rec(filt)

        # aggregations
        outs = params.get('output_columns') if isinstance(params, dict) else getattr(params, 'output_columns', None)
        if outs:
            for oc in outs:
                agg = oc.get('aggregation') if isinstance(oc, dict) else getattr(oc, 'aggregation', None)
                if agg is not None:
                    agg_ops.add(str(agg))

        # children
        for ch in getattr(node, 'children', []) or []:
            walk(ch)

    for p in plans:
        walk(p)

    # Tables from db_stats
    table_names = list(getattr(db_stats, 'table_stats', {}).keys()) if db_stats else []

    # Numeric scalers
    feature_statistics = {}
    for feat, vals in numeric_values.items():
        if len(vals) == 0:
            feature_statistics[feat] = dict(type='numeric', max=1.0, center=0.0, scale=1.0)
        else:
            vals_np = np.array(vals, dtype=np.float32).reshape(-1, 1)
            scaler = RobustScaler()
            scaler.fit(vals_np)
            feature_statistics[feat] = dict(
                type='numeric',
                max=float(vals_np.max()),
                center=float(scaler.center_.item()),
                scale=float(scaler.scale_.item()),
            )

    # Categorical vocabularies
    feature_statistics['op_name'] = dict(
        type='categorical',
        value_dict={v: i for i, v in enumerate(sorted(op_names))},
        no_vals=len(op_names),
    )
    feature_statistics['operator'] = dict(
        type='categorical',
        value_dict={v: i for i, v in enumerate(sorted(filter_ops))} if filter_ops else {'=': 0},
        no_vals=max(len(filter_ops), 1),
    )
    feature_statistics['aggregation'] = dict(
        type='categorical',
        value_dict={v: i for i, v in enumerate(sorted(agg_ops))} if agg_ops else {'': 0},
        no_vals=max(len(agg_ops), 1),
    )
    feature_statistics['join_conds'] = dict(
        type='categorical',
        value_dict={v: i for i, v in enumerate(sorted(join_conds))} if join_conds else {'': 0},
        no_vals=max(len(join_conds), 1),
    )
    feature_statistics['tablename'] = dict(
        type='categorical',
        value_dict={v: i for i, v in enumerate(sorted(table_names))} if table_names else {'': 0},
        no_vals=max(len(table_names), 1),
    )

    # Ranges
    num_columns = len(column_stats) if isinstance(column_stats, dict) else 1024
    feature_statistics['columns'] = dict(type='numeric', max=num_columns)
    feature_statistics['column'] = dict(type='numeric', max=num_columns)

    num_tables = len(table_names) if table_names else 64
    feature_statistics['table'] = dict(type='numeric', max=num_tables)

    # Alias
    feature_statistics['operator_type'] = feature_statistics['op_name']

    return feature_statistics


def evaluate_test(model, test_loader, device, test_dataset_name):
    """Evaluate model on test set with detailed metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch, labels, _ in test_loader:
            try:
                batch = tuple(b.to(device) for b in batch)
                labels = labels.to(device)
                
                predictions = model(batch)
                predictions_squeezed = predictions.squeeze(-1)
                
                all_predictions.append(predictions_squeezed.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
            except Exception:
                continue
    
    if len(all_predictions) == 0:
        return None
    
    all_predictions = np.concatenate(all_predictions).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    min_val = 0.1
    safe_preds = np.clip(all_predictions, min_val, np.inf)
    safe_labels = np.clip(all_labels, min_val, np.inf)
    
    # Calculate metrics
    rmse = RMSE().evaluate_metric(labels=all_labels, preds=all_predictions)
    median_q_error = QError(percentile=50, min_val=min_val).evaluate_metric(labels=safe_labels, preds=safe_preds)
    mean_q_error = float(np.mean(np.maximum(safe_preds / safe_labels, safe_labels / safe_preds)))
    p95_q_error = QError(percentile=95, min_val=min_val).evaluate_metric(labels=safe_labels, preds=safe_preds)
    p99_q_error = QError(percentile=99, min_val=min_val).evaluate_metric(labels=safe_labels, preds=safe_preds)
    max_q_error = QError(percentile=100, min_val=min_val).evaluate_metric(labels=safe_labels, preds=safe_preds)
    
    print(f"\n  📊 Test Results for {test_dataset_name}")
    print(f"  {'─'*70}")
    print(f"    Samples: {len(all_labels)}")
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
        'num_samples': int(len(all_labels))
    }


def run_leave_one_out(
    dbms_name: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    hidden_dim: int = 256,
    learning_rate: float = 0.001,
    device: str = "cuda:0",
    max_plans_per_dataset: int = None,
    statistics_dir: str = 'datasets_statistics'
):
    """
    Run leave-one-out cross-validation across all datasets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"LEAVE-ONE-OUT CROSS-VALIDATION - QueryFormer ({dbms_name.upper()})")
    print("="*80)
    print(f"Total Datasets: {len(DATASET_NAMES)}")
    print(f"Device: {device}")
    print(f"Epochs per fold: {epochs}")
    print(f"Batch size: {batch_size}")
    print("="*80)
    print()
    
    # Step 1: Load all plans once
    all_dataset_plans = load_all_plans_once(data_dir, dbms_name, max_plans_per_dataset)
    
    # Step 2: Load all table samples once (for sample_vec generation)
    all_table_samples = load_all_table_samples_once(no_samples=1000)
    
    # Step 3: Load all statistics once
    all_statistics = load_all_statistics_once(dbms_name, statistics_dir)
    
    # Step 4: Augment sample_vec for all plans (one-time operation)
    if dbms_name == "trino" and all_table_samples:
        print("🔧 Augmenting sample_vec for all plans (one-time operation)")
        print("="*80)
        from models.workload_driven.preprocessing.sample_vectors_trino import augment_sample
        
        for dataset_name, plans in all_dataset_plans.items():
            if dataset_name in all_table_samples and dataset_name in all_statistics:
                table_samples = all_table_samples[dataset_name]
                col_stats_list = all_statistics[dataset_name]['col_stats_list']
                
                if table_samples and col_stats_list:
                    print(f"  Augmenting {dataset_name}...")
                    success_count = 0
                    for plan in plans:
                        try:
                            augment_sample(table_samples, col_stats_list, plan)
                            success_count += 1
                        except Exception:
                            pass
                    print(f"    ✓ Augmented {success_count}/{len(plans)} plans")
        
        print(f"\n✓ Sample vector augmentation complete")
        print("="*80)
        print()
    
    # Step 5: Generate feature statistics from ALL plans
    print("🔧 Generating feature statistics from all plans")
    print("="*80)
    all_plans = []
    for plans in all_dataset_plans.values():
        all_plans.extend(plans)
    
    # Use first available dataset stats for initial feature building
    first_dataset = next(iter(all_statistics.values()))
    feature_statistics = build_feature_statistics(
        all_plans, 
        first_dataset['db_stats'], 
        first_dataset['column_stats'],
        dbms_name
    )
    print(f"✓ Generated feature statistics for {len(feature_statistics)} features")
    print()
    
    # Step 6: Leave-one-out loop
    all_results = []
    
    for fold_idx, test_dataset_name in enumerate(DATASET_NAMES):
        if test_dataset_name not in all_dataset_plans:
            print(f"⚠️  Fold {fold_idx+1}/{len(DATASET_NAMES)}: {test_dataset_name} not found, skipping")
            continue
        
        print("\n" + "="*80)
        print(f"FOLD {fold_idx+1}/{len(DATASET_NAMES)}: Test on {test_dataset_name}")
        print("="*80)
        
        # Prepare train and test plans
        train_plans = []
        for dataset_name, plans in all_dataset_plans.items():
            if dataset_name != test_dataset_name:
                train_plans.extend(plans)
        
        test_plans = all_dataset_plans[test_dataset_name]
        
        print(f"  Train plans: {len(train_plans)} (from {len(all_dataset_plans)-1} datasets)")
        print(f"  Test plans: {len(test_plans)} (from {test_dataset_name})")
        
        # Use first training dataset's statistics
        train_dataset_names = [d for d in DATASET_NAMES if d != test_dataset_name and d in all_statistics]
        if train_dataset_names:
            db_stats = {0: all_statistics[train_dataset_names[0]]['db_stats']}
            column_statistics = all_statistics[train_dataset_names[0]]['column_stats']
        else:
            db_stats = {0: StandardizedStatistics()}
            column_statistics = {}
        
        # Split train into train/val
        train_size = int(0.85 * len(train_plans))
        val_size = len(train_plans) - train_size
        train_split, val_split = random_split(
            train_plans,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create collator
        config = QueryFormerModelConfig(
            hidden_dim_plan=hidden_dim,
            device=device,
            max_num_filters=20  # Increase to handle complex queries (default: 6)
        )
        
        collate_fn = functools.partial(
            query_former_plan_collator,
            feature_statistics=feature_statistics,
            db_statistics=db_stats,
            column_statistics=column_statistics,
            word_embeddings=None,
            dim_word_hash=1000,
            dim_word_embedding=64,
            histogram_bin_size=config.histogram_bin_number,
            max_num_filters=config.max_num_filters,
            dim_bitmaps=1000
        )
        
        train_loader = DataLoader(
            UnifiedPlanDataset([train_plans[i] for i in train_split.indices]),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            UnifiedPlanDataset([train_plans[i] for i in val_split.indices]),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            UnifiedPlanDataset(test_plans),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        print(f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        # Create model
        input_dims = PlanModelInputDims(
            feature_statistics=feature_statistics,
            dim_word_embedding=64,
            dim_word_hash=1000,
            dim_bitmaps=1000
        )
        
        model = QueryFormer(config, input_dims, feature_statistics, label_norm=None)
        model = model.to(device)
        
        # Training
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Use model's internal loss function (QLoss by default, matching PostgreSQL version)
        criterion = model.loss_fxn
        
        best_val_loss = float('inf')
        
        print(f"\n  🚀 Training...")
        for epoch in range(epochs):
            # Train
            model.train()
            total_loss = 0.0
            
            for batch, labels, _ in train_loader:
                optimizer.zero_grad()
                
                batch = tuple(b.to(device) for b in batch)
                labels = labels.to(device)
                
                predictions = model(batch).squeeze(-1)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch, labels, _ in val_loader:
                    batch = tuple(b.to(device) for b in batch)
                    labels = labels.to(device)
                    
                    predictions = model(batch)
                    predictions_squeezed = predictions.squeeze(-1)
                    loss = criterion(predictions_squeezed, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1}/{epochs}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        # Evaluate on test set
        test_metrics = evaluate_test(model, test_loader, device, test_dataset_name)
        
        if test_metrics:
            all_results.append(test_metrics)
        
        # Clean up model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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
            'num_folds': len(all_results),
            'epochs_per_fold': epochs,
            'batch_size': batch_size,
            'hidden_dim': hidden_dim,
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
    parser = argparse.ArgumentParser(description='Leave-One-Out Cross-Validation for QueryFormer')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing all dataset files')
    parser.add_argument('--dbms', type=str, default='trino',
                       choices=['trino', 'postgres', 'mysql'],
                       help='DBMS name')
    parser.add_argument('--output_dir', type=str, default='models/queryformer_loo',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cuda:0, cpu, etc.)')
    parser.add_argument('--max_plans', type=int, default=10000,
                       help='Maximum plans per dataset')
    parser.add_argument('--statistics_dir', type=str, default='datasets_statistics',
                       help='Statistics directory')
    
    args = parser.parse_args()
    
    return run_leave_one_out(
        dbms_name=args.dbms,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        device=args.device,
        max_plans_per_dataset=args.max_plans,
        statistics_dir=args.statistics_dir
    )


if __name__ == "__main__":
    sys.exit(main())

