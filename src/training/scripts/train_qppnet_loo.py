"""
QPPNet Training Script with Leave-One-Out Cross-Validation for Trino

This script performs leave-one-out cross-validation across 20 datasets.
For each iteration, it trains on 19 datasets and tests on 1 held-out dataset.

Single-GPU Usage:
    python -m training.scripts.train_qppnet_loo \
        --data_dir /path/to/explain_analyze_results \
        --output_dir models/qppnet_loo \
        --epochs 50 \
        --batch_size 32 \
        --device cuda:0

Multi-GPU Usage (e.g., 7 GPUs):
    torchrun --nproc_per_node=7 -m training.scripts.train_qppnet_loo \
        --data_dir /path/to/explain_analyze_results \
        --output_dir models/qppnet_loo \
        --epochs 50 \
        --batch_size 32 \
        --device cuda
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add src to path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse
import functools
import torch
import torch.distributed as dist
import json
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
from datetime import datetime
from types import SimpleNamespace
import re

# Initialize plugin system
import core.init_plugins

from models.qppnet.qppnet_model import QPPNet
from models.qppnet.qppnet_dataloader import qppnet_collator
from classes.classes import QPPNetModelConfig
from classes.workload_runs import WorkloadRuns
from training.training.metrics import QError, RMSE
from training.training.utils import recursive_to
from sklearn.preprocessing import RobustScaler
from core.plugins.registry import DBMSRegistry
from core.capabilities import check_capabilities

# Dataset names (in order)
DATASET_NAMES = [
    'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
    'consumer', 'credit', 'employee', 'fhnk', 'financial',
    'geneea', 'genome', 'hepatitis', 'imdb', 'movielens',
    'seznam', 'ssb', 'tournament', 'tpc_h', 'walmart'
]


class PlanDataset(Dataset):
    """Simple dataset wrapper for query plans"""
    def __init__(self, plans, indices):
        self.plans = plans
        self.indices = indices
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx):
        return self.indices[idx], self.plans[idx]


def parse_args():
    parser = argparse.ArgumentParser(description='QPPNet Leave-One-Out Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing explain_analyze results')
    parser.add_argument('--statistics_dir', type=str, default='datasets_statistics',
                        help='Directory containing database statistics')
    parser.add_argument('--output_dir', type=str, default='models/qppnet_loo',
                        help='Output directory for models and results')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension for neural units')
    parser.add_argument('--limit_queries', type=int, default=None,
                        help='Limit number of queries per dataset (for testing)')
    parser.add_argument('--start_fold', type=int, default=0,
                        help='Start from specific fold (0-19)')
    parser.add_argument('--end_fold', type=int, default=None,
                        help='End at specific fold (0-19, inclusive)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed logging output')
    return parser.parse_args()


def load_trino_plans_from_file(file_path: Path, schema_name: str):
    """
    Load Trino query plans from EXPLAIN ANALYZE VERBOSE output file.
    
    Returns:
        list: List of TrinoPlanOperator objects
        list: List of corresponding runtimes (in ms)
    """
    if not file_path.exists():
        return [], []

    parser = DBMSRegistry.get_parser('trino')
    plans = []
    runtimes = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Split by -- stmt markers
    queries = []
    current_query = []
    for line in lines:
        line_stripped = line.rstrip('\n')
        if line_stripped.startswith('-- ') and 'stmt' in line_stripped:
            if current_query:
                queries.append('\n'.join(current_query))
                current_query = []
        else:
            current_query.append(line_stripped)
    if current_query:
        queries.append('\n'.join(current_query))
    
    # Parse each query
    for query_text in queries:
        try:
            # Remove surrounding quotes if present
            query_text = query_text.strip()
            if query_text.startswith('"') and query_text.endswith('"'):
                query_text = query_text[1:-1]
            
            # Parse the plan
            root_operator, execution_time, planning_time = parser.parse_raw_plan(
                query_text,
                analyze=True
            )
            
            if root_operator is not None:
                # Set plan_runtime and database_id
                root_operator.plan_runtime = execution_time / 1000  # Convert to seconds
                root_operator.database_id = schema_name
                
                plans.append(root_operator)
                runtimes.append(execution_time)
        except Exception as e:
            continue
    
    return plans, runtimes


def load_all_plans(data_dir: str, limit_per_dataset: int = None):
    """
    Load all query plans from all datasets.
    
    Returns:
        dict: {dataset_name: (plans, runtimes)}
    """
    data_dir = Path(data_dir)
    all_data = {}
    
    print("📂 Loading query plans...")
    print("="*80)
    
    for dataset_name in DATASET_NAMES:
        # Look for files matching the pattern
        patterns = [
            f"{dataset_name}_complex_workload_*_explain_analyze.txt",
            f"{dataset_name}_*_explain_analyze.txt"
        ]
        files = []
        for pattern in patterns:
            matched = list(data_dir.glob(pattern))
            files.extend(matched)
        files = list(set(files))  # Remove duplicates
        
        if not files:
            print(f"  ⚠️  {dataset_name}: No plan files found (searched in {data_dir})")
            all_data[dataset_name] = ([], [])
            continue
        
        all_plans = []
        all_runtimes = []
        
        for file_path in files:
            plans, runtimes = load_trino_plans_from_file(file_path, dataset_name)
            all_plans.extend(plans)
            all_runtimes.extend(runtimes)
        
        if limit_per_dataset:
            all_plans = all_plans[:limit_per_dataset]
            all_runtimes = all_runtimes[:limit_per_dataset]
        
        all_data[dataset_name] = (all_plans, all_runtimes)
        print(f"  ✓ {dataset_name}: {len(all_plans)} queries loaded")
    
    total_queries = sum(len(plans) for plans, _ in all_data.values())
    print(f"\n✓ Total queries loaded: {total_queries}")
    print("="*80)
    print()
    
    return all_data


def load_column_statistics(dataset_name: str, statistics_dir: str):
    """Load column statistics for a dataset."""
    from cross_db_benchmark.benchmark_tools.utils import load_json
    import ast
    
    stats_path = Path(statistics_dir) / f"iceberg_{dataset_name}" / 'column_stats.json'
    
    if not stats_path.exists():
        return {}
    
    try:
        raw_stats = load_json(str(stats_path), namespace=False)
        
        # Convert to nested dict format expected by QPPNet
        column_stats = {}
        for key, val in raw_stats.items():
            if isinstance(key, tuple) and len(key) == 2:
                table_name, col_name = key
            elif isinstance(key, str) and key.startswith('('):
                try:
                    table_name, col_name = ast.literal_eval(key)
                except:
                    continue
            else:
                continue
            
            if table_name not in column_stats:
                column_stats[table_name] = {}
            
            # Add num_unique for compatibility
            val_copy = dict(val)
            if 'num_unique' not in val_copy and 'distinct_count' in val_copy:
                val_copy['num_unique'] = val_copy['distinct_count']
            
            column_stats[table_name][col_name] = val_copy
        
        return column_stats
        
    except Exception as e:
        print(f"Warning: Failed to load column statistics for {dataset_name}: {e}")
        return {}


def collect_feature_statistics_from_plans(plans, featurization):
    """Collect feature statistics from loaded plans."""
    from collections import defaultdict
    from sklearn.preprocessing import RobustScaler
    from training.preprocessing.feature_statistics import FeatureType
    from models.qppnet.trino_adapter import TRINO_TO_POSTGRES_OP_MAPPING
    
    # Collect all feature values from plans
    value_dict = defaultdict(list)
    
    def collect_from_node(node):
        """再帰的にノードから特徴量を収集"""
        if not hasattr(node, 'plan_parameters'):
            return
        
        params = node.plan_parameters
        
        # Get op_name
        op_name = getattr(params, 'op_name', None) if hasattr(params, 'op_name') else None
        
        # Collect features based on operator type
        if op_name:
            # Get features for this operator type from featurization
            pg_op_name = TRINO_TO_POSTGRES_OP_MAPPING.get(op_name, op_name)
            features_for_op = featurization.QPP_NET_OPERATOR_TYPES.get(pg_op_name, featurization.PLAN_FEATURES)
            
            for feature in features_for_op:
                # Map feature names to Trino attributes
                value = None
                if feature == 'Plan Width':
                    value = getattr(params, 'est_width', None)
                    if value == 0:
                        value = None
                elif feature == 'Plan Rows':
                    value = getattr(params, 'est_rows', None)
                    if value == 0:
                        value = None
                elif feature == 'Total Cost':
                    value = getattr(params, 'est_cost', None)
                    if value == 0:
                        value = None
                elif feature == 'Actual Rows':
                    value = getattr(params, 'act_output_rows', None)
                    if value == 0:
                        value = None
                elif feature == 'Relation Name':
                    value = getattr(params, 'table', None)
                elif feature == 'Join Type':
                    if 'Inner' in op_name:
                        value = 'Inner'
                    elif 'Left' in op_name:
                        value = 'Left'
                    elif 'Right' in op_name:
                        value = 'Right'
                    elif 'Cross' in op_name:
                        value = 'Cross'
                    else:
                        value = 'Hash'
                elif feature == 'Hash Algorithm':
                    dist = getattr(params, 'distribution', None)
                    if dist == 'REPLICATED':
                        value = 'Broadcast'
                    elif dist == 'PARTITIONED':
                        value = 'Hash'
                    else:
                        value = 'Hash'
                elif feature == 'Parent Relationship':
                    value = 'Inner'
                elif feature == 'Hash Buckets':
                    est_mem = getattr(params, 'est_memory', None)
                    if est_mem:
                        value = max(1, int(est_mem / 1024))
                elif feature == 'Peak Memory Usage':
                    value = getattr(params, 'est_memory', None)
                elif feature == 'Sort Key':
                    value = 'unknown'
                elif feature == 'Sort Method':
                    value = 'quicksort'
                elif feature == 'Strategy':
                    value = 'Hashed' if 'Partial' in op_name or 'Final' in op_name else 'Plain'
                elif feature == 'Partial Mode':
                    if 'Partial' in op_name:
                        value = 'Partial'
                    elif 'Final' in op_name:
                        value = 'Finalize'
                    else:
                        value = 'Simple'
                elif feature in ['Min', 'Max', 'Mean']:
                    value = 0
                elif feature == 'Index Name':
                    value = 'unknown'
                elif feature == 'Scan Direction':
                    value = 'Forward'
                
                if value is not None:
                    value_dict[feature].append(value)
        
        # Recursively collect from children
        for child in node.children:
            collect_from_node(child)
    
    # Collect from all plans
    for plan in plans:
        collect_from_node(plan)
    
    # Generate statistics
    statistics_dict = {}
    for k, values in value_dict.items():
        values = [v for v in values if v is not None]
        if len(values) == 0:
            continue
        
        # Check if numeric or categorical
        if all([isinstance(v, (int, float)) for v in values]):
            scaler = RobustScaler()
            np_values = np.array(values, dtype=np.float32).reshape(-1, 1)
            scaler.fit(np_values)
            
            statistics_dict[k] = {
                'max': float(np_values.max()),
                'scale': float(scaler.scale_.item()),
                'center': float(scaler.center_.item()),
                'type': str(FeatureType.numeric)
            }
        else:
            unique_values = set(str(v) for v in values)
            statistics_dict[k] = {
                'value_dict': {v: id for id, v in enumerate(unique_values)},
                'no_vals': len(unique_values),
                'type': str(FeatureType.categorical)
            }
    
    # Add missing features with default values
    all_required_features = set()
    for op_type, features in featurization.QPP_NET_OPERATOR_TYPES.items():
        all_required_features.update(features)
    
    for feature in all_required_features:
        if feature not in statistics_dict:
            if feature in ['Plan Width', 'Plan Rows', 'Total Cost', 'Actual Rows', 
                          'Hash Buckets', 'Peak Memory Usage', 'Min', 'Max', 'Mean']:
                statistics_dict[feature] = {
                    'max': 1.0,
                    'scale': 1.0,
                    'center': 0.0,
                    'type': str(FeatureType.numeric)
                }
            else:
                default_values = {
                    'Relation Name': ['unknown'],
                    'Join Type': ['Inner', 'Left', 'Right', 'Cross', 'Hash'],
                    'Hash Algorithm': ['Broadcast', 'Hash', 'Nested Loop'],
                    'Parent Relationship': ['Inner', 'Outer', 'SubPlan'],
                    'Sort Key': ['unknown'],
                    'Sort Method': ['quicksort', 'external merge'],
                    'Strategy': ['Plain', 'Hashed', 'Sorted'],
                    'Partial Mode': ['Simple', 'Partial', 'Finalize'],
                    'Index Name': ['unknown'],
                    'Scan Direction': ['Forward', 'Backward']
                }
                
                values = default_values.get(feature, ['unknown'])
                statistics_dict[feature] = {
                    'value_dict': {v: id for id, v in enumerate(values)},
                    'no_vals': len(values),
                    'type': str(FeatureType.categorical)
                }
    
    # Ensure 'unknown' token exists for categorical features
    for feature_name, stats in statistics_dict.items():
        if stats.get('type') == str(FeatureType.categorical):
            value_dict = stats.setdefault('value_dict', {})
            if 'unknown' not in value_dict:
                value_dict['unknown'] = len(value_dict)
            stats['no_vals'] = len(value_dict)

    return statistics_dict


def add_scalers_to_feature_statistics(feature_statistics):
    """Add RobustScaler objects to numeric features in feature_statistics."""
    from training.preprocessing.feature_statistics import FeatureType
    
    for feature_name, stats in feature_statistics.items():
        if stats.get('type') == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = np.array([stats['center']])
            scaler.scale_ = np.array([stats['scale']])
            stats['scaler'] = scaler
    
    return feature_statistics


def train_epoch(model, train_loader, optimizer, device, epoch, verbose=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    target_model = model.module if hasattr(model, "module") else model
    criterion = target_model.loss_fxn

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) if verbose else None
    iterator = pbar if pbar is not None else train_loader

    for batch in iterator:
        if not batch or len(batch) != 3:
            continue
        query_plans, labels, sample_idxs = batch

        if not query_plans:
            continue

        recursive_to(query_plans, device)
        labels = torch.tensor(labels, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        predictions = model(query_plans)
        loss = criterion(predictions, labels)

        loss.backward()
        target_model.backward()  # Step operator-level optimizers
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if pbar is not None:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    if pbar is not None:
        pbar.close()

    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, val_loader, device):
    """Evaluate the model and compute Q-Error metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            if not batch or len(batch) != 3:
                continue

            query_plans, labels, sample_idxs = batch

            if not query_plans:
                continue

            recursive_to(query_plans, device)
            predictions = model(query_plans)

            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels)
    
    if not all_predictions:
        return float('inf'), float('inf'), float('inf')
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Compute Q-Error
    q_errors = np.maximum(predictions / labels, labels / predictions)
    median_qerror = float(np.median(q_errors))
    mean_qerror = float(np.mean(q_errors))
    p95_qerror = float(np.percentile(q_errors, 95))
    
    return median_qerror, mean_qerror, p95_qerror


def train_single_fold(fold_idx: int, train_datasets: list, test_dataset: str, 
                      all_plans_data: dict, args, results_file):
    """Train and evaluate a single fold."""
    is_distributed = dist.is_available() and dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device.lower() == 'cpu':
        device_obj = torch.device('cpu')
        is_distributed = False
    else:
        if is_distributed:
            device_obj = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device_obj)
        else:
            device_obj = torch.device(args.device)
    is_main_process = (not is_distributed) or (local_rank == 0)

    if is_main_process:
        print("\n" + "="*80)
        print(f"FOLD {fold_idx + 1}/20: Testing on {test_dataset}")
        print(f"Training on: {', '.join(train_datasets)}")
        print("="*80)
    
    # Create output directory for this fold
    fold_dir = Path(args.output_dir) / f"fold_{fold_idx:02d}_{test_dataset}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all training plans
    train_plans = []
    train_runtimes = []
    for dataset in train_datasets:
        plans, runtimes = all_plans_data[dataset]
        train_plans.extend(plans)
        train_runtimes.extend(runtimes)
    
    # Get test plans
    test_plans, test_runtimes = all_plans_data[test_dataset]
    
    if is_main_process:
        print(f"\n📊 Data summary:")
        print(f"  Training queries: {len(train_plans)}")
        print(f"  Test queries: {len(test_plans)}")
    
    if not train_plans or not test_plans:
        if is_main_process:
            print("⚠️  Insufficient data for this fold, skipping...")
        return None
    
    # Load column statistics (use first training dataset as representative)
    column_statistics = {}
    for dataset in train_datasets:
        stats = load_column_statistics(dataset, args.statistics_dir)
        column_statistics.update(stats)
    
    # Collect feature statistics from training plans
    from training.featurizations import QPPNetFeaturization
    featurization = QPPNetFeaturization()
    
    if is_main_process:
        print("\n📊 Collecting feature statistics from training plans...")
    feature_statistics = collect_feature_statistics_from_plans(train_plans, featurization)
    
    # Save statistics (without scalers for JSON serialization)
    stats_to_save = {k: {kk: vv for kk, vv in v.items() if kk != 'scaler'} 
                     for k, v in feature_statistics.items()}
    if is_main_process:
        with open(fold_dir / 'feature_statistics.json', 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        with open(fold_dir / 'column_statistics.json', 'w') as f:
            json.dump(column_statistics, f, indent=2, default=str)
    
    # Add scalers after saving
    feature_statistics = add_scalers_to_feature_statistics(feature_statistics)
    
    if is_main_process:
        print(f"  Collected statistics for {len(feature_statistics)} features")
    
    # Create datasets
    train_size = int(0.9 * len(train_plans))
    train_dataset = PlanDataset(train_plans[:train_size], list(range(train_size)))
    val_dataset = PlanDataset(train_plans[train_size:], list(range(train_size, len(train_plans))))
    test_dataset_obj = PlanDataset(test_plans, list(range(len(test_plans))))
    
    # Create collate function (now using dbms_name for registry-based adaptation)
    train_collate_fn = functools.partial(
        qppnet_collator,
        db_statistics={},
        feature_statistics=feature_statistics,
        column_statistics=column_statistics,
        plan_featurization=featurization,
        dbms_name='trino',  # Changed from use_trino=True (registry-aware)
        debug_print=args.verbose
    )
    
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset_obj, shuffle=False)
    else:
        train_sampler = val_sampler = test_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        collate_fn=train_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        collate_fn=train_collate_fn
    )
    test_loader = DataLoader(
        test_dataset_obj,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=0,
        collate_fn=train_collate_fn
    )
    
    # Create model config
    model_config = QPPNetModelConfig(
        device=str(device_obj),
        hidden_dim_plan=args.hidden_dim,
        batch_size=args.batch_size,
        featurization=featurization,
        loss_class_name='QPPLossGlobal',
        loss_class_kwargs={'loss_type': 'mse'}
    )
    
    # Initialize model
    if is_main_process:
        print("\n🤖 Initializing QPPNet model...")
    dummy_workload_runs = WorkloadRuns(
        train_workload_runs=[],
        test_workload_runs=[]
    )
    
    model = QPPNet(
        model_config=model_config,
        workload_runs=dummy_workload_runs,
        feature_statistics=feature_statistics,
        use_global_loss=True
    ).to(device_obj)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process:
        print(f"Model parameters: {total_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    if is_main_process:
        print("\n🏋️  Starting training...")
    best_val_qerror = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_loader, optimizer, device_obj, epoch, verbose=args.verbose)
        val_median_qerror, val_mean_qerror, val_p95_qerror = evaluate(model, val_loader, device_obj)
        
        if is_main_process:
            print(f"Epoch {epoch}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Median Q-Error: {val_median_qerror:.2f}")
        
        # Save best model
        if val_median_qerror < best_val_qerror:
            best_val_qerror = val_median_qerror
            if is_main_process:
                model_to_save = model.module if is_distributed else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_qerror': val_median_qerror,
                }, fold_dir / 'best_model.pt')
                print(f"  💾 Saved best model (Q-Error: {val_median_qerror:.2f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            if is_main_process:
                print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Load best model and evaluate on test set
    fold_result = None
    if is_main_process:
        print("\n📊 Evaluating on test set...")
        checkpoint = torch.load(fold_dir / 'best_model.pt', map_location=device_obj)
        model_to_use = model.module if is_distributed else model
        model_to_use.load_state_dict(checkpoint['model_state_dict'])
        
        test_median_qerror, test_mean_qerror, test_p95_qerror = evaluate(model_to_use, test_loader, device_obj)
        
        print(f"\n✅ Test Results:")
        print(f"  Median Q-Error: {test_median_qerror:.2f}")
        print(f"  Mean Q-Error: {test_mean_qerror:.2f}")
        print(f"  95th Percentile Q-Error: {test_p95_qerror:.2f}")
        
        # Save fold results
        fold_result = {
            'fold': fold_idx,
            'test_dataset': test_dataset,
            'train_datasets': train_datasets,
            'num_train_queries': len(train_plans),
            'num_test_queries': len(test_plans),
            'best_epoch': checkpoint['epoch'],
            'val_median_qerror': float(best_val_qerror),
            'test_median_qerror': float(test_median_qerror),
            'test_mean_qerror': float(test_mean_qerror),
            'test_p95_qerror': float(test_p95_qerror),
        }
        
        # Append to results file
        with open(results_file, 'a') as f:
            f.write(json.dumps(fold_result) + '\n')
    
    if is_distributed:
        dist.barrier()
    
    return fold_result


def main():
    args = parse_args()
    dbms_name = 'trino'
    
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl' if args.device.startswith('cuda') else 'gloo')
    is_distributed = dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_distributed else 0
    is_main_process = (not is_distributed) or (local_rank == 0)
    
    if is_main_process:
        print("="*80)
        print("QPPNet Leave-One-Out Cross-Validation")
        print("="*80)
        print(f"Output directory: {args.output_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Device: {args.device}")
        print("="*80)
        print()

        # Capability check
        try:
            plugin = DBMSRegistry.get_plugin(dbms_name)
            provided_caps = plugin.get_capabilities()
            temp_config = QPPNetModelConfig()
            required_caps = temp_config.required_capabilities
            missing_caps = check_capabilities(
                required_caps,
                provided_caps,
                temp_config.name.NAME if temp_config.name else "qppnet",
                dbms_name
            )

            if missing_caps:
                print("="*80)
                print(f"⚠️  WARNING: DBMS '{dbms_name}' is missing capabilities required by QPPNet: {missing_caps}")
                print("    Training may fail or produce suboptimal results.")
                print("="*80)
            else:
                print(f"✓ Capability check passed: {dbms_name} provides all required capabilities for QPPNet.\n")
        except Exception as e:
            print(f"⚠️  Capability check could not be completed: {e}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results file
    results_file = output_dir / 'loo_results.jsonl'
    
    # Load all plans once
    all_plans_data = load_all_plans(args.data_dir, args.limit_queries)
    
    # Determine fold range
    start_fold = args.start_fold
    end_fold = args.end_fold if args.end_fold is not None else len(DATASET_NAMES) - 1
    
    # Run LOO cross-validation
    all_results = []
    
    for fold_idx in range(start_fold, end_fold + 1):
        test_dataset = DATASET_NAMES[fold_idx]
        train_datasets = [d for d in DATASET_NAMES if d != test_dataset]
        
        try:
            result = train_single_fold(fold_idx, train_datasets, test_dataset, 
                                      all_plans_data, args, results_file)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error in fold {fold_idx} ({test_dataset}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute overall statistics
    if is_main_process and all_results:
        print("\n" + "="*80)
        print("OVERALL RESULTS")
        print("="*80)
        
        median_qerrors = [r['test_median_qerror'] for r in all_results]
        mean_qerrors = [r['test_mean_qerror'] for r in all_results]
        
        print(f"Average Median Q-Error: {np.mean(median_qerrors):.2f} ± {np.std(median_qerrors):.2f}")
        print(f"Average Mean Q-Error: {np.mean(mean_qerrors):.2f} ± {np.std(mean_qerrors):.2f}")
        print(f"Completed folds: {len(all_results)}/{end_fold - start_fold + 1}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_folds': len(all_results),
            'avg_median_qerror': float(np.mean(median_qerrors)),
            'std_median_qerror': float(np.std(median_qerrors)),
            'avg_mean_qerror': float(np.mean(mean_qerrors)),
            'std_mean_qerror': float(np.std(mean_qerrors)),
            'fold_results': all_results
        }
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Results saved to {output_dir}")
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

