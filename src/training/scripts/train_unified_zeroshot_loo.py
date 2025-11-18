"""
Unified Trino Zero-Shot Training Script with Leave-One-Out Cross-Validation

This script performs leave-one-out cross-validation across 20 datasets.
For each iteration, it trains on 19 datasets and tests on 1 held-out dataset.

Single-GPU Usage:
    python -m training.scripts.train_unified_zeroshot_loo \
        --dbms trino \
        --data_dir /path/to/explain_analyze_results \
        --output_dir models/zeroshot_loo \
        --device cuda:0

Multi-GPU Usage (e.g., 7 GPUs):
    torchrun --nproc_per_node=7 -m training.scripts.train_unified_zeroshot_loo \
        --dbms trino \
        --data_dir /path/to/explain_analyze_results \
        --output_dir models/zeroshot_loo \
        --device cuda
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

# Environment setup
for i in range(11):
    env_key = f'NODE{i:02d}'
    if os.environ.get(env_key) in (None, '', 'None'):
        os.environ[env_key] = '[]'

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
import copy
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Initialize plugin system
import core.init_plugins

# Import unified components
from core.plugins.registry import DBMSRegistry
from core.graph.unified_collator import unified_plan_collator
from core.statistics.schema import StandardizedStatistics
from training.unified_featurizations import UnifiedTrueCardDetail, UnifiedEstSystemCardDetail
from models.zeroshot.zero_shot_model import ZeroShotModel
from classes.classes import ZeroShotModelConfig
from training.preprocessing.feature_statistics import gather_feature_statistics, FeatureType
from training.training.metrics import QError, RMSE


# Dataset names (in order)
DATASET_NAMES = [
    'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis',
    'consumer', 'credit', 'employee', 'fhnk', 'financial',
    'geneea', 'genome', 'hepatitis', 'imdb', 'movielens',
    'seznam', 'ssb', 'tournament', 'tpc_h', 'walmart'
]


class UnifiedPlanDataset(Dataset):
    """Universal plan dataset that works for all DBMS."""
    
    def __init__(self, plans):
        self.plans = plans
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx):
        return idx, self.plans[idx]


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
                plan.dataset_name = dataset_name  # Tag with dataset name
            
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


def load_all_statistics_once(dbms_name: str, statistics_dir: str = 'datasets_statistics'):
    """
    Load all database statistics once and cache them.
    
    Returns:
        dict: {dataset_name: StandardizedStatistics}
    """
    print("📊 Loading all database statistics (one-time operation)")
    print("="*80)
    
    all_stats = {}
    
    for dataset_name in DATASET_NAMES:
        try:
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
            all_stats[dataset_name] = standardized
            
            print(f"  ✓ {dataset_name}: {len(standardized.column_stats)} columns, {len(standardized.table_stats)} tables")
            
        except Exception as e:
            print(f"  ⚠️  {dataset_name}: Statistics loading failed - {e}")
            all_stats[dataset_name] = StandardizedStatistics()
    
    print(f"\n✓ Loaded statistics for {len(all_stats)} datasets")
    print("="*80)
    print()
    
    return all_stats


def create_feature_statistics_from_plans(plans, plan_featurization, dbms_name: str):
    """Generate feature statistics from plans (same as before)."""
    from sklearn.preprocessing import RobustScaler
    from core.features.mapper import FeatureMapper

    mapper = FeatureMapper(dbms_name)

    # Collect actual operators used
    actual_op_names = set()
    filter_operators = set()
    aggregations = set()
    
    def collect_from_node(node):
        if hasattr(node, 'plan_parameters'):
            params = node.plan_parameters
            
            op_name = None
            if isinstance(params, dict):
                op_name = params.get('op_name')
            elif hasattr(params, 'op_name'):
                op_name = params.op_name
            
            if op_name:
                actual_op_names.add(op_name)
            
            # Collect filter operators
            filter_col = params.get('filter_columns') if isinstance(params, dict) else getattr(params, 'filter_columns', None)
            if filter_col:
                def collect_filter_ops(fn):
                    op = fn.get('operator') if isinstance(fn, dict) else getattr(fn, 'operator', None)
                    if op:
                        filter_operators.add(str(op))
                    children = fn.get('children', []) if isinstance(fn, dict) else getattr(fn, 'children', [])
                    for child in children:
                        collect_filter_ops(child)
                collect_filter_ops(filter_col)
            
            # Collect aggregations
            output_cols = params.get('output_columns') if isinstance(params, dict) else getattr(params, 'output_columns', None)
            if output_cols:
                for oc in output_cols:
                    agg = oc.get('aggregation') if isinstance(oc, dict) else getattr(oc, 'aggregation', None)
                    if agg:
                        aggregations.add(str(agg))
        
        for child in node.children:
            collect_from_node(child)
    
    for plan in plans:
        collect_from_node(plan)
    
    # Collect numeric feature values
    numeric_feature_values = {
        'actual_cardinality': [],
        'estimated_cardinality': [],
        'estimated_width': [],
        'estimated_cost': [],
        'workers_planned': [],
        'actual_children_cardinality': [],
        'avg_width': [],
        'correlation': [],
        'n_distinct': [],
        'null_frac': [],
        'row_count': [],
        'page_count': [],
    }
    
    def collect_numeric_features(node):
        if hasattr(node, 'plan_parameters'):
            params = node.plan_parameters
            
            for feat_name in numeric_feature_values.keys():
                try:
                    value = mapper.get_feature(feat_name, params)
                    if value is not None and not isinstance(value, str):
                        numeric_feature_values[feat_name].append(float(value))
                except:
                    pass
        
        for child in node.children:
            collect_numeric_features(child)
    
    for plan in plans:
        collect_numeric_features(plan)
    
    # Build feature statistics
    feature_statistics = {}
    
    # op_name (categorical)
    feature_statistics['op_name'] = {
        'type': 'categorical',
        'value_dict': {op: i for i, op in enumerate(sorted(actual_op_names))},
        'no_vals': len(actual_op_names)
    }
    
    # operator (categorical)
    feature_statistics['operator'] = {
        'type': 'categorical',
        'value_dict': {op: i for i, op in enumerate(sorted(filter_operators))},
        'no_vals': len(filter_operators)
    }
    
    # aggregation (categorical)
    all_aggs = list(aggregations) + ['none', 'None']
    feature_statistics['aggregation'] = {
        'type': 'categorical',
        'value_dict': {agg: i for i, agg in enumerate(all_aggs)},
        'no_vals': len(all_aggs)
    }
    
    # Numeric features
    for feat_name, values in numeric_feature_values.items():
        if len(values) > 0:
            values_array = np.array(values, dtype=np.float32).reshape(-1, 1)
            scaler = RobustScaler()
            scaler.fit(values_array)
            
            feature_statistics[feat_name] = {
                'type': 'numeric',
                'max': float(values_array.max()),
                'scale': float(scaler.scale_.item()),
                'center': float(scaler.center_.item())
            }
        else:
            feature_statistics[feat_name] = {
                'type': 'numeric',
                'max': 1.0,
                'scale': 1.0,
                'center': 0.0
            }
    
    # Other categorical features
    feature_statistics['data_type'] = {
        'type': 'categorical',
        'value_dict': {'integer': 0, 'bigint': 1, 'varchar': 2, 'double': 3, 'unknown': 4},
        'no_vals': 10
    }
    
    feature_statistics['literal_feature'] = {
        'type': 'numeric',
        'max': 1000.0,
        'scale': 1.0,
        'center': 0.0
    }
    
    # Aliases
    feature_statistics['filter_operator'] = feature_statistics['operator']
    feature_statistics['operator_type'] = feature_statistics['op_name']
    
    return feature_statistics


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    criterion = model.module.loss_fxn if hasattr(model, "module") else model.loss_fxn
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        try:
            graph, features, labels, sample_idxs = batch
            
            graph = graph.to(device)
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            
            predictions = model((graph, features))
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        except Exception as e:
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, val_loader, device):
    """Validate model with metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_labels = []
    criterion = model.module.loss_fxn if hasattr(model, "module") else model.loss_fxn
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                graph, features, labels, sample_idxs = batch
                
                graph = graph.to(device)
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)
                
                predictions = model((graph, features))
                
                loss = criterion(predictions, labels)
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
            except Exception:
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    if len(all_predictions) > 0:
        all_predictions = np.concatenate(all_predictions).flatten()
        all_labels = np.concatenate(all_labels).flatten()
        
        min_val = 0.1
        safe_preds = np.clip(all_predictions, min_val, np.inf)
        safe_labels = np.clip(all_labels, min_val, np.inf)
        
        median_q_error = QError(percentile=50, min_val=min_val).evaluate_metric(
            labels=safe_labels, preds=safe_preds
        )
        rmse = RMSE().evaluate_metric(labels=all_labels, preds=all_predictions)
        
        return avg_loss, median_q_error, rmse
    
    return avg_loss, None, None


def evaluate_test(model, test_loader, device, test_dataset_name):
    """Evaluate model on test set with detailed metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                graph, features, labels, sample_idxs = batch
                
                graph = graph.to(device)
                features = {k: v.to(device) for k, v in features.items()}
                
                predictions = model((graph, features))
                
                all_predictions.append(predictions.cpu().numpy())
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
    batch_size: int = 32,
    hidden_dim: int = 128,
    learning_rate: float = 0.001,
    device: str = "cuda:0",
    max_plans_per_dataset: int = None,
    statistics_dir: str = 'datasets_statistics'
):
    """
    Run leave-one-out cross-validation across all datasets.
    Supports multi-GPU training via DistributedDataParallel when launched with torchrun.
    """
    # Initialize distributed training if requested
    is_distributed = dist.is_available() and dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size() if is_distributed else 1
    
    # Resolve device
    if device.lower() == "cpu":
        device_obj = torch.device("cpu")
        is_distributed = False
    else:
        if is_distributed:
            device_obj = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device_obj)
        else:
            device_obj = torch.device(device)
    
    is_main_process = (not is_distributed) or (local_rank == 0)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_main_process:
        print("="*80)
        print(f"LEAVE-ONE-OUT CROSS-VALIDATION - Zero-Shot Model ({dbms_name.upper()})")
        print("="*80)
        print(f"Total Datasets: {len(DATASET_NAMES)}")
        print(f"Device: {device_obj}")
        if is_distributed:
            print(f"Distributed: {world_size} GPUs (local_rank={local_rank})")
        print(f"Epochs per fold: {epochs}")
        print(f"Batch size: {batch_size} (per GPU)" if is_distributed else f"Batch size: {batch_size}")
        print("="*80)
        print()
    
    # Step 1: Load all plans once
    all_dataset_plans = load_all_plans_once(data_dir, dbms_name, max_plans_per_dataset)
    
    # Step 2: Load all statistics once
    all_statistics = load_all_statistics_once(dbms_name, statistics_dir)
    
    # Step 3: Generate feature statistics from ALL plans
    if is_main_process:
        print("🔧 Generating feature statistics from all plans")
        print("="*80)
    all_plans = []
    for plans in all_dataset_plans.values():
        all_plans.extend(plans)
    
    featurization = UnifiedTrueCardDetail
    feature_statistics = create_feature_statistics_from_plans(all_plans, featurization, dbms_name)
    if is_main_process:
        print(f"✓ Generated feature statistics for {len(feature_statistics)} features")
        print()
    
    # Step 4: Leave-one-out loop
    all_results = []
    
    for fold_idx, test_dataset_name in enumerate(DATASET_NAMES):
        if test_dataset_name not in all_dataset_plans:
            if is_main_process:
                print(f"⚠️  Fold {fold_idx+1}/{len(DATASET_NAMES)}: {test_dataset_name} not found, skipping")
            continue
        
        if is_main_process:
            print("\n" + "="*80)
            print(f"FOLD {fold_idx+1}/{len(DATASET_NAMES)}: Test on {test_dataset_name}")
            print("="*80)
        
        # Prepare train and test plans
        train_plans = []
        for dataset_name, plans in all_dataset_plans.items():
            if dataset_name != test_dataset_name:
                train_plans.extend(plans)
        
        test_plans = all_dataset_plans[test_dataset_name]
        
        if is_main_process:
            print(f"  Train plans: {len(train_plans)} (from {len(all_dataset_plans)-1} datasets)")
            print(f"  Test plans: {len(test_plans)} (from {test_dataset_name})")
        
        # Combine statistics for training datasets
        train_dataset_names = [d for d in DATASET_NAMES if d != test_dataset_name and d in all_statistics]
        db_statistics = {0: all_statistics[train_dataset_names[0]]} if train_dataset_names else {0: StandardizedStatistics()}
        
        # Split train into train/val
        train_size = int(0.85 * len(train_plans))
        val_size = len(train_plans) - train_size
        train_split, val_split = random_split(
            train_plans,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        collate_fn = functools.partial(
            unified_plan_collator,
            dbms_name=dbms_name,
            feature_statistics=feature_statistics,
            db_statistics=db_statistics,
            plan_featurization=featurization
        )
        
        train_dataset = UnifiedPlanDataset([train_plans[i] for i in train_split.indices])
        val_dataset = UnifiedPlanDataset([train_plans[i] for i in val_split.indices])
        test_dataset = UnifiedPlanDataset(test_plans)
        
        if is_distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collate_fn
        )
        
        if is_main_process:
            print(f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        # Create model
        model_config = ZeroShotModelConfig(
            hidden_dim=hidden_dim,
            hidden_dim_plan=hidden_dim,
            hidden_dim_pred=hidden_dim,
            featurization=featurization,
            batch_size=batch_size
        )
        
        encoders = [
            ('column', featurization.COLUMN_FEATURES),
            ('table', featurization.TABLE_FEATURES),
            ('output_column', featurization.OUTPUT_COLUMN_FEATURES),
            ('filter_column', featurization.FILTER_FEATURES + featurization.COLUMN_FEATURES),
            ('plan', featurization.PLAN_FEATURES),
            ('logical_pred', featurization.FILTER_FEATURES),
        ]
        
        prepasses = [
            dict(model_name='table_output_col', e_name='table_output_col', n_dest='output_column'),
            dict(model_name='col_output_col', e_name='col_output_col', n_dest='output_column'),
        ]
        
        add_tree_model_types = ['table_output_col', 'col_output_col']
        
        model = ZeroShotModel(
            model_config=model_config,
            device=str(device_obj),
            feature_statistics=feature_statistics,
            add_tree_model_types=add_tree_model_types,
            prepasses=prepasses,
            plan_featurization=featurization,
            encoders=encoders,
            allow_empty_edges=True
        )
        
        model = model.to(device_obj)
        
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        if is_main_process:
            print(f"\n  🚀 Training...")
        for epoch in range(epochs):
            if is_distributed:
                train_sampler.set_epoch(epoch)
            
            train_loss = train_epoch(model, train_loader, optimizer, device_obj)
            val_loss, median_q_error, rmse = validate(model, val_loader, device_obj)
            
            if is_main_process and (epoch % 10 == 0 or epoch == epochs - 1):
                metrics_str = f"train={train_loss:.4f}, val={val_loss:.4f}"
                if median_q_error is not None:
                    metrics_str += f", q-err={median_q_error:.4f}"
                print(f"    Epoch {epoch+1}/{epochs}: {metrics_str}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                target_model = model.module if is_distributed else model
                best_model_state = copy.deepcopy(target_model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if is_main_process:
                        print(f"    ⏹ Early stopping triggered at epoch {epoch+1}")
                    break
        
        if best_model_state is not None:
            target_model = model.module if is_distributed else model
            target_model.load_state_dict(best_model_state)
        
        # Evaluate on test set (only rank 0)
        if is_main_process:
            eval_model = model.module if is_distributed else model
            test_metrics = evaluate_test(eval_model, test_loader, device_obj, test_dataset_name)
            
            if test_metrics:
                all_results.append(test_metrics)
        
        if is_distributed:
            dist.barrier()
        
        # Clean up model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save all results (only rank 0)
    if is_main_process:
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
    
    if is_distributed:
        dist.barrier()
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Leave-One-Out Cross-Validation for Zero-Shot Model')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing all dataset files')
    # Get available DBMS from registry
    available_dbms = DBMSRegistry.get_cli_choices() if DBMSRegistry.list_plugins() else ['trino', 'postgres', 'mysql']
    parser.add_argument('--dbms', type=str, default='trino',
                       choices=available_dbms,
                       help='DBMS name')
    parser.add_argument('--output_dir', type=str, default='models/zeroshot_loo',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu, cuda:0, or cuda for multi-GPU (use with torchrun)')
    parser.add_argument('--max_plans', type=int, default=10000,
                       help='Maximum plans per dataset')
    parser.add_argument('--statistics_dir', type=str, default='datasets_statistics',
                       help='Statistics directory')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (epochs)')
    
    args = parser.parse_args()
    
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl' if args.device.startswith('cuda') else 'gloo')
    
    result = run_leave_one_out(
        dbms_name=args.dbms,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        device=args.device,
        max_plans_per_dataset=args.max_plans,
        statistics_dir=args.statistics_dir,
        patience=args.patience
    )
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return result


if __name__ == "__main__":
    sys.exit(main())

