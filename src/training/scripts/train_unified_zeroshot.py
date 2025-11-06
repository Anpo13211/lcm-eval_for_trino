"""
Unified Trino Zero-Shot Training Script using Plugin Architecture

This script demonstrates how to train models on Trino data using the new
plugin-based architecture. The same code works for PostgreSQL with minimal changes.

Usage:
    python -m trino_lcm.scripts.train_unified_zeroshot \
        --train_files /path/to/accidents_*.txt \
        --test_file /path/to/accidents_test.txt \
        --schema accidents \
        --output_dir models/trino_unified
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchdata')

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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

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


class UnifiedPlanDataset(Dataset):
    """Universal plan dataset that works for all DBMS."""
    
    def __init__(self, plans):
        self.plans = plans
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx):
        return idx, self.plans[idx]


def load_plans_from_txt(file_paths: list, dbms_name: str, max_plans_per_file: int = None):
    """
    Load plans from EXPLAIN ANALYZE txt files.
    
    Args:
        file_paths: List of .txt file paths
        dbms_name: DBMS name ("trino", "postgres", etc.)
        max_plans_per_file: Maximum plans to load per file
    
    Returns:
        List of parsed plans
    """
    parser = DBMSRegistry.get_parser(dbms_name)
    
    all_plans = []
    
    for file_path in file_paths:
        print(f"  Loading from {Path(file_path).name}...")
        
        parsed_plans, runtimes = parser.parse_explain_analyze_file(
            str(file_path),
            min_runtime=0,
            max_runtime=float('inf')
        )
        
        # Set plan_runtime for each plan
        for plan, runtime in zip(parsed_plans, runtimes):
            plan.plan_runtime = runtime
            plan.database_id = 0  # Single database for now
        
        # Limit if needed
        if max_plans_per_file and len(parsed_plans) > max_plans_per_file:
            parsed_plans = parsed_plans[:max_plans_per_file]
        
        all_plans.extend(parsed_plans)
        print(f"    Loaded {len(parsed_plans)} plans")
    
    print(f"  Total plans loaded: {len(all_plans)}")
    return all_plans


def create_feature_statistics_from_plans(plans, plan_featurization):
    """
    Generate feature statistics from plans.
    
    Args:
        plans: List of plan operators
        plan_featurization: Featurization configuration
    
    Returns:
        feature_statistics dict
    """
    print("📊 Collecting feature statistics from plans...")
    
    # Collect actual operators used
    actual_op_names = set()
    filter_operators = set()
    aggregations = set()
    data_types = set()
    
    def collect_from_node(node):
        if hasattr(node, 'plan_parameters'):
            params = node.plan_parameters
            
            # Get op_name
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
    
    print(f"  Found {len(actual_op_names)} operator types")
    print(f"  Found {len(filter_operators)} filter operators")
    print(f"  Found {len(aggregations)} aggregation types")
    
    # Collect numeric feature values from actual plans
    from sklearn.preprocessing import RobustScaler
    
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
        """Collect actual values from plans."""
        if hasattr(node, 'plan_parameters'):
            params = node.plan_parameters
            
            # Use FeatureMapper to extract values
            from core.features.mapper import FeatureMapper
            mapper = FeatureMapper('trino')  # Will work for any DBMS
            
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
    
    # Use ONLY logical names - with actual statistics from plans
    for feat_name, values in numeric_feature_values.items():
        if len(values) > 0:
            values_array = np.array(values, dtype=np.float32).reshape(-1, 1)
            
            # Use RobustScaler to compute statistics
            scaler = RobustScaler()
            scaler.fit(values_array)
            
            feature_statistics[feat_name] = {
                'type': 'numeric',
                'max': float(values_array.max()),
                'scale': float(scaler.scale_.item()),
                'center': float(scaler.center_.item())
            }
            print(f"    {feat_name}: max={values_array.max():.2f}, scale={scaler.scale_.item():.2f}, samples={len(values)}")
        else:
            # No values found, use defaults
            feature_statistics[feat_name] = {
                'type': 'numeric',
                'max': 1.0,
                'scale': 1.0,
                'center': 0.0
            }
    
    # Categorical features
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
    
    print(f"  ✓ Collected statistics from {len(plans)} plans")
    
    return feature_statistics


def load_database_statistics(schema_name: str, dbms_name: str, statistics_dir: str = None):
    """
    Load database statistics and convert to StandardizedStatistics.
    
    Args:
        schema_name: Schema/database name
        dbms_name: DBMS name
        statistics_dir: Statistics directory path
    
    Returns:
        Dictionary mapping database_id to StandardizedStatistics
    """
    if dbms_name == "trino":
        # Use Trino-specific loader
        from models.zeroshot.utils.statistics_loader import load_database_statistics_for_zeroshot
        
        # Pass schema_name as-is (function will add iceberg_ prefix if needed)
        raw_stats = load_database_statistics_for_zeroshot(
            dataset=schema_name,  # e.g., "accidents" → will look in "iceberg_accidents"
            stats_dir=statistics_dir or 'datasets_statistics',
            prefer_zero_shot=True
        )
    elif dbms_name == "postgres":
        # Use PostgreSQL-specific loader
        from models.zeroshot.utils.statistics_loader import load_database_statistics_for_zeroshot
        
        raw_stats = load_database_statistics_for_zeroshot(
            dataset=schema_name,
            stats_dir=statistics_dir or 'datasets_statistics'
        )
    else:
        raise ValueError(f"Statistics loading not implemented for DBMS: {dbms_name}")
    
    # Convert to StandardizedStatistics
    converter = DBMSRegistry.get_statistics_converter(dbms_name)
    standardized = converter.convert(raw_stats)
    
    return {0: standardized}


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        # Forward pass
        try:
            # Collator returns (graph, features, labels, sample_idxs)
            graph, features, labels, sample_idxs = batch
            
            # Move to device (same as validation)
            graph = graph.to(device)
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            
            # Model expects (graph, features) tuple
            predictions = model((graph, features))
            
            # Compute loss
            loss = model.loss_fxn(predictions, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        except Exception as e:
            print(f"⚠️  Batch failed: {e}")
            import traceback
            traceback.print_exc()
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
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                # Collator returns (graph, features, labels, sample_idxs)
                graph, features, labels, sample_idxs = batch
                
                # Move to device
                graph = graph.to(device)
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)
                
                # Model expects (graph, features) tuple
                predictions = model((graph, features))
                
                loss = model.loss_fxn(predictions, labels)
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
            except Exception as e:
                print(f"⚠️  Validation batch failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Calculate metrics
    if len(all_predictions) > 0:
        all_predictions = np.concatenate(all_predictions).flatten()
        all_labels = np.concatenate(all_labels).flatten()
        
        # Clip to avoid division by zero
        min_val = 0.1  # 100ms
        safe_preds = np.clip(all_predictions, min_val, np.inf)
        safe_labels = np.clip(all_labels, min_val, np.inf)
        
        # Q-Error
        median_q_error = QError(percentile=50, min_val=min_val).evaluate_metric(
            labels=safe_labels, preds=safe_preds
        )
        
        # RMSE
        rmse = RMSE().evaluate_metric(labels=all_labels, preds=all_predictions)
        
        return avg_loss, median_q_error, rmse
    
    return avg_loss, None, None


def evaluate_test(model, test_loader, device):
    """Evaluate model on test set with detailed metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            try:
                graph, features, labels, sample_idxs = batch
                
                # Move to device
                graph = graph.to(device)
                features = {k: v.to(device) for k, v in features.items()}
                
                predictions = model((graph, features))
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
            except Exception as e:
                print(f"⚠️  Test batch failed: {e}")
                continue
    
    if len(all_predictions) == 0:
        return None
    
    all_predictions = np.concatenate(all_predictions).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # Clip for Q-Error calculation
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
    
    print("\n" + "="*80)
    print("📊 Test Set Evaluation Results")
    print("="*80)
    print(f"  Samples: {len(all_labels)}")
    print(f"  RMSE: {rmse:.4f} sec ({rmse*1000:.2f} ms)")
    print(f"  Median Q-Error: {median_q_error:.4f}")
    print(f"  Mean Q-Error: {mean_q_error:.4f}")
    print(f"  P95 Q-Error: {p95_q_error:.4f}")
    print(f"  P99 Q-Error: {p99_q_error:.4f}")
    print(f"  Max Q-Error: {max_q_error:.4f}")
    print("="*80)
    
    return {
        'rmse': float(rmse) if rmse is not None else None,
        'median_q_error': float(median_q_error) if median_q_error is not None else None,
        'mean_q_error': float(mean_q_error) if mean_q_error is not None else None,
        'p95_q_error': float(p95_q_error) if p95_q_error is not None else None,
        'p99_q_error': float(p99_q_error) if p99_q_error is not None else None,
        'max_q_error': float(max_q_error) if max_q_error is not None else None,
        'num_samples': int(len(all_labels))
    }


def run_training(
    dbms_name: str,
    train_files: list,
    test_files: list,
    schema_name: str,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 32,
    hidden_dim: int = 128,
    learning_rate: float = 0.001,
    device: str = "cuda:0",
    max_plans_per_file: int = None,
    statistics_dir: str = None
):
    """
    Main training function using unified architecture.
    
    Args:
        dbms_name: DBMS name ("trino", "postgres", etc.)
        train_files: List of training file paths
        test_files: List of test file paths
        schema_name: Schema/database name
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        hidden_dim: Hidden dimension
        learning_rate: Learning rate
        device: Device string
        max_plans_per_file: Maximum plans per file
        statistics_dir: Statistics directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"Unified Zero-Shot Training for {dbms_name.upper()}")
    print("="*80)
    print(f"DBMS: {dbms_name}")
    print(f"Schema: {schema_name}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Device: {device}")
    print("="*80)
    print()
    
    # Step 1: Load plans
    print("📂 Step 1: Loading plans")
    print("-"*80)
    
    train_plans = load_plans_from_txt(train_files, dbms_name, max_plans_per_file)
    test_plans = load_plans_from_txt(test_files, dbms_name, max_plans_per_file)
    
    print(f"✓ Loaded {len(train_plans)} training plans")
    print(f"✓ Loaded {len(test_plans)} test plans")
    print()
    
    # Step 2: Load database statistics
    print("📊 Step 2: Loading database statistics")
    print("-"*80)
    
    try:
        db_statistics = load_database_statistics(schema_name, dbms_name, statistics_dir)
        stats = db_statistics[0]
        print(f"✓ Loaded statistics: {len(stats.column_stats)} columns, {len(stats.table_stats)} tables")
        
        if len(stats.column_stats) == 0:
            print(f"⚠️  No column statistics found for schema: {schema_name}")
            print(f"   Checked dataset name: iceberg_{schema_name}")
            print(f"   Statistics dir: {statistics_dir}")
    except Exception as e:
        print(f"⚠️  Statistics loading failed: {e}")
        import traceback
        traceback.print_exc()
        db_statistics = {0: StandardizedStatistics()}
    
    print()
    
    # Step 3: Generate feature statistics
    print("🔧 Step 3: Generating feature statistics")
    print("-"*80)
    
    # Use unified featurization
    featurization = UnifiedTrueCardDetail
    
    # Gather statistics from training plans
    all_plans_for_stats = train_plans + test_plans
    
    feature_statistics = create_feature_statistics_from_plans(
        all_plans_for_stats,
        featurization
    )
    
    print(f"✓ Generated feature statistics for {len(feature_statistics)} features")
    print()
    
    # Step 4: Create dataloaders
    print("📦 Step 4: Creating dataloaders")
    print("-"*80)
    
    # Split training into train/val
    from torch.utils.data import random_split
    
    train_size = int(0.85 * len(train_plans))
    val_size = len(train_plans) - train_size
    
    train_split, val_split = random_split(
        train_plans,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create unified collator (works for all DBMS!)
    from core.graph.unified_collator import unified_plan_collator
    
    collate_fn = functools.partial(
        unified_plan_collator,
        dbms_name=dbms_name,
        feature_statistics=feature_statistics,
        db_statistics=db_statistics,
        plan_featurization=featurization
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
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    print()
    
    # Step 5: Create model
    print("🤖 Step 5: Creating model")
    print("-"*80)
    
    model_config = ZeroShotModelConfig(
        hidden_dim=hidden_dim,
        hidden_dim_plan=hidden_dim,
        hidden_dim_pred=hidden_dim,
        featurization=featurization,
        batch_size=batch_size
    )
    
    # Create encoders
    encoders = [
        ('column', featurization.COLUMN_FEATURES),
        ('table', featurization.TABLE_FEATURES),
        ('output_column', featurization.OUTPUT_COLUMN_FEATURES),
        ('filter_column', featurization.FILTER_FEATURES + featurization.COLUMN_FEATURES),
        ('plan', featurization.PLAN_FEATURES),
        ('logical_pred', featurization.FILTER_FEATURES),
    ]
    
    # Prepasses for message passing
    prepasses = [
        dict(model_name='table_output_col', e_name='table_output_col', n_dest='output_column'),
        dict(model_name='col_output_col', e_name='col_output_col', n_dest='output_column'),
    ]
    
    # Add these prepasses to tree_model_types
    add_tree_model_types = ['table_output_col', 'col_output_col']
    
    model = ZeroShotModel(
        model_config=model_config,
        device=device,
        feature_statistics=feature_statistics,
        add_tree_model_types=add_tree_model_types,
        prepasses=prepasses,
        plan_featurization=featurization,
        encoders=encoders,
        allow_empty_edges=True  # For Trino compatibility
    )
    
    model = model.to(device)
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print()
    
    # Step 6: Training
    print("🚀 Step 6: Training")
    print("-"*80)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, median_q_error, rmse = validate(model, val_loader, device)
        
        # Display metrics
        metrics_str = f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        if median_q_error is not None:
            metrics_str += f", median_qerr={median_q_error:.4f}"
        if rmse is not None:
            metrics_str += f", rmse={rmse:.4f}"
        
        print(f"Epoch {epoch+1}/{epochs}: {metrics_str}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = output_dir / f"{dbms_name}_zeroshot_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'median_q_error': median_q_error,
                'rmse': rmse,
            }, model_path)
            q_err_str = f"{median_q_error:.4f}" if median_q_error is not None else "N/A"
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f}, q-err={q_err_str})")
    
    print()
    print(f"✅ Training complete! Best val_loss: {best_val_loss:.4f}")
    
    # Final evaluation on test set
    if test_loader:
        print("\n" + "="*80)
        print("📊 Final Test Set Evaluation")
        print("="*80)
        
        # Load best model
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = evaluate_test(model, test_loader, device)
        
        # Save test results
        results_path = output_dir / f"{dbms_name}_test_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        print(f"\n✓ Test results saved to {results_path}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Unified Zero-Shot Training')
    
    # Required arguments
    parser.add_argument('--train_files', type=str, required=True,
                       help='Training file paths (comma-separated)')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Test file path')
    parser.add_argument('--schema', type=str, required=True,
                       help='Schema/database name')
    
    # Optional arguments
    parser.add_argument('--dbms', type=str, default='trino',
                       choices=['trino', 'postgres', 'mysql'],
                       help='DBMS name')
    parser.add_argument('--output_dir', type=str, default='models/unified_zeroshot',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device (cuda:0, cpu, etc.)')
    parser.add_argument('--max_plans', type=int, default=None,
                       help='Maximum plans per file')
    parser.add_argument('--statistics_dir', type=str, default='datasets_statistics',
                       help='Statistics directory')
    
    args = parser.parse_args()
    
    # Parse file paths
    train_files = [Path(p.strip()) for p in args.train_files.split(',')]
    test_files = [Path(args.test_file)]
    output_dir = Path(args.output_dir)
    
    # Run training
    return run_training(
        dbms_name=args.dbms,
        train_files=train_files,
        test_files=test_files,
        schema_name=args.schema,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        device=args.device,
        max_plans_per_file=args.max_plans,
        statistics_dir=args.statistics_dir
    )


if __name__ == "__main__":
    sys.exit(main())

