"""
QPPNet Training Script for Trino

This script trains QPPNet model on Trino query plans using global loss (query-level runtime).
Unlike PostgreSQL version, it doesn't require operator-level wall times.

Usage:
    python -m training.scripts.train_qppnet_trino \
        --train_files data/runs/trino/accidents_train.txt \
        --schema accidents \
        --output_dir models/qppnet_trino \
        --epochs 100
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

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
import json
import functools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Initialize plugin system
import core.init_plugins

# Import components
from core.plugins.registry import DBMSRegistry
from models.qppnet.qppnet_model import QPPNet
from models.qppnet.qppnet_dataloader import create_qppnet_dataloader
from models.qppnet.trino_adapter import TRINO_TO_POSTGRES_OP_MAPPING
from classes.classes import QPPNetModelConfig, DataLoaderOptions
from classes.workload_runs import WorkloadRuns
from training.featurizations import QPPNetFeaturization
from training.preprocessing.feature_statistics import FeatureType
from training.training.metrics import QError
from core.capabilities import check_capabilities


def parse_args():
    parser = argparse.ArgumentParser(description='Train QPPNet on Trino plans')
    
    # Data arguments
    parser.add_argument('--train_files', type=str, required=True, nargs='+',
                        help='Training plan files (EXPLAIN ANALYZE output)')
    parser.add_argument('--test_files', type=str, nargs='+', default=None,
                        help='Test plan files (optional)')
    parser.add_argument('--schema', type=str, required=True,
                        help='Schema name (e.g., accidents, airline)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='models/qppnet_trino',
                        help='Output directory for model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for neural units')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'qerror'],
                        help='Loss function type')
    
    # Data arguments
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--limit_queries', type=int, default=None,
                        help='Limit number of training queries')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    
    return parser.parse_args()


def load_trino_plans(file_paths, schema_name, dbms_name='trino'):
    """Load Trino plans from EXPLAIN ANALYZE files."""
    parser = DBMSRegistry.get_parser(dbms_name)
    
    all_plans = []
    all_runtimes = []
    
    for file_path in file_paths:
        print(f"Loading from {Path(file_path).name}...")
        
        # Read file and split by query delimiters
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split by statement delimiters (-- ... stmt X)
        queries = []
        current_query = []
        for line in content.split('\n'):
            if line.startswith('-- ') and 'stmt' in line:
                if current_query:
                    queries.append('\n'.join(current_query))
                    current_query = []
            else:
                current_query.append(line)
        if current_query:
            queries.append('\n'.join(current_query))
        
        print(f"  Found {len(queries)} queries in file")
        
        # Parse each query
        parsed_count = 0
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
                    
                    all_plans.append(root_operator)
                    all_runtimes.append(execution_time)
                    parsed_count += 1
            except Exception as e:
                # Skip failed queries
                print(f"    Warning: Failed to parse query: {str(e)[:50]}...")
                continue
        
        print(f"  Successfully loaded {parsed_count}/{len(queries)} plans")
    
    print(f"Total: {len(all_plans)} plans loaded")
    return all_plans, all_runtimes


def load_column_statistics(schema_name, base_dir='datasets_statistics'):
    """Load column statistics from datasets_statistics directory."""
    stats_dir = Path(base_dir) / f'iceberg_{schema_name}'
    column_stats_file = stats_dir / 'column_stats.json'
    
    if not column_stats_file.exists():
        print(f"Warning: Column statistics not found at {column_stats_file}")
        return {}
    
    with open(column_stats_file, 'r') as f:
        column_stats_raw = json.load(f)
    
    # Convert from "('table', 'column')" format to nested dict format
    # Expected format: {table_name: {column_name: {stats...}}}
    column_stats = {}
    for key, stats in column_stats_raw.items():
        # Parse the tuple string key
        if key.startswith("('") and key.endswith("')"):
            # Remove parens and quotes
            key_clean = key[2:-2]  # "('table', 'column')" -> "table', 'column"
            parts = key_clean.split("', '")  # ["table", "column"]
            if len(parts) == 2:
                table_name, column_name = parts
                
                if table_name not in column_stats:
                    column_stats[table_name] = {}
                
                # Add compatibility key: num_unique (used by OperatorTree.map_column_statistics)
                # Trino uses 'distinct_values_count', PostgreSQL uses 'num_unique'
                if 'num_unique' not in stats and 'distinct_values_count' in stats:
                    stats['num_unique'] = stats['distinct_values_count'] or 0
                elif 'num_unique' not in stats:
                    stats['num_unique'] = 0
                
                column_stats[table_name][column_name] = stats
    
    print(f"Loaded column statistics from {column_stats_file}")
    print(f"  Tables: {list(column_stats.keys())}")
    return column_stats


def collect_feature_statistics_from_plans(plans, featurization):
    """Collect feature statistics from loaded plans."""
    print("Collecting feature statistics from plans...")
    
    from collections import defaultdict
    from sklearn.preprocessing import RobustScaler
    from training.preprocessing.feature_statistics import FeatureType
    
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
                        value = None  # 0は統計として使わない
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
                    # テーブル名がある場合のみ追加
                elif feature == 'Join Type':
                    # Infer from op_name
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
                    value = 0  # Will be filled from column_statistics
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
    
    # Add missing features with default values (for operators that don't appear in data)
    all_required_features = set()
    for op_type, features in featurization.QPP_NET_OPERATOR_TYPES.items():
        all_required_features.update(features)
    
    for feature in all_required_features:
        if feature not in statistics_dict:
            # Add default statistics for missing features
            if feature in ['Plan Width', 'Plan Rows', 'Total Cost', 'Actual Rows', 
                          'Hash Buckets', 'Peak Memory Usage', 'Min', 'Max', 'Mean']:
                # Numeric features
                statistics_dict[feature] = {
                    'max': 1.0,
                    'scale': 1.0,
                    'center': 0.0,
                    'type': str(FeatureType.numeric)
                }
            else:
                # Categorical features
                default_values = {
                    'Relation Name': ['nesreca', 'oseba', 'upravna_enota', 'unknown'],  # Accidents tables
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
    
    print(f"Collected statistics for {len(statistics_dict)} features (including defaults)")
    return statistics_dict


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        query_plans, labels, sample_idxs = batch
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        
        # Forward pass
        predictions = model(query_plans)
        
        # Compute loss using global loss function
        loss = model.loss_fxn(predictions, labels)
        
        # Backward pass
        if optimizer is not None:
            optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        model.backward()  # Step operator-level optimizers
        
        if optimizer is not None:
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def evaluate(model, val_loader, device):
    """Evaluate on validation set."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            query_plans, labels, sample_idxs = batch
            
            predictions = model(query_plans)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Q-Error
    q_errors = np.maximum(all_predictions / all_labels, all_labels / all_predictions)
    median_qerror = np.median(q_errors)
    mean_qerror = np.mean(q_errors)
    
    return {
        'median_qerror': median_qerror,
        'mean_qerror': mean_qerror,
        'predictions': all_predictions,
        'labels': all_labels
    }


def main():
    args = parse_args()
    dbms_name = 'trino'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("QPPNet Training for Trino")
    print("=" * 80)
    print(f"Schema: {args.schema}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
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
    
    # Load training plans
    print("📂 Loading training plans...")
    train_plans, train_runtimes = load_trino_plans(args.train_files, args.schema)
    
    # Load test plans if provided
    if args.test_files:
        print("📂 Loading test plans...")
        test_plans, test_runtimes = load_trino_plans(args.test_files, args.schema)
    else:
        test_plans = None
    
    # Load column statistics
    print("\n📊 Loading column statistics...")
    column_statistics = load_column_statistics(args.schema)
    
    # Create featurization
    featurization = QPPNetFeaturization()
    
    # Collect feature statistics
    print("\n📊 Collecting feature statistics...")
    feature_statistics = collect_feature_statistics_from_plans(train_plans, featurization)
    
    # Save feature statistics
    feature_stats_path = output_dir / 'feature_statistics.json'
    with open(feature_stats_path, 'w') as f:
        json.dump(feature_statistics, f, indent=2)
    print(f"Saved feature statistics to {feature_stats_path}")
    
    # Add scalers for numeric features (after saving JSON)
    # This is required by OperatorTree.encode_features
    from sklearn.preprocessing import RobustScaler
    for k, v in feature_statistics.items():
        if v.get('type') == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = np.array([v['center']])
            scaler.scale_ = np.array([v['scale']])
            feature_statistics[k]['scaler'] = scaler
    
    # Save column statistics
    column_stats_path = output_dir / 'column_statistics.json'
    with open(column_stats_path, 'w') as f:
        json.dump(column_statistics, f, indent=2)
    print(f"Saved column statistics to {column_stats_path}")
    
    # Create model config
    model_config = QPPNetModelConfig(
        name='qppnet_trino',
        featurization=featurization,
        hidden_dim_plan=args.hidden_dim,
        batch_size=args.batch_size,
        num_workers=0,
        device=args.device,
        loss_class_name='QPPLossGlobal',  # Use global loss
        loss_class_kwargs={'loss_type': args.loss_type},
        optimizer_kwargs={'lr': args.lr},
        limit_queries=args.limit_queries,
        execution_mode=None
    )
    
    # Create dataloaders directly from plans (bypass workload_runs)
    print("\n🔨 Creating data loaders...")
    
    from torch.utils.data import DataLoader
    from training.dataset.plan_dataset import PlanDataset
    import functools
    from models.qppnet.qppnet_dataloader import qppnet_collator
    
    # Split into train/val
    num_plans = len(train_plans)
    train_size = int(num_plans * (1 - args.val_ratio))
    
    train_dataset = PlanDataset(train_plans[:train_size], list(range(train_size)))
    val_dataset = PlanDataset(train_plans[train_size:], list(range(train_size, num_plans)))
    
    print(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # Create collate function
    train_collate_fn = functools.partial(
        qppnet_collator,
        db_statistics={},
        feature_statistics=feature_statistics,
        column_statistics=column_statistics,
        plan_featurization=model_config.featurization,
        dbms_name='trino'  # ★ Registry-aware (changed from use_trino=True)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=train_collate_fn
    )
    
    # Create dummy workload runs for model initialization
    workload_runs = WorkloadRuns(
        train_workload_runs=[],
        test_workload_runs=[]
    )
    
    # Create model
    print("\n🤖 Initializing QPPNet model...")
    model = QPPNet(
        model_config=model_config,
        workload_runs=workload_runs,
        feature_statistics=feature_statistics,
        label_norm=None,
        use_global_loss=True  # ★ Enable global loss mode
    )
    model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer (global optimizer for backward compatibility)
    # Note: Individual operator optimizers are managed by model.backward()
    
    # Training loop
    print("\n🏋️  Starting training...")
    best_val_qerror = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, None, args.device, epoch)
        
        # Validate
        val_metrics = evaluate(model, val_loader, args.device)
        
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Median Q-Error: {val_metrics['median_qerror']:.2f}")
        
        # Save best model
        if val_metrics['median_qerror'] < best_val_qerror:
            best_val_qerror = val_metrics['median_qerror']
            model_path = output_dir / 'best_model.pt'
            torch.save(model.state_dict(), model_path)
            print(f"  💾 Saved best model (Q-Error: {best_val_qerror:.2f})")
    
    print("\n✅ Training complete!")
    print(f"Best validation Q-Error: {best_val_qerror:.2f}")
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)


if __name__ == '__main__':
    main()

