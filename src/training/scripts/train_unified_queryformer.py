"""
Unified QueryFormer Training Script

This script works for all DBMS (PostgreSQL, Trino, MySQL, etc.)

Usage:
    python -m training.scripts.train_unified_queryformer \
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
import functools
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import numpy as np

# Initialize plugin system
import core.init_plugins

from core.plugins.registry import DBMSRegistry
from core.statistics.schema import StandardizedStatistics
from models.query_former.unified_queryformer_dataloader import unified_queryformer_collator
from models.query_former.model import QueryFormer
from classes.classes import QueryFormerModelConfig, InputDims
from training.unified_featurizations import UnifiedQueryFormerFeaturization
from training.training.metrics import QError, RMSE


class UnifiedPlanDataset(Dataset):
    def __init__(self, plans):
        self.plans = plans
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx):
        return idx, self.plans[idx]


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
            plan.database_id = 0
        
        if max_plans and len(parsed_plans) > max_plans:
            parsed_plans = parsed_plans[:max_plans]
        
        all_plans.extend(parsed_plans)
    
    return all_plans


def create_feature_statistics(plans, featurization):
    """Generate feature statistics from plans."""
    print("📊 Collecting feature statistics...")
    
    from sklearn.preprocessing import RobustScaler
    from core.features.mapper import FeatureMapper
    
    actual_op_names = set()
    filter_operators = set()
    numeric_feature_values = {
        'estimated_cardinality': [],
        'estimated_cost': [],
        'estimated_width': [],
    }
    
    def collect_from_node(node):
        if hasattr(node, 'plan_parameters'):
            params = node.plan_parameters
            op = params.get('op_name') if isinstance(params, dict) else getattr(params, 'op_name', None)
            if op:
                actual_op_names.add(op)
            
            # Collect numeric values
            mapper = FeatureMapper('trino')
            for feat_name in numeric_feature_values.keys():
                try:
                    value = mapper.get_feature(feat_name, params)
                    if value is not None and not isinstance(value, str):
                        numeric_feature_values[feat_name].append(float(value))
                except:
                    pass
            
            # Collect filter operators
            filter_col = params.get('filter_columns') if isinstance(params, dict) else getattr(params, 'filter_columns', None)
            if filter_col:
                def collect_filter_ops(fc):
                    op = fc.get('operator') if isinstance(fc, dict) else getattr(fc, 'operator', None)
                    if op:
                        filter_operators.add(str(op))
                    children = fc.get('children', []) if isinstance(fc, dict) else getattr(fc, 'children', [])
                    for child in children:
                        collect_filter_ops(child)
                collect_filter_ops(filter_col)
        
        for child in node.children:
            collect_from_node(child)
    
    for plan in plans:
        collect_from_node(plan)
    
    print(f"  Found {len(actual_op_names)} operator types")
    print(f"  Found {len(filter_operators)} filter operators")
    
    feature_statistics = {
        'op_name': {
            'type': 'categorical',
            'value_dict': {op: i for i, op in enumerate(sorted(actual_op_names))},
            'no_vals': len(actual_op_names)
        },
        'operator': {
            'type': 'categorical',
            'value_dict': {op: i for i, op in enumerate(sorted(filter_operators))} if filter_operators else {'=': 0},
            'no_vals': max(len(filter_operators), 1)
        },
        'column': {
            'max': 100,
            'type': 'numeric'
        }
    }
    
    # Aliases
    feature_statistics['operator_type'] = feature_statistics['op_name']
    
    # Numeric features with actual statistics
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
            print(f"    {feat_name}: max={values_array.max():.2f}, samples={len(values)}")
        else:
            feature_statistics[feat_name] = {
                'type': 'numeric',
                'max': 1.0,
                'scale': 1.0,
                'center': 0.0
            }
    
    return feature_statistics


def run_training(
    dbms_name: str,
    train_files: list,
    test_files: list,
    schema_name: str,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    hidden_dim: int = 256,
    learning_rate: float = 0.001,
    device: str = "cuda:0",
    max_plans: int = None
):
    """Train QueryFormer model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"Unified QueryFormer Training for {dbms_name.upper()}")
    print("="*80)
    print()
    
    # Load plans
    print("📂 Loading plans")
    train_plans = load_plans_from_txt(train_files, dbms_name, max_plans)
    test_plans = load_plans_from_txt(test_files, dbms_name, max_plans)
    print(f"✓ Train: {len(train_plans)}, Test: {len(test_plans)}")
    print()
    
    # Feature statistics
    all_plans = train_plans + test_plans
    feature_statistics = create_feature_statistics(all_plans, UnifiedQueryFormerFeaturization)
    print(f"✓ Generated {len(feature_statistics)} feature types")
    print()
    
    # Load database statistics
    print("📊 Loading database statistics")
    try:
        if dbms_name == "trino":
            from models.zeroshot.utils.statistics_loader import load_database_statistics_for_zeroshot
            raw_stats = load_database_statistics_for_zeroshot(
                dataset=schema_name,
                stats_dir='datasets_statistics',
                prefer_zero_shot=True
            )
        else:
            from models.zeroshot.utils.statistics_loader import load_database_statistics_for_zeroshot
            raw_stats = load_database_statistics_for_zeroshot(
                dataset=schema_name,
                stats_dir='datasets_statistics'
            )
        
        converter = DBMSRegistry.get_statistics_converter(dbms_name)
        db_stats = {0: converter.convert(raw_stats)}
        
        stats = db_stats[0]
        print(f"✓ Loaded statistics: {len(stats.column_stats)} columns, {len(stats.table_stats)} tables")
    except Exception as e:
        print(f"⚠️  Statistics loading failed: {e}")
        db_stats = {0: StandardizedStatistics()}
    
    print()
    
    # Split train/val
    print("📦 Creating dataloaders")
    train_size = int(0.85 * len(train_plans))
    val_size = len(train_plans) - train_size
    train_split, val_split = random_split(
        train_plans, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create collator
    collate_fn = functools.partial(
        unified_queryformer_collator,
        dbms_name=dbms_name,
        feature_statistics=feature_statistics,
        db_statistics=db_stats
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
    
    print(f"✓ Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    print()
    
    # Create model
    print("🤖 Creating QueryFormer model")
    
    config = QueryFormerModelConfig(
        hidden_dim_plan=hidden_dim,
        device=device
    )
    
    input_dims = InputDims(
        dim_word_hash=1000,
        dim_word_emdb=64,
        dim_bitmaps=1000
    )
    
    model = QueryFormer(config, input_dims, feature_statistics, label_norm=None)
    model = model.to(device)
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print()
    
    # Training
    print("🚀 Training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        
        for batch, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            labels = labels.to(device)
            
            predictions = model(batch)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch, labels, _ in val_loader:
                batch = tuple(b.to(device) for b in batch)
                labels = labels.to(device)
                
                predictions = model(batch)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        # Metrics
        if all_preds:
            all_preds = np.concatenate(all_preds).flatten()
            all_labels = np.concatenate(all_labels).flatten()
            safe_preds = np.clip(all_preds, 0.1, np.inf)
            safe_labels = np.clip(all_labels, 0.1, np.inf)
            median_q = QError(percentile=50, min_val=0.1).evaluate_metric(safe_labels, safe_preds)
            rmse = RMSE().evaluate_metric(all_labels, all_preds)
            
            print(f"Epoch {epoch+1}/{epochs}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, "
                  f"q-err={median_q:.4f}, rmse={rmse:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / f"{dbms_name}_queryformer_best.pt")
    
    print(f"\n✅ Training complete! Best val_loss: {best_val_loss:.4f}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--schema', type=str, required=True)
    parser.add_argument('--dbms', type=str, default='trino')
    parser.add_argument('--output_dir', type=str, default='models/unified_queryformer')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_plans', type=int, default=None)
    
    args = parser.parse_args()
    
    return run_training(
        dbms_name=args.dbms,
        train_files=[Path(p.strip()) for p in args.train_files.split(',')],
        test_files=[Path(args.test_file)],
        schema_name=args.schema,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        device=args.device,
        max_plans=args.max_plans
    )


if __name__ == "__main__":
    sys.exit(main())

