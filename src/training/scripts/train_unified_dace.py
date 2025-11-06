"""
Unified DACE Training Script using Plugin Architecture

This script works for all DBMS (PostgreSQL, Trino, MySQL, etc.)

Usage:
    python -m training.scripts.train_unified_dace \
        --dbms trino \
        --train_files accidents.txt \
        --test_file accidents.txt \
        --schema accidents \
        --output_dir models/dace_unified
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
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import numpy as np

# Initialize plugin system
import core.init_plugins

from core.plugins.registry import DBMSRegistry
from models.dace.unified_dace_collator import unified_dace_collator
from training.unified_featurizations import UnifiedDACEFeaturization
from models.dace.dace_model import DACELora
from classes.classes import DACEModelConfig
from training.training.metrics import QError, RMSE


class UnifiedPlanDataset(Dataset):
    """Universal plan dataset."""
    
    def __init__(self, plans):
        self.plans = plans
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx):
        return idx, self.plans[idx]


def load_plans_from_txt(file_paths: list, dbms_name: str, max_plans_per_file: int = None):
    """Load plans from EXPLAIN ANALYZE txt files."""
    parser = DBMSRegistry.get_parser(dbms_name)
    
    all_plans = []
    for file_path in file_paths:
        print(f"  Loading from {Path(file_path).name}...")
        
        parsed_plans, runtimes = parser.parse_explain_analyze_file(
            str(file_path), min_runtime=0, max_runtime=float('inf')
        )
        
        for plan, runtime in zip(parsed_plans, runtimes):
            plan.plan_runtime = runtime
            plan.database_id = 0
        
        if max_plans_per_file and len(parsed_plans) > max_plans_per_file:
            parsed_plans = parsed_plans[:max_plans_per_file]
        
        all_plans.extend(parsed_plans)
        print(f"    Loaded {len(parsed_plans)} plans")
    
    print(f"  Total plans loaded: {len(all_plans)}")
    return all_plans


def create_feature_statistics_from_plans(plans, plan_featurization):
    """Generate feature statistics from plans."""
    print("📊 Collecting feature statistics...")
    
    from sklearn.preprocessing import RobustScaler
    from core.features.mapper import FeatureMapper
    
    actual_op_names = set()
    numeric_feature_values = {
        'estimated_cardinality': [],
        'estimated_cost': [],
        'estimated_width': [],
    }
    
    def collect_from_node(node):
        if hasattr(node, 'plan_parameters'):
            params = node.plan_parameters
            op_name = params.get('op_name') if isinstance(params, dict) else getattr(params, 'op_name', None)
            if op_name:
                actual_op_names.add(op_name)
            
            # Collect numeric values
            mapper = FeatureMapper('trino')
            for feat_name in numeric_feature_values.keys():
                try:
                    value = mapper.get_feature(feat_name, params)
                    if value is not None and not isinstance(value, str):
                        numeric_feature_values[feat_name].append(float(value))
                except:
                    pass
        
        for child in node.children:
            collect_from_node(child)
    
    for plan in plans:
        collect_from_node(plan)
    
    print(f"  Found {len(actual_op_names)} operator types")
    
    # Build feature statistics
    feature_statistics = {}
    
    # op_name (categorical)
    feature_statistics['op_name'] = {
        'type': 'categorical',
        'value_dict': {op: i for i, op in enumerate(sorted(actual_op_names))},
        'no_vals': len(actual_op_names)
    }
    
    feature_statistics['operator_type'] = feature_statistics['op_name']
    
    # Numeric features - collect actual statistics from plans
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
            print(f"    {feat_name}: max={values_array.max():.2f}, scale={scaler.scale_.item():.2f}, samples={len(values)}")
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
    batch_size: int = 32,
    hidden_dim: int = 128,
    node_length: int = 22,
    learning_rate: float = 0.001,
    device: str = "cuda:0",
    max_plans_per_file: int = None
):
    """Main training function for unified DACE."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"Unified DACE Training for {dbms_name.upper()}")
    print("="*80)
    print(f"DBMS: {dbms_name}")
    print(f"Schema: {schema_name}")
    print("="*80)
    print()
    
    # Load plans
    print("📂 Loading plans")
    print("-"*80)
    train_plans = load_plans_from_txt(train_files, dbms_name, max_plans_per_file)
    test_plans = load_plans_from_txt(test_files, dbms_name, max_plans_per_file)
    print(f"✓ Train: {len(train_plans)}, Test: {len(test_plans)}")
    print()
    
    # Generate feature statistics
    print("🔧 Generating feature statistics")
    print("-"*80)
    all_plans = train_plans + test_plans
    feature_statistics = create_feature_statistics_from_plans(all_plans, UnifiedDACEFeaturization)
    print(f"✓ Generated {len(feature_statistics)} features")
    print()
    
    # Split train/val
    print("📦 Creating dataloaders")
    print("-"*80)
    train_size = int(0.85 * len(train_plans))
    val_size = len(train_plans) - train_size
    train_split, val_split = random_split(
        train_plans, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create unified collator
    collate_fn = functools.partial(
        unified_dace_collator,
        dbms_name=dbms_name,
        feature_statistics=feature_statistics,
        config=DACEModelConfig(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            node_length=node_length
        )
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
    
    print(f"✓ Train: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}")
    print()
    
    # Create model
    print("🤖 Creating DACE model")
    print("-"*80)
    
    model_config = DACEModelConfig(
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        node_length=node_length,
        featurization=UnifiedDACEFeaturization,
        device=device
    )
    
    model = DACELora(config=model_config)
    model = model.to(device)
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print()
    
    # Training
    print("🚀 Training")
    print("-"*80)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            seq_encodings, attention_masks, loss_masks, run_times, labels, _ = batch
            
            seq_encodings = seq_encodings.to(device)
            run_times = run_times.to(device)
            labels = labels.to(device)
            
            # Masks may be None
            if attention_masks is not None:
                attention_masks = attention_masks.to(device)
            if loss_masks is not None:
                loss_masks = loss_masks.to(device)
            
            optimizer.zero_grad()
            predictions = model((seq_encodings, attention_masks, loss_masks, run_times))
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
            for batch in val_loader:
                seq_encodings, attention_masks, loss_masks, run_times, labels, _ = batch
                seq_encodings = seq_encodings.to(device)
                run_times = run_times.to(device)
                labels = labels.to(device)
                
                if attention_masks is not None:
                    attention_masks = attention_masks.to(device)
                if loss_masks is not None:
                    loss_masks = loss_masks.to(device)
                
                predictions = model((seq_encodings, attention_masks, loss_masks, run_times))
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        # Calculate metrics
        if all_preds:
            all_preds = np.concatenate(all_preds).flatten()
            all_labels = np.concatenate(all_labels).flatten()
            safe_preds = np.clip(all_preds, 0.1, np.inf)
            safe_labels = np.clip(all_labels, 0.1, np.inf)
            median_q_error = QError(percentile=50, min_val=0.1).evaluate_metric(safe_labels, safe_preds)
            rmse = RMSE().evaluate_metric(all_labels, all_preds)
        else:
            median_q_error, rmse = None, None
        
        metrics_str = f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
        if median_q_error:
            metrics_str += f", q-err={median_q_error:.4f}"
        if rmse:
            metrics_str += f", rmse={rmse:.4f}"
        
        print(f"Epoch {epoch+1}/{epochs}: {metrics_str}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / f"{dbms_name}_dace_best.pt")
            print(f"  ✓ Saved best model")
    
    print(f"\n✅ Training complete! Best val_loss: {best_val_loss:.4f}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Unified DACE Training')
    
    parser.add_argument('--train_files', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--schema', type=str, required=True)
    parser.add_argument('--dbms', type=str, default='trino', choices=['trino', 'postgres', 'mysql'])
    parser.add_argument('--output_dir', type=str, default='models/unified_dace')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--node_length', type=int, default=22)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_plans', type=int, default=None)
    
    args = parser.parse_args()
    
    train_files = [Path(p.strip()) for p in args.train_files.split(',')]
    test_files = [Path(args.test_file)]
    
    return run_training(
        dbms_name=args.dbms,
        train_files=train_files,
        test_files=test_files,
        schema_name=args.schema,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        node_length=args.node_length,
        learning_rate=args.lr,
        device=args.device,
        max_plans_per_file=args.max_plans
    )


if __name__ == "__main__":
    sys.exit(main())

