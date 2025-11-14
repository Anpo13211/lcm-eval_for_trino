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
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import numpy as np

# Initialize plugin system
import core.init_plugins

from core.plugins.registry import DBMSRegistry
from core.statistics.schema import StandardizedStatistics
from models.query_former.dataloader import query_former_plan_collator
from models.query_former.model import QueryFormer
from classes.classes import QueryFormerModelConfig
from models.workload_driven.dataset.dataset_creation import PlanModelInputDims
from training.training.metrics import QError, RMSE
from types import SimpleNamespace


class UnifiedPlanDataset(Dataset):
    def __init__(self, plans):
        self.plans = plans
    
    def __len__(self):
        return len(self.plans)
    
    def __getitem__(self, idx):
        return idx, self.plans[idx]


def load_table_samples(schema_name: str, no_samples: int = 1000):
    """Load table samples from CSV files for sample_vec generation."""
    import pandas as pd
    from cross_db_benchmark.benchmark_tools.utils import load_schema_json
    
    # Try zero-shot-data location
    data_dirs = [
        f'/Users/an/query_engine/lakehouse/zero-shot_datasets/{schema_name}',
        f'../zero-shot-data/datasets/{schema_name}',
        f'data/{schema_name}'
    ]
    
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
                except Exception as e:
                    print(f"  ⚠️  Failed to read {csv_path}: {e}")
        
        if df is not None:
            table_samples[table_name] = df
        else:
            print(f"  ⚠️  No data found for table {table_name}")
    
    return table_samples if table_samples else None


def load_plans_from_txt(file_paths: list, dbms_name: str, schema_name: str, max_plans: int = None, 
                        table_samples=None, col_stats=None):
    """Load plans from txt files with sample_vec generation."""
    parser = DBMSRegistry.get_parser(dbms_name)
    
    all_plans = []
    for file_path in file_paths:
        print(f"  Loading {Path(file_path).name}...")
        parsed_plans, runtimes = parser.parse_explain_analyze_file(
            str(file_path), min_runtime=0, max_runtime=float('inf')
        )
        
        # Augment sample_vec if table_samples available
        if table_samples and col_stats and dbms_name == "trino":
            from models.workload_driven.preprocessing.sample_vectors_trino import augment_sample
            for plan in parsed_plans:
                try:
                    augment_sample(table_samples, col_stats, plan)
                except Exception as e:
                    pass  # Continue even if sample_vec generation fails
        
        for plan, runtime in zip(parsed_plans, runtimes):
            plan.plan_runtime = runtime
            plan.database_id = 0
        
        if max_plans and len(parsed_plans) > max_plans:
            parsed_plans = parsed_plans[:max_plans]
        
        all_plans.extend(parsed_plans)
    
    return all_plans


def build_feature_statistics(plans, db_stats, column_stats, dbms_name: str):
    """
    Build feature_statistics by combining (1) plan-derived vocabularies and numeric ranges,
    and (2) database/column statistics for max ranges.
    """
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

        # filter operators (traverse tree)
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

        # aggregations (output_columns)
        outs = params.get('output_columns') if isinstance(params, dict) else getattr(params, 'output_columns', None)
        if outs:
            for oc in outs:
                agg = oc.get('aggregation') if isinstance(oc, dict) else getattr(oc, 'aggregation', None)
                if agg is not None:
                    agg_ops.add(str(agg))

        # join conds if present at root level
        if hasattr(node, 'join_conds') and getattr(node, 'join_conds'):
            for jc in getattr(node, 'join_conds'):
                join_conds.add(str(jc))

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
    # columns.max: use number of distinct columns in column_stats
    num_columns = len(column_stats) if isinstance(column_stats, dict) else 1024
    feature_statistics['columns'] = dict(type='numeric', max=num_columns)
    feature_statistics['column'] = dict(type='numeric', max=num_columns)

    # table.max from db_stats
    num_tables = len(table_names) if table_names else 64
    feature_statistics['table'] = dict(type='numeric', max=num_tables)

    # Alias
    feature_statistics['operator_type'] = feature_statistics['op_name']

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
    
    # Load table samples for sample_vec generation
    print("📂 Loading table samples")
    table_samples = load_table_samples(schema_name, no_samples=1000)
    if table_samples:
        print(f"✓ Loaded {len(table_samples)} table samples")
    else:
        print("⚠️  No table samples found, sample_vec will be default values")
    print()
    
    # Load plans
    print("📂 Loading plans")
    # Note: col_stats will be loaded later, so first pass without sample_vec
    train_plans = load_plans_from_txt(train_files, dbms_name, schema_name, max_plans)
    test_plans = load_plans_from_txt(test_files, dbms_name, schema_name, max_plans)
    print(f"✓ Train: {len(train_plans)}, Test: {len(test_plans)}")
    print()
    
    all_plans = train_plans + test_plans
    
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
    
    # Load column statistics and augment sample_vec
    print("📊 Loading column statistics and augmenting sample_vec")
    try:
        # Use absolute path from project root
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        col_stats_path = project_root / 'datasets_statistics' / f"iceberg_{schema_name}" / 'column_stats.json'
        from cross_db_benchmark.benchmark_tools.utils import load_json
        column_statistics = load_json(str(col_stats_path), namespace=False)
        
        # Convert column_statistics to list format for augment_sample
        col_stats_list = []
        for key, val in column_statistics.items():
            # Key can be tuple or string "(table, column)"
            if isinstance(key, tuple) and len(key) == 2:
                table_name, col_name = key
            elif isinstance(key, str) and key.startswith('('):
                # Parse string like "('table', 'column')"
                import ast
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
        
        # Augment sample_vec for all plans
        if table_samples and col_stats_list:
            from models.workload_driven.preprocessing.sample_vectors_trino import augment_sample
            print(f"  Generating sample_vec for {len(all_plans)} plans...")
            for plan in all_plans:
                try:
                    augment_sample(table_samples, col_stats_list, plan)
                except Exception:
                    pass  # Continue even if fails
            print(f"✓ Sample vectors generated")
        else:
            print(f"  ⚠️  Skipping sample_vec generation (missing data)")
            column_statistics = {}
    except Exception as e:
        print(f"⚠️  Column statistics loading failed: {e}")
        column_statistics = {}
    
    # Build feature statistics from plans and DB stats
    print("📊 Building feature statistics from plans and DB stats")
    feature_statistics = build_feature_statistics(all_plans, db_stats.get(0), column_statistics, dbms_name)
    print(f"✓ Built {len(feature_statistics)} feature types")
    print()
    
    # Split train/val
    print("📦 Creating dataloaders")
    train_size = int(0.85 * len(train_plans))
    val_size = len(train_plans) - train_size
    train_split, val_split = random_split(
        train_plans, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create model
    print("🤖 Creating QueryFormer model")
    
    config = QueryFormerModelConfig(
        hidden_dim_plan=hidden_dim,
        device=device,
        max_num_filters=12  # Increase to handle complex queries (default: 6)
    )
    
    # column_statistics already loaded above, just rebuild feature_statistics with it
    feature_statistics = build_feature_statistics(all_plans, db_stats.get(0), column_statistics, dbms_name)

    # Create collator (use original query_former_plan_collator)
    collate_fn = functools.partial(
        query_former_plan_collator,
        feature_statistics=feature_statistics,
        db_statistics=db_stats,
        column_statistics=column_statistics,
        word_embeddings=None,  # Not needed for basic version
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
    
    print(f"✓ Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    print()
    
    input_dims = PlanModelInputDims(
        feature_statistics=feature_statistics,
        dim_word_embedding=64,
        dim_word_hash=1000,
        dim_bitmaps=1000
    )
    
    model = QueryFormer(config, input_dims, feature_statistics, label_norm=None)
    model = model.to(device)
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print()
    
    # Training
    print("🚀 Training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Use model's internal loss function (QLoss by default, matching PostgreSQL version)
    criterion = model.loss_fxn
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        
        for batch, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            labels = labels.to(device)
            
            predictions = model(batch).squeeze(-1)  # [batch, 1] -> [batch]
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
                # Squeeze before loss and metrics
                predictions_squeezed = predictions.squeeze(-1)  # [batch, 1] -> [batch]
                loss = criterion(predictions_squeezed, labels)
                val_loss += loss.item()
                
                all_preds.append(predictions_squeezed.cpu().numpy())
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
    # Get available DBMS from registry
    available_dbms = DBMSRegistry.get_cli_choices() if DBMSRegistry.list_plugins() else ['trino', 'postgres', 'mysql']
    parser.add_argument('--dbms', type=str, default='trino', choices=available_dbms)
    parser.add_argument('--output_dir', type=str, default='models/unified_queryformer')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_plans', type=int, default=10000)
    
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

