"""
Unified plan collator

This single collator function replaces:
- postgres_plan_collator
- trino_plan_collator
- (future DBMS-specific collators)

It uses UnifiedPlanConverter to convert plans to graphs in a DBMS-agnostic way.
"""

from typing import List, Dict, Any, Optional
from types import SimpleNamespace
import numpy as np
import torch

from core.graph.converter import UnifiedPlanConverter
from core.statistics.schema import StandardizedStatistics
from core.statistics.converter import StatisticsConverter, PostgreSQLStatisticsConverter, TrinoStatisticsConverter
from training.featurizations import Featurization


def unified_plan_collator(
    plans: List[Any],
    feature_statistics: Dict = None,
    db_statistics: Dict = None,
    plan_featurization: Featurization = None,
    dbms_name: str = None
) -> Dict[str, Any]:
    """
    Unified plan collator that works for all DBMS.
    
    This replaces the need for separate collators per DBMS:
    - postgres_plan_collator → unified_plan_collator(dbms_name="postgres")
    - trino_plan_collator → unified_plan_collator(dbms_name="trino")
    
    Args:
        plans: List of (sample_idx, plan) tuples
        feature_statistics: Global feature statistics for encoding
        db_statistics: Database statistics (can be DBMS-specific or StandardizedStatistics)
        plan_featurization: Feature configuration
        dbms_name: DBMS name (if None, inferred from plans)
    
    Returns:
        Dictionary with:
        - labels: Tensor of plan runtimes
        - graph_data: Dict with all graph components
        - sample_idxs: List of sample indices
    
    Example:
        # PostgreSQL
        collator = functools.partial(
            unified_plan_collator,
            dbms_name="postgres",
            plan_featurization=PostgresTrueCardDetail,
            feature_statistics=feature_stats
        )
        
        # Trino
        collator = functools.partial(
            unified_plan_collator,
            dbms_name="trino",
            plan_featurization=PostgresTrueCardDetail,  # Same featurization!
            feature_statistics=feature_stats
        )
    """
    # Infer DBMS name if not provided
    if dbms_name is None:
        dbms_name = _infer_dbms_name(plans)
    
    # Convert db_statistics to StandardizedStatistics if needed
    standardized_stats = _ensure_standardized_statistics(db_statistics, dbms_name)
    
    # Initialize accumulators for all graph components
    all_plan_depths = []
    all_plan_features = []
    all_plan_to_plan_edges = []
    all_filter_to_plan_edges = []
    all_predicate_col_features = []
    all_output_column_to_plan_edges = []
    all_output_column_features = []
    all_column_to_output_column_edges = []
    all_column_features = []
    all_table_features = []
    all_table_to_plan_edges = []
    all_predicate_depths = []
    all_intra_predicate_edges = []
    all_logical_preds = []
    
    labels = []
    sample_idxs = []
    
    # Prepare robust encoder for numerical fields (from existing code)
    _add_numerical_scalers(feature_statistics)
    
    # Track global indices across all plans
    plan_offset = 0
    filter_offset = 0
    output_column_offset = 0
    column_offset = 0
    table_offset = 0
    
    # Create converter
    converter = UnifiedPlanConverter(
        dbms_name=dbms_name,
        plan_featurization=plan_featurization,
        feature_statistics=feature_statistics,
        db_statistics=standardized_stats
    )
    
    # Process each plan
    for sample_idx, plan in plans:
        sample_idxs.append(sample_idx)
        
        # Extract runtime label
        if hasattr(plan, 'plan_runtime'):
            runtime = plan.plan_runtime
        elif hasattr(plan, 'analyze_plans'):
            runtime = getattr(plan.analyze_plans[0], 'plan_runtime', 0.0)
        else:
            runtime = 0.0
        
        labels.append(runtime)
        
        # Get database ID
        database_id = getattr(plan, 'database_id', 0)
        
        # Convert plan to graph
        graph_data = converter.convert(plan, database_id)
        
        # Adjust indices and append to global lists
        _append_with_offset(
            all_plan_depths,
            all_plan_features,
            all_plan_to_plan_edges,
            all_filter_to_plan_edges,
            all_predicate_col_features,
            all_output_column_to_plan_edges,
            all_output_column_features,
            all_column_to_output_column_edges,
            all_column_features,
            all_table_features,
            all_table_to_plan_edges,
            all_predicate_depths,
            all_intra_predicate_edges,
            all_logical_preds,
            graph_data,
            plan_offset,
            filter_offset,
            output_column_offset,
            column_offset,
            table_offset
        )
        
        # Update offsets
        plan_offset += len(graph_data['plan_depths'])
        filter_offset += len(graph_data['predicate_col_features'])
        output_column_offset += len(graph_data['output_column_features'])
        column_offset += len(graph_data['column_features'])
        table_offset += len(graph_data['table_features'])
    
    # Convert to tensors
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # Tensorize graph data
    graph_tensors = _tensorize_graph_data(
        all_plan_depths,
        all_plan_features,
        all_plan_to_plan_edges,
        all_filter_to_plan_edges,
        all_predicate_col_features,
        all_output_column_to_plan_edges,
        all_output_column_features,
        all_column_to_output_column_edges,
        all_column_features,
        all_table_features,
        all_table_to_plan_edges,
        all_predicate_depths,
        all_intra_predicate_edges,
        all_logical_preds
    )
    
    return {
        'labels': labels,
        **graph_tensors,
        'sample_idxs': sample_idxs
    }


def _infer_dbms_name(plans: List[Any]) -> str:
    """
    Infer DBMS name from plan structure.
    
    Args:
        plans: List of plans
    
    Returns:
        DBMS name (e.g., "postgres", "trino")
    """
    if not plans:
        return "postgres"  # Default
    
    # Get first plan
    _, first_plan = plans[0] if isinstance(plans[0], tuple) else (0, plans[0])
    
    # Check for DBMS-specific attributes
    if hasattr(first_plan, 'plan_parameters'):
        params = first_plan.plan_parameters
        
        # Check for PostgreSQL-specific attributes
        if hasattr(params, 'Node Type') or (isinstance(params, dict) and 'Node Type' in params):
            return "postgres"
        
        # Check for Trino-specific attributes
        if hasattr(params, 'op_name') or (isinstance(params, dict) and 'op_name' in params):
            return "trino"
    
    return "postgres"  # Default


def _ensure_standardized_statistics(
    db_statistics: Any,
    dbms_name: str
) -> StandardizedStatistics:
    """
    Ensure db_statistics is in StandardizedStatistics format.
    
    If it's already standardized, return as-is.
    If it's DBMS-specific format, convert it.
    
    Args:
        db_statistics: Database statistics (various formats)
        dbms_name: DBMS name
    
    Returns:
        StandardizedStatistics instance
    """
    # If already StandardizedStatistics, return as-is
    if isinstance(db_statistics, StandardizedStatistics):
        return db_statistics
    
    # If it's a dict of StandardizedStatistics, extract the right one
    if isinstance(db_statistics, dict):
        # Check if values are StandardizedStatistics
        for key, value in db_statistics.items():
            if isinstance(value, StandardizedStatistics):
                return value
        
        # Otherwise, assume it's DBMS-specific format and convert
        if dbms_name == "postgres":
            converter = PostgreSQLStatisticsConverter()
        elif dbms_name == "trino":
            converter = TrinoStatisticsConverter()
        else:
            # Unknown DBMS, return empty statistics
            return StandardizedStatistics()
        
        # If db_statistics is a dict with a single key, extract that
        if len(db_statistics) == 1:
            raw_stats = list(db_statistics.values())[0]
        else:
            raw_stats = db_statistics
        
        return converter.convert(raw_stats)
    
    # If it's a SimpleNamespace or object, try to convert
    if hasattr(db_statistics, '__dict__'):
        if dbms_name == "postgres":
            converter = PostgreSQLStatisticsConverter()
        elif dbms_name == "trino":
            converter = TrinoStatisticsConverter()
        else:
            return StandardizedStatistics()
        
        return converter.convert(db_statistics)
    
    # Fallback: return empty statistics
    return StandardizedStatistics()


def _add_numerical_scalers(feature_statistics: dict):
    """
    Add numerical scalers to feature statistics (from existing code).
    
    This is copied from postgres_plan_batching.py for compatibility.
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import Pipeline
    
    if feature_statistics is None:
        return
    
    # Add scalers for numerical features if not already present
    numerical_features = [
        'est_card', 'act_card', 'est_width', 'est_cost',
        'reltuples', 'relpages', 'avg_width', 'n_distinct', 'null_frac'
    ]
    
    for feat_name in numerical_features:
        if feat_name in feature_statistics:
            if not isinstance(feature_statistics[feat_name], Pipeline):
                # Create a simple pass-through for now
                # In production, this would fit on training data
                pass


def _append_with_offset(
    all_plan_depths, all_plan_features, all_plan_to_plan_edges,
    all_filter_to_plan_edges, all_predicate_col_features,
    all_output_column_to_plan_edges, all_output_column_features,
    all_column_to_output_column_edges, all_column_features,
    all_table_features, all_table_to_plan_edges,
    all_predicate_depths, all_intra_predicate_edges, all_logical_preds,
    graph_data, plan_offset, filter_offset, output_column_offset,
    column_offset, table_offset
):
    """
    Append graph data with proper index offsets.
    """
    # Plan nodes
    all_plan_depths.extend(graph_data['plan_depths'])
    all_plan_features.extend(graph_data['plan_features'])
    
    # Plan edges (adjust indices)
    for src, dst in graph_data['plan_to_plan_edges']:
        all_plan_to_plan_edges.append((src + plan_offset, dst + plan_offset))
    
    # Filter edges
    for src, dst in graph_data['filter_to_plan_edges']:
        all_filter_to_plan_edges.append((src + filter_offset, dst + plan_offset))
    
    # Predicate features
    all_predicate_col_features.extend(graph_data['predicate_col_features'])
    all_predicate_depths.extend(graph_data['predicate_depths'])
    all_logical_preds.extend(graph_data['logical_preds'])
    
    # Predicate edges
    for src, dst in graph_data['intra_predicate_edges']:
        all_intra_predicate_edges.append((src + filter_offset, dst + filter_offset))
    
    # Output column edges
    for src, dst in graph_data['output_column_to_plan_edges']:
        all_output_column_to_plan_edges.append((src + output_column_offset, dst + plan_offset))
    
    # Output column features
    all_output_column_features.extend(graph_data['output_column_features'])
    
    # Column edges
    for src, dst in graph_data['column_to_output_column_edges']:
        all_column_to_output_column_edges.append((src + column_offset, dst + output_column_offset))
    
    # Column features
    all_column_features.extend(graph_data['column_features'])
    
    # Table edges
    for src, dst in graph_data['table_to_plan_edges']:
        all_table_to_plan_edges.append((src + table_offset, dst + plan_offset))
    
    # Table features
    all_table_features.extend(graph_data['table_features'])


def _tensorize_graph_data(
    plan_depths, plan_features, plan_to_plan_edges,
    filter_to_plan_edges, predicate_col_features,
    output_column_to_plan_edges, output_column_features,
    column_to_output_column_edges, column_features,
    table_features, table_to_plan_edges,
    predicate_depths, intra_predicate_edges, logical_preds
) -> Dict[str, torch.Tensor]:
    """
    Convert graph data to PyTorch tensors.
    """
    result = {}
    
    # Plan depths
    if plan_depths:
        result['plan_depths'] = torch.tensor(plan_depths, dtype=torch.long)
    
    # Plan features
    if plan_features:
        result['plan_features'] = torch.tensor(np.array(plan_features), dtype=torch.float32)
    
    # Edges
    if plan_to_plan_edges:
        result['plan_to_plan_edges'] = torch.tensor(plan_to_plan_edges, dtype=torch.long).T
    
    if filter_to_plan_edges:
        result['filter_to_plan_edges'] = torch.tensor(filter_to_plan_edges, dtype=torch.long).T
    
    if output_column_to_plan_edges:
        result['output_column_to_plan_edges'] = torch.tensor(output_column_to_plan_edges, dtype=torch.long).T
    
    if column_to_output_column_edges:
        result['column_to_output_column_edges'] = torch.tensor(column_to_output_column_edges, dtype=torch.long).T
    
    if table_to_plan_edges:
        result['table_to_plan_edges'] = torch.tensor(table_to_plan_edges, dtype=torch.long).T
    
    if intra_predicate_edges:
        result['intra_predicate_edges'] = torch.tensor(intra_predicate_edges, dtype=torch.long).T
    
    # Features
    if predicate_col_features:
        result['predicate_col_features'] = torch.tensor(np.array(predicate_col_features), dtype=torch.float32)
    
    if output_column_features:
        result['output_column_features'] = torch.tensor(np.array(output_column_features), dtype=torch.float32)
    
    if column_features:
        result['column_features'] = torch.tensor(np.array(column_features), dtype=torch.float32)
    
    if table_features:
        result['table_features'] = torch.tensor(np.array(table_features), dtype=torch.float32)
    
    # Other
    if predicate_depths:
        result['predicate_depths'] = torch.tensor(predicate_depths, dtype=torch.long)
    
    if logical_preds:
        result['logical_preds'] = torch.tensor(logical_preds, dtype=torch.bool)
    
    return result

