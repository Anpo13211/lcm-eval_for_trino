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
import collections
import numpy as np
import torch
import dgl

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
    
    # Create heterograph (DGL format)
    graph, features = _create_heterograph(
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
    
    # Process labels
    labels = postprocess_labels(labels)
    
    return graph, features, labels, sample_idxs


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
    from training.preprocessing.feature_statistics import FeatureType
    
    if feature_statistics is None:
        return
    
    # Exactly the same as original implementation
    for k, v in feature_statistics.items():
        if v.get('type') == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = v['center']
            scaler.scale_ = v['scale']
            feature_statistics[k]['scaler'] = scaler


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


def _create_heterograph(
    plan_depths, plan_features, plan_to_plan_edges,
    filter_to_plan_edges, predicate_col_features,
    output_column_to_plan_edges, output_column_features,
    column_to_output_column_edges, column_features,
    table_features, table_to_plan_edges,
    predicate_depths, intra_predicate_edges, logical_preds
):
    """
    Create DGL heterograph from graph data.
    
    This is the unified version that works for all DBMS.
    Based on postgres_plan_collator logic.
    """
    # Create node types per depth for plan nodes
    data_dict, nodes_per_depth, plan_dict = create_node_types_per_depth(plan_depths, plan_to_plan_edges)
    
    # Create node types for predicates
    pred_dict = dict()
    nodes_per_pred_depth = collections.defaultdict(int)
    no_filter_columns = 0
    
    for pred_node, d in enumerate(predicate_depths):
        if logical_preds[pred_node]:
            # Logical predicate (AND/OR)
            pred_dict[pred_node] = (nodes_per_pred_depth[d], d)
            nodes_per_pred_depth[d] += 1
        else:
            # Filter column (comparison operator)
            pred_dict[pred_node] = no_filter_columns
            no_filter_columns += 1
    
    # Adapt predicate edges
    adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, 
                         logical_preds, plan_dict, pred_dict, pred_node_type_id)
    
    # Add other edge types
    data_dict[('column', 'col_output_col', 'output_column')] = column_to_output_column_edges
    
    for u, v in output_column_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[('output_column', 'to_plan', f'plan{d_v}')].append((u, v_node_id))
    
    for u, v in table_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[('table', 'to_plan', f'plan{d_v}')].append((u, v_node_id))
    
    # Calculate node counts
    max_depth, max_pred_depth = get_depths(plan_depths, predicate_depths)
    num_nodes_dict = {
        'column': len(column_features),
        'table': len(table_features),
        'output_column': len(output_column_features),
        'filter_column': len(logical_preds) - sum(logical_preds),
    }
    num_nodes_dict = update_node_counts(max_depth, max_pred_depth, nodes_per_depth, 
                                       nodes_per_pred_depth, num_nodes_dict)
    
    # Create heterograph
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    graph.max_depth = max_depth
    graph.max_pred_depth = max_pred_depth
    
    # Organize features by node type
    features = collections.defaultdict(list)
    features.update(dict(
        column=column_features,
        table=table_features,
        output_column=output_column_features,
        filter_column=[f for f, log_pred in zip(predicate_col_features, logical_preds) if not log_pred]
    ))
    
    # Sort plan features by depth
    for u, plan_feat in enumerate(plan_features):
        u_node_id, d_u = plan_dict[u]
        features[f'plan{d_u}'].append(plan_feat)
    
    # Sort predicate features by depth
    for pred_node_id, pred_feat in enumerate(predicate_col_features):
        if not logical_preds[pred_node_id]:
            continue
        node_type, _ = pred_node_type_id(logical_preds, pred_dict, pred_node_id)
        features[node_type].append(pred_feat)
    
    # Postprocess features
    features = postprocess_feats(features, num_nodes_dict)
    
    return graph, features


def create_node_types_per_depth(plan_depths, plan_to_plan_edges):
    """Create node types based on depth."""
    plan_dict = dict()
    nodes_per_depth = collections.defaultdict(int)
    
    for plan_node, d in enumerate(plan_depths):
        plan_dict[plan_node] = (nodes_per_depth[d], d)
        nodes_per_depth[d] += 1
    
    # Create edge dict
    data_dict = collections.defaultdict(list)
    for u, v in plan_to_plan_edges:
        u_node_id, d_u = plan_dict[u]
        v_node_id, d_v = plan_dict[v]
        assert d_v < d_u, f"Plan edges should go from deeper to shallower: {d_u} -> {d_v}"
        data_dict[(f'plan{d_u}', f'intra_plan', f'plan{d_v}')].append((u_node_id, v_node_id))
    
    return data_dict, nodes_per_depth, plan_dict


def adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, 
                          logical_preds, plan_dict, pred_dict, pred_node_type_id_func):
    """Adapt predicate edges to heterograph format."""
    # Filter to plan edges
    for u, v in filter_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)
        data_dict[(node_type, 'to_plan', f'plan{d_v}')].append((u_node_id, v_node_id))
    
    # Intra predicate edges
    for u, v in intra_predicate_edges:
        u_node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)
        v_node_type, v_node_id = pred_node_type_id_func(logical_preds, pred_dict, v)
        data_dict[(u_node_type, 'intra_predicate', v_node_type)].append((u_node_id, v_node_id))


def pred_node_type_id(logical_preds, pred_dict, pred_node):
    """Determine predicate node type and ID."""
    if logical_preds[pred_node]:
        # Logical predicate
        node_id, depth = pred_dict[pred_node]
        return f'logical_pred_{depth}', node_id
    else:
        # Filter column
        return 'filter_column', pred_dict[pred_node]


def update_node_counts(max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth, num_nodes_dict):
    """Update node counts with depth-specific nodes."""
    num_nodes_dict.update({f'plan{d}': nodes_per_depth[d] for d in range(max_depth + 1)})
    num_nodes_dict.update({f'logical_pred_{d}': nodes_per_pred_depth[d] for d in range(max_pred_depth + 1)})
    # Filter out zero nodes
    num_nodes_dict = {k: v for k, v in num_nodes_dict.items() if v > 0}
    return num_nodes_dict


def get_depths(plan_depths, predicate_depths):
    """Get maximum depths for plans and predicates."""
    max_depth = max(plan_depths) if plan_depths else 0
    max_pred_depth = max(predicate_depths) if predicate_depths else 0
    return max_depth, max_pred_depth


def postprocess_labels(labels):
    """Convert labels to tensor (runtime in seconds)."""
    labels = np.array(labels, dtype=np.float32)
    labels /= 1000  # Convert ms to seconds
    labels = torch.from_numpy(labels)  # Convert to tensor
    return labels


def postprocess_feats(features, num_nodes_dict):
    """Convert features to tensors."""
    for k in list(features.keys()):
        v = features[k]
        v = np.array(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0)
        v = torch.from_numpy(v)
        features[k] = v
    
    # Filter out node types with zero nodes
    features = {k: v for k, v in features.items() if k in num_nodes_dict}
    return features

