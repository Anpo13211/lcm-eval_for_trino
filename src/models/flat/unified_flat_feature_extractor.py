"""
Unified Flat Model Feature Extraction

This works for all DBMS by using FeatureMapper.
"""

import numpy as np
from typing import List, Any, Iterable, Sequence
from core.features.mapper import FeatureMapper
from core.statistics.schema import StandardizedStatistics


OTHER_OPERATOR_BUCKET = "__other__"
_BASE_OPERATOR_TYPES = [
    'Aggregate',
    'HashAggregate',
    'TableScan',
    'ScanFilterProject',
    'Filter',
    'Project',
    'Join',
    'HashJoin',
    'MergeJoin',
    'NestedLoopJoin',
    'Sort',
    'Limit',
    'Exchange',
    'RemoteSource',
    'SeqScan',
    'IndexScan',
    'IndexOnlyScan',
    'BitmapHeapScan',
    'BitmapIndexScan',
    'Values',
    'Window',
    'Union',
    'Distinct',
    'TopN',
    'TableFinish',
    'TableWriter',
    'Output',
    'ExplainAnalyze',
    'Materialize',
    'unnamed',
    'unknown',
]
_BASE_OPERATOR_TYPES.append(OTHER_OPERATOR_BUCKET)
DEFAULT_OPERATOR_TYPES = tuple(dict.fromkeys(_BASE_OPERATOR_TYPES))


def extract_flat_features_unified(
    plan,
    dbms_name: str,
    feature_statistics: dict,
    db_statistics: StandardizedStatistics = None,
    operator_vocab: Sequence[str] = None
) -> np.ndarray:
    """
    Extract flat feature vector from plan (unified version).
    
    This replaces DBMS-specific feature extraction with unified logic.
    
    Args:
        plan: Plan operator tree
        dbms_name: DBMS name
        feature_statistics: Feature statistics
        db_statistics: Standardized database statistics
    
    Returns:
        Flat feature vector (numpy array)
    """
    feature_mapper = FeatureMapper(dbms_name)
    resolved_vocab = _resolve_operator_vocab(feature_statistics, operator_vocab)
    operator_index = {op: idx for idx, op in enumerate(resolved_vocab)}
    other_idx = operator_index.get(OTHER_OPERATOR_BUCKET)
    
    feature_vector = np.zeros(len(resolved_vocab), dtype=np.float32)
    
    for node in _walk_plan_nodes(plan):
        plan_params = getattr(node, 'plan_parameters', None)
        op_type = feature_mapper.get_feature('operator_type', plan_params) or OTHER_OPERATOR_BUCKET
        est_card = _to_non_negative_float(
            feature_mapper.get_feature('estimated_cardinality', plan_params)
        )
        
        idx = operator_index.get(op_type, other_idx)
        if idx is not None:
            feature_vector[idx] += est_card
    
    return feature_vector


def extract_batch_features_unified(
    plans: List[Any],
    dbms_name: str,
    feature_statistics: dict,
    db_statistics: StandardizedStatistics = None
) -> np.ndarray:
    """
    Extract flat features for a batch of plans.
    
    Returns:
        Feature matrix (num_plans × num_features)
    """
    resolved_vocab = _resolve_operator_vocab(feature_statistics)
    local_feature_stats = dict(feature_statistics or {})
    local_feature_stats['flat_vector_operator_vocab'] = resolved_vocab
    
    features_list = []
    for plan in plans:
        features = extract_flat_features_unified(
            plan=plan,
            dbms_name=dbms_name,
            feature_statistics=local_feature_stats,
            db_statistics=db_statistics,
            operator_vocab=resolved_vocab
        )
        features_list.append(features)
    
    return np.array(features_list, dtype=np.float32)


def build_operator_vocab(
    plans: Iterable[Any],
    dbms_name: str,
    include_other_bucket: bool = True
) -> List[str]:
    """
    Build operator vocabulary from a collection of plans.
    
    Args:
        plans: Iterable of plan roots.
        dbms_name: DBMS identifier for FeatureMapper.
        include_other_bucket: Whether to append the fallback bucket.
    
    Returns:
        Sorted list of operator names plus optional OTHER bucket.
    """
    mapper = FeatureMapper(dbms_name)
    operator_names = set()
    
    for plan in plans:
        for node in _walk_plan_nodes(plan):
            plan_params = getattr(node, 'plan_parameters', None)
            op_type = mapper.get_feature('operator_type', plan_params)
            if op_type:
                operator_names.add(op_type)
    
    if not operator_names:
        vocab = list(DEFAULT_OPERATOR_TYPES)
    else:
        vocab = sorted(operator_names)
        if include_other_bucket and OTHER_OPERATOR_BUCKET not in vocab:
            vocab.append(OTHER_OPERATOR_BUCKET)
    return vocab


def _resolve_operator_vocab(feature_statistics: dict, operator_vocab: Sequence[str] = None) -> List[str]:
    if operator_vocab is not None:
        resolved = list(dict.fromkeys(operator_vocab))
    else:
        candidate = None
        if feature_statistics:
            if isinstance(feature_statistics, dict):
                candidate = (
                    feature_statistics.get('flat_vector_operator_vocab')
                    or feature_statistics.get('operator_vocab')
                )
                flat_group = feature_statistics.get('flat_vector')
                if isinstance(flat_group, dict):
                    candidate = candidate or flat_group.get('operator_vocab')
        if candidate:
            resolved = list(dict.fromkeys(candidate))
        else:
            resolved = list(DEFAULT_OPERATOR_TYPES)
    if OTHER_OPERATOR_BUCKET not in resolved:
        resolved.append(OTHER_OPERATOR_BUCKET)
    return resolved


def _walk_plan_nodes(plan: Any):
    """Depth-first traversal yielding all nodes in a plan."""
    if plan is None:
        return
    stack = [plan]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        yield node
        children = getattr(node, 'children', None)
        if children:
            # Preserve original order as much as possible
            stack.extend(children)


def _to_non_negative_float(value: Any) -> float:
    """Safely convert a value to non-negative float."""
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if result > 0 else 0.0

