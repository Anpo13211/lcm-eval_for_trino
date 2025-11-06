"""
Unified Flat Model Feature Extraction

This works for all DBMS by using FeatureMapper.
"""

import numpy as np
from typing import List, Any
from core.features.mapper import FeatureMapper
from core.statistics.schema import StandardizedStatistics


def extract_flat_features_unified(
    plan,
    dbms_name: str,
    feature_statistics: dict,
    db_statistics: StandardizedStatistics = None
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
    
    # Aggregate features from entire plan tree
    all_features = []
    
    # Plan-level features
    plan_features = [
        'operator_type',
        'estimated_cardinality',
        'estimated_cost',
        'estimated_width',
        'workers_planned'
    ]
    
    # Count operators
    operator_counts = {}
    total_est_card = 0.0
    total_est_cost = 0.0
    max_est_card = 0.0
    
    def traverse(node):
        nonlocal total_est_card, total_est_cost, max_est_card
        
        # Extract features
        op_type = feature_mapper.get_feature('operator_type', node.plan_parameters)
        est_card = feature_mapper.get_feature('estimated_cardinality', node.plan_parameters)
        est_cost = feature_mapper.get_feature('estimated_cost', node.plan_parameters)
        
        # Count operators
        operator_counts[op_type] = operator_counts.get(op_type, 0) + 1
        
        # Aggregate numeric features
        if est_card:
            total_est_card += est_card
            max_est_card = max(max_est_card, est_card)
        
        if est_cost:
            total_est_cost += est_cost
        
        # Traverse children
        for child in node.children:
            traverse(child)
    
    traverse(plan)
    
    # Build feature vector
    features = [
        len(operator_counts),  # Number of operator types
        total_est_card,
        total_est_cost,
        max_est_card,
    ]
    
    # Add operator type counts (one-hot style)
    common_operators = [
        'Aggregate', 'TableScan', 'ScanFilterProject', 'Join', 'HashJoin',
        'NestedLoopJoin', 'Sort', 'Limit', 'Exchange', 'RemoteSource',
        'SeqScan', 'IndexScan', 'MergeJoin'
    ]
    
    for op in common_operators:
        features.append(operator_counts.get(op, 0))
    
    # Add database statistics if available
    if db_statistics:
        # Total table count
        features.append(len(db_statistics.table_stats))
        # Total column count
        features.append(len(db_statistics.column_stats))
        # Average table size
        if db_statistics.table_stats:
            avg_size = np.mean([t.row_count for t in db_statistics.table_stats.values()])
            features.append(avg_size)
        else:
            features.append(0.0)
    else:
        features.extend([0.0, 0.0, 0.0])
    
    return np.array(features, dtype=np.float32)


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
    features_list = []
    
    for plan in plans:
        features = extract_flat_features_unified(
            plan,
            dbms_name,
            feature_statistics,
            db_statistics
        )
        features_list.append(features)
    
    return np.array(features_list)

