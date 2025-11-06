"""
Unified MSCN batching for all DBMS

MSCN uses set-based representation:
- Table set
- Predicate set  
- Aggregate set

This unified version works for PostgreSQL, Trino, MySQL, etc.
"""

import numpy as np
import torch
from typing import List, Any, Dict

from core.features.mapper import FeatureMapper
from core.statistics.schema import StandardizedStatistics


def extract_sets_unified(
    plan,
    dbms_name: str,
    feature_statistics: dict,
    db_statistics: StandardizedStatistics = None
) -> tuple:
    """
    Extract table/predicate/aggregate sets from plan (unified).
    
    Args:
        plan: Plan operator tree
        dbms_name: DBMS name
        feature_statistics: Feature statistics
        db_statistics: Standardized database statistics
    
    Returns:
        tuple: (table_features, predicate_features, aggregate_features)
    """
    feature_mapper = FeatureMapper(dbms_name)
    
    # Collect sets
    tables = set()
    predicates = []
    aggregates = []
    
    def traverse(node):
        # Extract table
        table = feature_mapper.get_feature('table', node.plan_parameters) if hasattr(node, 'plan_parameters') else None
        if table:
            tables.add(table)
        
        # Extract predicates
        if hasattr(node, 'plan_parameters'):
            params = node.plan_parameters
            filter_col = params.get('filter_columns') if isinstance(params, dict) else getattr(params, 'filter_columns', None)
            
            if filter_col:
                # Extract predicates recursively
                def extract_predicates(fc):
                    operator = fc.get('operator') if isinstance(fc, dict) else getattr(fc, 'operator', None)
                    column = fc.get('column') if isinstance(fc, dict) else getattr(fc, 'column', None)
                    
                    if operator and str(operator) not in ['AND', 'OR']:
                        # Comparison predicate
                        predicates.append((column, operator))
                    
                    children = fc.get('children', []) if isinstance(fc, dict) else getattr(fc, 'children', [])
                    for child in children:
                        extract_predicates(child)
                
                extract_predicates(filter_col)
            
            # Extract aggregates
            output_cols = params.get('output_columns') if isinstance(params, dict) else getattr(params, 'output_columns', None)
            if output_cols:
                for oc in output_cols:
                    agg = oc.get('aggregation') if isinstance(oc, dict) else getattr(oc, 'aggregation', None)
                    if agg and agg != 'None':
                        aggregates.append(agg)
        
        for child in node.children:
            traverse(child)
    
    traverse(plan)
    
    # Encode sets as vectors
    # Table features (one-hot over known tables)
    table_vec = encode_set_as_vector(tables, feature_statistics.get('table_names', {}))
    
    # Predicate features (count per operator type)
    pred_vec = encode_predicates_as_vector(predicates, feature_statistics.get('operators', {}))
    
    # Aggregate features (one-hot)
    agg_vec = encode_set_as_vector(aggregates, feature_statistics.get('aggregation', {}))
    
    return table_vec, pred_vec, agg_vec


def encode_set_as_vector(items: set, value_dict: dict) -> np.ndarray:
    """Encode set as binary vector."""
    vec = np.zeros(max(len(value_dict), 10))
    for item in items:
        idx = value_dict.get('value_dict', {}).get(str(item), -1)
        if 0 <= idx < len(vec):
            vec[idx] = 1.0
    return vec


def encode_predicates_as_vector(predicates: list, operator_dict: dict) -> np.ndarray:
    """Encode predicates as count vector."""
    vec = np.zeros(max(len(operator_dict), 10))
    for column, operator in predicates:
        idx = operator_dict.get('value_dict', {}).get(str(operator), -1)
        if 0 <= idx < len(vec):
            vec[idx] += 1.0
    return vec


def unified_mscn_plan_collator(
    plans: List,
    dbms_name: str,
    feature_statistics: dict = None,
    db_statistics: dict = None,
    **kwargs
):
    """
    Unified MSCN collator for all DBMS.
    
    Args:
        plans: List of (sample_idx, plan) tuples
        dbms_name: DBMS name
        feature_statistics: Feature statistics
        db_statistics: Database statistics (StandardizedStatistics format)
    
    Returns:
        tuple: Features and labels for MSCN
    """
    all_table_feats = []
    all_pred_feats = []
    all_agg_feats = []
    labels = []
    sample_idxs = []
    
    # Get standardized stats
    db_stats = db_statistics.get(0) if db_statistics else None
    
    for sample_idx, plan in plans:
        sample_idxs.append(sample_idx)
        
        # Extract sets
        table_vec, pred_vec, agg_vec = extract_sets_unified(
            plan,
            dbms_name,
            feature_statistics,
            db_stats
        )
        
        all_table_feats.append(table_vec)
        all_pred_feats.append(pred_vec)
        all_agg_feats.append(agg_vec)
        labels.append(plan.plan_runtime / 1000)  # Convert to sec
    
    # Stack to tensors
    table_feats = torch.tensor(np.array(all_table_feats), dtype=torch.float32)
    pred_feats = torch.tensor(np.array(all_pred_feats), dtype=torch.float32)
    agg_feats = torch.tensor(np.array(all_agg_feats), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return table_feats, pred_feats, agg_feats, labels, sample_idxs

