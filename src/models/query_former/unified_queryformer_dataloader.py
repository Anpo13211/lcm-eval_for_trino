"""
Unified QueryFormer dataloader for all DBMS

This replaces PostgreSQL-specific QueryFormer dataloader with a unified version
that works for PostgreSQL, Trino, MySQL, etc.
"""

from types import SimpleNamespace
from typing import List, Tuple
import numpy as np
import torch

from core.features.mapper import FeatureMapper
from core.statistics.schema import StandardizedStatistics
from models.query_former.utils import TreeNode, floyd_warshall_transform
from models.query_former.dataloader import (
    topological_sort,
    calculate_node_heights,
    get_sample_vector,
    parse_filter_information
)


def recursively_convert_plan_unified(
    plan,
    index: int,
    dbms_name: str,
    feature_statistics: dict,
    db_statistics: StandardizedStatistics,
    dim_word_embedding: int = 64,
    dim_word_hash: int = 1000,
    word_embeddings = None,
    dim_bitmaps: int = 1000,
    max_filter_number: int = 5,
    histogram_bin_size: int = 10
) -> TreeNode:
    """
    Recursively convert plan to TreeNode (unified version).
    
    This uses FeatureMapper to extract features in a DBMS-agnostic way.
    
    Args:
        plan: Plan operator
        index: Query index
        dbms_name: DBMS name
        feature_statistics: Feature statistics
        db_statistics: Standardized database statistics
        ... (other parameters same as original)
    
    Returns:
        TreeNode with encoded features
    """
    # Create feature mapper
    feature_mapper = FeatureMapper(dbms_name)
    
    plan_parameters = plan.plan_parameters
    
    # 1. Get operator type using FeatureMapper
    operator_name = feature_mapper.get_feature('operator_type', plan_parameters)
    operator_type_id = feature_statistics.get('op_name', {}).get('value_dict', {}).get(str(operator_name), 0)
    
    # 2. Get table name
    table = feature_mapper.get_feature('table', plan_parameters) if hasattr(feature_mapper, 'get_feature') else None
    
    # Convert table to ID
    table_name = 0  # Default
    if table:
        # Look up table in statistics
        if db_statistics and table in db_statistics.table_stats:
            # Use index as table_name
            table_names = list(db_statistics.table_stats.keys())
            if table in table_names:
                table_name = table_names.index(table)
    
    # 3. Get cardinality estimates using FeatureMapper
    estimated_cardinality = feature_mapper.get_feature('estimated_cardinality', plan_parameters) or 1.0
    
    # 4. Extract filter information
    filter_info = []
    encoded_filter_info = np.zeros(max_filter_number * 3)
    encoded_histogram_info = np.zeros(max_filter_number * histogram_bin_size)
    sample_bitmap_vec = np.zeros(dim_bitmaps)
    
    filter_columns = None
    if isinstance(plan_parameters, dict):
        filter_columns = plan_parameters.get('filter_columns')
    else:
        filter_columns = getattr(plan_parameters, 'filter_columns', None)
    
    if filter_columns is not None:
        # Convert dict to SimpleNamespace if needed
        if isinstance(filter_columns, dict):
            from types import SimpleNamespace
            def dict_to_namespace(d):
                if isinstance(d, dict):
                    ns = SimpleNamespace()
                    for k, v in d.items():
                        if isinstance(v, dict):
                            setattr(ns, k, dict_to_namespace(v))
                        elif isinstance(v, list):
                            setattr(ns, k, [dict_to_namespace(item) if isinstance(item, dict) else item for item in v])
                        else:
                            setattr(ns, k, v)
                    return ns
                return d
            filter_columns = dict_to_namespace(filter_columns)
        
        filter_info = parse_filter_information(filter_columns=filter_columns)
        
        # Encode filters (simplified - full implementation would use column_statistics)
        for i, (column, operator, literal) in enumerate(filter_info[:max_filter_number]):
            op_str = str(operator) if operator else 'EQ'
            op_id = feature_statistics.get('operator', {}).get('value_dict', {}).get(op_str, 0)
            
            encoded_filter_info[i*3] = op_id
            encoded_filter_info[i*3+1] = column if isinstance(column, int) else 0
            encoded_filter_info[i*3+2] = 0.0  # Literal (simplified)
    
    # 5. Create TreeNode
    tree_node = TreeNode(
        operator_type_id=operator_type_id,
        filter_info=encoded_filter_info,
        sample_vec=sample_bitmap_vec,
        histogram_info=encoded_histogram_info,
        estimated_cardinality=estimated_cardinality,
        table_name=table_name
    )
    
    tree_node.feature_vector = tree_node.featurize()
    
    # Recursively convert children
    for child in plan.children:
        child.parent = plan
        child_node = recursively_convert_plan_unified(
            plan=child,
            index=index,
            dbms_name=dbms_name,
            feature_statistics=feature_statistics,
            db_statistics=db_statistics,
            dim_word_embedding=dim_word_embedding,
            dim_word_hash=dim_word_hash,
            word_embeddings=word_embeddings,
            dim_bitmaps=dim_bitmaps,
            max_filter_number=max_filter_number,
            histogram_bin_size=histogram_bin_size
        )
        tree_node.parent = plan
        tree_node.add_child(child_node)
    
    return tree_node


def encode_query_plan_unified(
    query_index: int,
    query_plan,
    dbms_name: str,
    feature_statistics: dict,
    db_statistics: StandardizedStatistics,
    max_node: int = 30,
    rel_pos_max: int = 20,
    max_filter_number: int = 5,
    max_num_joins: int = 5,
    histogram_bin_size: int = 10,
    **kwargs
):
    """
    Encode query plan for QueryFormer (unified version).
    
    Args:
        query_index: Query index
        query_plan: Plan operator tree
        dbms_name: DBMS name
        feature_statistics: Feature statistics
        db_statistics: Standardized database statistics
        ... (other parameters)
    
    Returns:
        tuple: (features, join_ids, attention_bias, rel_pos, node_heights)
    """
    # Get join IDs (simplified - full implementation would extract from plan)
    join_ids = np.zeros(max_num_joins)
    
    # Convert plan to tree node
    tree_node = recursively_convert_plan_unified(
        plan=query_plan,
        index=query_index,
        dbms_name=dbms_name,
        feature_statistics=feature_statistics,
        db_statistics=db_statistics,
        max_filter_number=max_filter_number,
        histogram_bin_size=histogram_bin_size,
        **kwargs
    )
    
    # Get adjacency matrix and features
    adjacency_matrix, number_of_children, features = topological_sort(root_node=tree_node)
    node_heights = calculate_node_heights(adjacency_matrix, len(features))
    
    # Convert to tensors
    features = torch.tensor(np.array(features), dtype=torch.float)
    node_heights = torch.tensor(np.array(node_heights), dtype=torch.long)
    adjacency_matrix = torch.tensor(np.array(adjacency_matrix), dtype=torch.long)
    join_ids = torch.Tensor(join_ids.reshape(1, max_num_joins))
    
    # Initialize attention bias
    attention_bias = torch.zeros([len(features) + 1, len(features) + 1], dtype=torch.float)
    
    # Calculate shortest paths
    edge_index = adjacency_matrix.t()
    if len(edge_index) == 0:
        shortest_path_result = np.array([[0]])
    else:
        boolean_adjacency = torch.zeros([len(features), len(features)], dtype=torch.bool)
        boolean_adjacency[edge_index[0, :], edge_index[1, :]] = True
        shortest_path_result = floyd_warshall_transform(boolean_adjacency.numpy())
    
    rel_pos = torch.from_numpy(shortest_path_result).long()
    
    return features, join_ids, attention_bias, rel_pos, node_heights


def unified_queryformer_collator(
    plans: List,
    dbms_name: str,
    feature_statistics: dict = None,
    db_statistics: dict = None,
    max_filter_number: int = 5,
    histogram_bin_size: int = 10,
    **kwargs
) -> Tuple:
    """
    Unified QueryFormer collator for all DBMS.
    
    Args:
        plans: List of (sample_idx, plan) tuples
        dbms_name: DBMS name
        feature_statistics: Feature statistics
        db_statistics: Database statistics (StandardizedStatistics format)
        max_filter_number: Maximum number of filters
        histogram_bin_size: Histogram bucket size
    
    Returns:
        tuple: ((features, join_ids, attention_bias, rel_pos, node_heights), labels, sample_idxs)
    
    Example:
        # PostgreSQL
        collate_fn = functools.partial(
            unified_queryformer_collator,
            dbms_name="postgres",
            feature_statistics=stats,
            db_statistics=db_stats
        )
        
        # Trino (same code!)
        collate_fn = functools.partial(
            unified_queryformer_collator,
            dbms_name="trino",  # ← Only this changes
            feature_statistics=stats,
            db_statistics=db_stats
        )
    """
    from models.query_former.dataloader import pad_1d_unsqueeze, pad_2d_unsqueeze, pad_attn_bias_unsqueeze, pad_rel_pos_unsqueeze
    
    all_features = []
    all_join_ids = []
    all_attention_bias = []
    all_rel_pos = []
    all_node_heights = []
    labels = []
    sample_idxs = []
    
    # Get standardized statistics
    db_stats = db_statistics.get(0) if db_statistics else StandardizedStatistics()
    
    for plan_index, p in plans:
        # Encode plan
        features, join_ids, attention_bias, rel_pos, node_heights = encode_query_plan_unified(
            query_index=plan_index,
            query_plan=p,
            dbms_name=dbms_name,
            feature_statistics=feature_statistics,
            db_statistics=db_stats,
            max_filter_number=max_filter_number,
            histogram_bin_size=histogram_bin_size,
            **kwargs
        )
        
        all_features.append(features)
        all_join_ids.append(join_ids)
        all_attention_bias.append(attention_bias)
        all_rel_pos.append(rel_pos)
        all_node_heights.append(node_heights)
        sample_idxs.append(plan_index)
        labels.append(torch.tensor(p.plan_runtime) / 1000)  # Convert to sec
    
    # Concatenate
    all_features = torch.cat(all_features)
    all_join_ids = torch.cat(all_join_ids)
    all_attention_bias = torch.cat(all_attention_bias)
    all_rel_pos = torch.cat(all_rel_pos)
    all_node_heights = torch.cat(all_node_heights)
    labels = torch.stack(labels)
    
    return (all_features, all_join_ids, all_attention_bias, all_rel_pos, all_node_heights), labels, sample_idxs

