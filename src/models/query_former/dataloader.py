from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
from collections import deque
from torch import Tensor

from cross_db_benchmark.benchmark_tools.generate_workload import LogicalOperator
from models.query_former.utils import TreeNode, pad_attn_bias_unsqueeze, floyd_warshall_transform, pad_1d_unsqueeze, \
    pad_rel_pos_unsqueeze, pad_2d_unsqueeze
from training.training.utils import recursive_to


def calculate_node_heights(adjacency_matrix: List[Tuple[int, int]], tree_size: int):
    if tree_size == 1:
        return np.array([0])
    adjacency_matrix = np.array(adjacency_matrix)
    node_ids = np.arange(tree_size, dtype=int)
    node_order = np.zeros(tree_size, dtype=int)
    uneval_nodes = np.ones(tree_size, dtype=bool)
    parent_nodes = adjacency_matrix[:, 0]
    child_nodes = adjacency_matrix[:, 1]
    n = 0
    while uneval_nodes.any():
        uneval_mask = uneval_nodes[child_nodes]
        unready_parents = parent_nodes[uneval_mask]
        node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
        node_order[node2eval] = n
        uneval_nodes[node2eval] = False
        n += 1
    return node_order


def topological_sort(root_node: TreeNode) -> Tuple[List[Tuple[int, int]], List[int], List[np.ndarray]]:
    adjacency_matrix: List[Tuple[int, int]] = []  # from parent to children
    num_child: List[int] = []
    features: List[np.ndarray] = []

    to_visit = deque()
    to_visit.append((0, root_node))
    next_id = 1

    while to_visit:
        idx, node = to_visit.popleft()
        features.append(node.feature_vector)
        num_child.append(len(node.children))
        for child in node.children:
            to_visit.append((next_id, child))
            adjacency_matrix.append((idx, next_id))
            next_id += 1
    return adjacency_matrix, num_child, features


def parse_filter_information(filter_columns: SimpleNamespace, filter_info=None):
    if filter_info is None:
        filter_info = []
    filter_col = filter_columns.column
    filter_op = filter_columns.operator
    filter_literal = filter_columns.literal
    
    # columnがNoneかつliteralがリスト形式の場合はスキップ（IN句の子ノード）
    if filter_col is None and isinstance(filter_literal, (list, tuple)):
        # このノードは追加しない（IN句の不完全な子ノード）
        pass
    else:
        if (filter_col, filter_op, filter_literal) not in filter_info:
            filter_info.append((filter_col, filter_op, filter_literal))
    
    if hasattr(filter_columns, "children"):
        for child in filter_columns.children:
            filter_info = parse_filter_information(child, filter_info)
    return filter_info


def get_encoded_filter(filter_info: List[Tuple[int, int, object]],
                       feature_statistics: dict,
                       column_statistics: dict,
                       database_statistics: SimpleNamespace) -> np.ndarray:
    encoded_filters = []
    for (column, operator, literal) in filter_info:
        # literalがリスト、タプル、または空文字列の場合は0.0に変換
        # （QueryFormerは文字列リテラルをサポートしていない）
        if isinstance(literal, (list, tuple)) or (isinstance(literal, str) and not literal):
            literal = 0.0
        
        # Encode the filter operator that always need to exist
        # operatorを文字列に変換（Operator enumの場合）
        operator_str = str(operator) if operator is not None else 'EQ'
        
        # value_dictに存在しない場合はダミー値を使用
        if operator_str in feature_statistics['operator']['value_dict']:
            encoded_operator = feature_statistics['operator']['value_dict'][operator_str]
        else:
            # 未知の演算子の場合はダミー値（最大値+1）
            encoded_operator = feature_statistics['operator']['no_vals'] + 1

        # Some combinations do not have a filter column, in this case set to NaN (which is the max value + 1)
        if operator in {str(LogicalOperator.AND), str(LogicalOperator.OR)}:
            encoded_column = feature_statistics['column']['max'] + 1
            encoded_literal = 0.0

        else:
            # columnがNoneまたは無効な場合はダミー値を使用
            if column is None or not isinstance(column, int):
                encoded_column = feature_statistics['column']['max']
                encoded_literal = 0.0
            else:
                encoded_column = column  # No scaling required, as it is a categorical identifier
                column_name = database_statistics.column_stats[column].attname
                table_name = database_statistics.column_stats[column].tablename
                col_statistics = column_statistics[table_name][column_name]
                if col_statistics['datatype'] in {'float', 'int'}:
                    assert literal is not None
                    encoded_literal = (literal - col_statistics['min'])
                    if col_statistics['max'] - col_statistics['min'] > 0:
                        encoded_literal /= (col_statistics['max'] - col_statistics['min'])
                else:
                    # According to the official code, QueryFormer does not support string predicates!"
                    encoded_literal = 0.0
        
        # 最終確認: encoded_literalが数値型であることを保証
        if not isinstance(encoded_literal, (int, float, np.number)):
            encoded_literal = 0.0
        
        encoded_filters.append((encoded_column, encoded_operator, encoded_literal))
    
    # すべての要素が数値型であることを確認
    result = np.asarray(encoded_filters, dtype=np.float64)
    return result


def get_encoded_histograms(filter_info: List[Tuple[int, int, object]],
                           column_statistics: dict,
                           database_statistics: SimpleNamespace,
                           histogram_bucket_size: int = 10) -> np.ndarray:
    encoded_histograms = []
    for (column, operator, literal) in filter_info:
        if operator in {str(LogicalOperator.AND), str(LogicalOperator.OR)}:
            encoded_histograms.append(np.zeros(histogram_bucket_size))
        else:
            # columnがNoneまたは無効な場合は空のヒストグラムを使用
            if column is None or not isinstance(column, int):
                encoded_histograms.append(np.zeros(histogram_bucket_size))
            else:
                col_name = database_statistics.column_stats[column].attname
                table_name = database_statistics.column_stats[column].tablename
                col_stats = column_statistics[table_name][col_name]
                encoded_histogram = np.zeros(histogram_bucket_size)
                # Do histogram encoding as written in the paper.
                # Categorical columns are not supported
                if col_stats['datatype'] in {'float', 'int'}:
                    percentiles = col_stats['percentiles']
                    if percentiles:  # percentilesがNoneでないことを確認
                        for i in range(len(percentiles) - 1):
                            start_val = percentiles[i]
                            end_val = percentiles[i + 1]
                            if isinstance(literal, (int, float)):  # literalが数値型の場合のみ
                                if start_val <= literal < end_val:
                                    encoded_histogram[i] = 1
                                if '<' in operator and literal < end_val:
                                    encoded_histogram[i] = 1
                                elif '>' in operator and literal >= start_val:
                                    encoded_histogram[i] = 1
                encoded_histograms.append(encoded_histogram)
    return np.asarray(encoded_histograms)


def get_sample_vector(sample_bitmap_vec: List[int], dim_bitmaps: int) -> torch.Tensor:
    if sample_bitmap_vec is None:
        sample_bitmap_vec = [0] * dim_bitmaps
    if len(sample_bitmap_vec) < dim_bitmaps:
        sample_bitmap_vec = sample_bitmap_vec + [0] * (dim_bitmaps - len(sample_bitmap_vec))
    assert len(sample_bitmap_vec) == dim_bitmaps
    return torch.tensor(sample_bitmap_vec)


def recursively_convert_plan(plan: SimpleNamespace,
                             index: int,
                             feature_statistics: dict,
                             column_statistics: dict,
                             database_statistics: SimpleNamespace,
                             dim_word_embedding: int,
                             dim_word_hash: int,
                             word_embeddings: object,
                             dim_bitmaps: int = 1000,
                             max_filter_number: int = 5,
                             histogram_bin_size: int = 10) -> TreeNode:
    # vars() で SimpleNamespace を辞書に変換
    plan_parameters = vars(plan.plan_parameters)

    # 1. Get operator type and type id
    operator_name = plan_parameters.get('op_name', 'Unknown')
    operator_type_id = feature_statistics['op_name']['value_dict'][operator_name]

    # 2. Get table name (encode to numeric ID)
    table_name_raw = plan_parameters.get('tablename', None)
    if isinstance(table_name_raw, int):
        table_name = table_name_raw
    elif isinstance(table_name_raw, str):
        # Map string table name to numeric ID; fallback to dummy if missing
        try:
            value_dict = feature_statistics.get('tablename', {}).get('value_dict', {})
        except Exception:
            value_dict = {}
        table_name = value_dict.get(table_name_raw, feature_statistics['tablename']['no_vals'] + 1)
    else:
        table_name = feature_statistics['tablename']['no_vals'] + 1  # dummy table

    # Parse and encode filters and sample vectors.
    # Each operator has a filter according to the original code. It however can be "empty"
    filter_info = [None]
    empty_filter_vec = np.array([feature_statistics['column']['max'] + 1,        # dummy column
                                 feature_statistics['operator']['no_vals'] + 1,  # dummy operator
                                 0.0]                                            # dummy literal
                               ).reshape(1, 3)

    histogram_info = np.empty([], dtype=float)
    filter_columns = plan_parameters.get('filter_columns', None)
    if filter_columns is not None:
        filter_info = parse_filter_information(filter_columns=filter_columns)
        assert len(filter_info) <= max_filter_number, f"Filter number exceeds max filter number, {filter_info}"
        # Get sample vector
        sample_bitmap_vec = get_sample_vector(sample_bitmap_vec=plan_parameters.get('sample_vec', None),
                                              dim_bitmaps=dim_bitmaps)

        encoded_filter_info = get_encoded_filter(filter_info=filter_info,
                                                 feature_statistics=feature_statistics,
                                                 column_statistics=column_statistics,
                                                 database_statistics=database_statistics)

        encoded_histogram_info = get_encoded_histograms(filter_info=filter_info,
                                                        column_statistics=column_statistics,
                                                        database_statistics=database_statistics,
                                                        histogram_bucket_size=histogram_bin_size)
        assert not np.isnan(encoded_histogram_info).any(), f"Nans in histogram info: {encoded_histogram_info}"
        
        # histogram_infoにencoded_histogram_infoを設定
        histogram_info = encoded_histogram_info

    else:
        sample_bitmap_vec = np.zeros(dim_bitmaps)
        encoded_filter_info = empty_filter_vec
        histogram_info = np.zeros(shape=(1, histogram_bin_size), dtype=np.float64)
    assert not np.isnan(sample_bitmap_vec).any(), f"Nans in sample bitmap found, {sample_bitmap_vec}"

    # Pad filter information to max_filter_number
    for i in range(len(filter_info), max_filter_number):
        encoded_filter_info = np.concatenate([encoded_filter_info, empty_filter_vec])
    assert not np.isnan(encoded_filter_info).any(), f"NANs in encoded filter info found, {encoded_filter_info}"

    # Pad Histograms
    if histogram_info.ndim != 0:
        num_histograms = histogram_info.shape[0]
    else:
        histogram_info = np.zeros(shape=(1, histogram_bin_size), dtype=np.float64)
        num_histograms = 1
    for i in range(num_histograms, max_filter_number):
        histogram_info = np.concatenate([histogram_info, np.zeros(shape=(1, histogram_bin_size), dtype=np.float64)])

    assert histogram_info.shape == (max_filter_number, histogram_bin_size)
    assert not np.isnan(histogram_info).any(), f"NANs in histogram info found, {histogram_info}"

    # Create a filter mask
    filter_mask = np.concatenate((np.ones(len(filter_info)), np.zeros(max_filter_number - len(filter_info))))
    assert not np.isnan(filter_mask).any(), f"NANs in filter_mask, found, {filter_mask}"

    tree_node = TreeNode(operator_type_id=operator_type_id,
                         filter_info=encoded_filter_info,
                         filter_mask=filter_mask,
                         sample_bitmap_vec=sample_bitmap_vec,
                         histogram_info=histogram_info,
                         table_name=table_name)

    tree_node.feature_vector = tree_node.featurize()
    # 安全なNaNチェック（数値型の配列の場合のみ）
    try:
        if tree_node.feature_vector.dtype.kind in ('f', 'i', 'u'):  # float、int、unsigned int
            assert not np.isnan(tree_node.feature_vector).any(), "NaN in feature vector"
    except (AttributeError, TypeError):
        # dtype属性がない、または他の型の場合はスキップ
        pass

    # Recursively convert children
    for children in plan.children:
        children.parent = plan  # Add parent plan to children as well.
        child = recursively_convert_plan(plan=children,
                                         index=index,
                                         feature_statistics=feature_statistics,
                                         column_statistics=column_statistics,
                                         database_statistics=database_statistics,
                                         dim_word_embedding=dim_word_embedding,
                                         dim_word_hash=dim_word_hash,
                                         word_embeddings=word_embeddings,
                                         max_filter_number=max_filter_number)
        tree_node.parent = plan
        tree_node.add_child(child)
    return tree_node


def encode_query_plan(query_index: int,
                      query_plan: SimpleNamespace,
                      feature_statistics: dict,
                      column_statistics: dict,
                      word_embeddings: object,
                      dim_word_embedding: int,
                      dim_word_hash: int, dim_bitmaps: int,
                      database_statistics: SimpleNamespace,
                      max_node: int = 30,
                      rel_pos_max: int = 20,
                      max_filter_number: int = 5,
                      max_num_joins: int = 5,
                      histogram_bin_size: int = 10):
    # Get Join IDs per query
    join_ids = []
    # join_condsが存在しない場合は空リストとして扱う
    if hasattr(query_plan, 'join_conds') and query_plan.join_conds:
        for join_cond in query_plan.join_conds:
            if join_cond in feature_statistics['join_conds']['value_dict']:
                join_ids.append(feature_statistics['join_conds']['value_dict'][join_cond])
            else:
                # 未知の結合条件の場合は0を追加（ダミー値）
                join_ids.append(0)
    # 結合条件が不足している場合は0でパディング
    for i in range(len(join_ids), max_num_joins):
        join_ids.append(0)
    join_ids = torch.Tensor(np.array(join_ids).reshape(1, max_num_joins))

    # Convert plan to tree node
    tree_node: TreeNode = recursively_convert_plan(plan=query_plan,
                                                   index=query_index,
                                                   feature_statistics=feature_statistics,
                                                   column_statistics=column_statistics,
                                                   database_statistics=database_statistics,
                                                   dim_bitmaps=dim_bitmaps,
                                                   dim_word_hash=dim_word_hash,
                                                   dim_word_embedding=dim_word_embedding,
                                                   word_embeddings=word_embeddings,
                                                   max_filter_number=max_filter_number,
                                                   histogram_bin_size=histogram_bin_size)

    # Get adjacency matrix, num_child, features
    adjacency_matrix, number_of_children, features = topological_sort(root_node=tree_node)
    node_heights = calculate_node_heights(adjacency_matrix, len(features))

    # Do conversions
    features = torch.tensor(np.array(features), dtype=torch.float)
    node_heights = torch.tensor(np.array(node_heights), dtype=torch.long)
    adjacency_matrix = torch.tensor(np.array(adjacency_matrix), dtype=torch.long)

    # Initialize attention bias according to num_features plus extra entry
    attention_bias = torch.zeros([len(features) + 1, len(features) + 1], dtype=torch.float)

    # Transpose adjacency matrix to get edge index
    edge_index = adjacency_matrix.t()

    # Calculate the shortest path between all pairs of nodes in the graph
    if len(edge_index) == 0:
        shortest_path_result = np.array([[0]])
    else:
        boolean_adjacency = torch.zeros([len(features), len(features)], dtype=torch.bool)
        boolean_adjacency[edge_index[0, :], edge_index[1, :]] = True
        shortest_path_result = floyd_warshall_transform(boolean_adjacency.numpy())

    # Convert the shortest path result to a tensor
    rel_pos = torch.from_numpy(shortest_path_result).long()

    # Set elements of attention_bias to -inf if the shortest path is greater than rel_pos_max
    # This is to prevent the model from attending to nodes that are too far away
    attention_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')

    # Pad the attention bias tensor and extra dimension.
    attention_bias = pad_attn_bias_unsqueeze(attention_bias, max_node + 1)

    rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)
    final_node_heights = pad_1d_unsqueeze(node_heights, max_node)
    features = pad_2d_unsqueeze(features, max_node)

    return features, join_ids, attention_bias, rel_pos, final_node_heights


def query_former_plan_collator(plans: List[SimpleNamespace],
                               feature_statistics: dict = None,
                               db_statistics: SimpleNamespace = None,
                               column_statistics: dict = None,
                               word_embeddings=None,
                               dim_word_hash=None,
                               dim_word_embedding=None,
                               histogram_bin_size: int = None,
                               max_num_filters: int = None,
                               dim_bitmaps=None) -> Tuple[
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor, List[int]]:
    labels = []
    all_features = []
    all_join_ids = []
    all_attention_bias = []
    all_rel_pos = []
    all_node_heights = []
    sample_idxs = []

    for plan_index, p in plans:
        features, join_ids, attention_bias, rel_pos, node_heights = encode_query_plan(query_index=plan_index,
                                                                                      query_plan=p,
                                                                                      feature_statistics=feature_statistics,
                                                                                      column_statistics=column_statistics,
                                                                                      database_statistics=db_statistics[
                                                                                          0],
                                                                                      dim_word_embedding=dim_word_embedding,
                                                                                      dim_word_hash=dim_word_hash,
                                                                                      dim_bitmaps=dim_bitmaps,
                                                                                      word_embeddings=word_embeddings,
                                                                                      histogram_bin_size=histogram_bin_size,
                                                                                      max_filter_number=max_num_filters)
        all_features.append(features)
        all_join_ids.append(join_ids)
        all_attention_bias.append(attention_bias)
        all_rel_pos.append(rel_pos)
        all_node_heights.append(node_heights)
        sample_idxs.append(plan_index)
        labels.append(torch.tensor(p.plan_runtime) / 1000)

    all_features = torch.cat(all_features)
    all_join_ids = torch.cat(all_join_ids)
    all_attention_bias = torch.cat(all_attention_bias)
    all_rel_pos = torch.cat(all_rel_pos)
    all_node_heights = torch.cat(all_node_heights)

    return (all_features, all_join_ids, all_attention_bias, all_rel_pos, all_node_heights), torch.Tensor(
        labels), sample_idxs


def query_former_batch_to(batch, device, label_norm):
    (features, join_ids, attention_bias, rel_pos, node_heights), label, sample_idxs = batch
    if label_norm is not None:
        label = label_norm.transform(np.asarray(label).reshape(-1, 1))
        label = label.reshape(-1)

    label = torch.as_tensor(label, device=device, dtype=torch.float)
    recursive_to(features, device)
    recursive_to(join_ids, device)
    recursive_to(attention_bias, device)
    recursive_to(rel_pos, device)
    recursive_to(node_heights, device)
    return (features, join_ids, attention_bias, rel_pos, node_heights), label, sample_idxs
