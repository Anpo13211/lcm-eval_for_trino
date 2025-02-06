import dgl
import numpy as np

from models.workload_driven.dataset.plan_tree_batching import extract_dimensions, encode_predicate, encode_one_hot, \
    tensorize_feats
from cross_db_benchmark.benchmark_tools.generate_workload import LogicalOperator


def extract_predicate_features(n, pred_feats, column_statistics, db_statistics, dim_word_emdb, dim_word_hash,
                               feature_statistics, word_embeddings):
    # (column one hot, operator one hot, predicate)
    if not n.operator in {str(LogicalOperator.AND), str(LogicalOperator.OR)}:
        pred_feat_vec = encode_predicate(column_statistics, db_statistics, dim_word_emdb, dim_word_hash,
                                         feature_statistics, n, word_embeddings)
        pred_feats.append(pred_feat_vec)

    for c in n.children:
        extract_predicate_features(c, pred_feats, column_statistics, db_statistics, dim_word_emdb, dim_word_hash,
                                   feature_statistics, word_embeddings)


def extract_set_features(plan, feature_statistics, dim_word_hash, dim_word_emdb, column_statistics, db_statistics,
                         word_embeddings, dim_bitmaps, table_feats=None, pred_feats=None):
    if table_feats is None:
        table_feats = []
    if pred_feats is None:
        pred_feats = []

    params = vars(plan.plan_parameters)
    # predicate feature set
    filter_columns = params.get('filter_columns')
    sample_bitmap = np.ones(dim_bitmaps)
    if filter_columns is not None:
        sample_bitmap_vec = params.get('sample_vec')
        assert sample_bitmap_vec is not None, "Please make sure to augment sample vectors before use"
        sample_bitmap[:len(sample_bitmap_vec)] = np.array(sample_bitmap_vec)
        extract_predicate_features(filter_columns, pred_feats, column_statistics, db_statistics, dim_word_emdb,
                                   dim_word_hash, feature_statistics, word_embeddings)

    # table feature set
    table = params.get('table')
    if table is not None:
        _, _, dim_tables, _ = extract_dimensions(feature_statistics)
        table_vec = np.zeros(dim_tables)
        table_vec[table] = 1
        table_feat = np.concatenate([table_vec, sample_bitmap])
        table_feats.append(table_feat)

    for c in plan.children:
        extract_set_features(c, feature_statistics, dim_word_hash, dim_word_emdb, column_statistics, db_statistics,
                             word_embeddings, dim_bitmaps, table_feats=table_feats, pred_feats=pred_feats)

    return table_feats, pred_feats


def mscn_plan_collator(plans, feature_statistics=None, db_statistics=None, column_statistics=None, word_embeddings=None,
                       dim_word_hash=None, dim_word_emdb=None, dim_bitmaps=None):
    """
    Combines physical plans into a large graph that can be fed into ML models.
    :return:
    """
    # features pro query are sets:
    # - table set (table one hot, bitmap)
    # - join set (one hot: which tables are joined)
    # - predicate set (column one hot, operator one hot, predicate one hot)
    # - output column set (which aggregation and which column)

    # compute input dimensions for model
    db_statistics = db_statistics[0]
    features = {
        'table': [],
        'pred': [],
        'agg': [],
        'join': []
    }
    edge_dict = {
        ('table', 'table_plan', 'plan'): [],
        ('pred', 'pred_plan', 'plan'): [],
        ('agg', 'agg_plan', 'plan'): [],
        ('join', 'join_plan', 'plan'): []
    }
    labels = []

    sample_idxs = []
    for plan_id, (sample_idx, p) in enumerate(plans):
        sample_idxs.append(sample_idx)
        labels.append(p.plan_runtime)
        curr_table_feats, curr_pred_feats = extract_set_features(p, feature_statistics, dim_word_hash, dim_word_emdb,
                                                                 column_statistics, db_statistics, word_embeddings,
                                                                 dim_bitmaps)
        curr_agg_feats, curr_join_feats = extract_joins_aggregations(feature_statistics, p)

        # sanity checks
        assert curr_agg_feats is not None

        def append_edges_and_feats(features, plan_id, node_type, curr_feats, edges):
            edges += [(len(features[node_type]) + i, plan_id) for i in range(len(curr_feats))]
            features[node_type] += curr_feats

        append_edges_and_feats(features, plan_id, 'table', curr_table_feats, edge_dict[('table', 'table_plan', 'plan')])
        append_edges_and_feats(features, plan_id, 'pred', curr_pred_feats, edge_dict[('pred', 'pred_plan', 'plan')])
        append_edges_and_feats(features, plan_id, 'agg', curr_agg_feats, edge_dict[('agg', 'agg_plan', 'plan')])
        if curr_join_feats:
            append_edges_and_feats(features, plan_id, 'join', curr_join_feats, edge_dict[('join', 'join_plan', 'plan')])

    num_nodes_dict = {k: len(v) for k, v in features.items()}
    num_nodes_dict.update({'plan': len(plans)})

    graph = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)

    features = tensorize_feats(features)
    labels = np.array(labels, dtype=np.float32)
    labels /= 1000

    return graph, features, labels, sample_idxs


def extract_joins_aggregations(feature_statistics, p):
    # join feature set
    curr_join_feats = []
    for join_cond in p.join_conds:
        join_vec = encode_one_hot('join_conds', feature_statistics, join_cond)
        curr_join_feats.append(join_vec)
    # output column set
    curr_agg_feats = []
    dim_cols, _, _, _ = extract_dimensions(feature_statistics)
    for output_col in p.plan_parameters.output_columns:
        if output_col.aggregation is None:
            continue
        agg_vec = encode_one_hot('aggregation', feature_statistics, output_col.aggregation)
        col_vec = np.zeros(dim_cols)
        for c_idx in output_col.columns:
            col_vec[c_idx] = 1
        agg_feat = np.concatenate([agg_vec, col_vec])
        curr_agg_feats.append(agg_feat)
    return curr_agg_feats, curr_join_feats
