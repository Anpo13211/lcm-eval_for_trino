import collections

from cross_db_benchmark.benchmark_tools.postgres.plan_operator import PlanOperator
import dgl
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from cross_db_benchmark.benchmark_tools.postgres.parse_filter import PredicateNode
from cross_db_benchmark.benchmark_tools.generate_workload import Operator
from training.featurizations import Featurization
from training.preprocessing.feature_statistics import FeatureType


def encode(column, plan_params, feature_statistics):
    """特徴量をエンコードする（dict/SimpleNamespace両対応）"""
    # SimpleNamespace統一により、dict/SimpleNamespace両対応
    use_dict = isinstance(plan_params, dict)
    
    # fallback in case actual cardinality is not in plan parameters
    if column == 'act_card' or column == 'act_output_rows':
        if use_dict:
            has_value = column in plan_params
        else:
            has_value = hasattr(plan_params, column)
        
        if not has_value:
            value = 0
        else:
            value = plan_params[column] if use_dict else getattr(plan_params, column, 0)
    else:
        value = plan_params[column] if use_dict else getattr(plan_params, column, 0)
    
    if feature_statistics[column].get('type') == str(FeatureType.numeric):
        enc_value = feature_statistics[column]['scaler'].transform(np.array([[value]])).item()
    elif feature_statistics[column].get('type') == str(FeatureType.categorical):
        value_dict = feature_statistics[column]['value_dict']
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        
        # 未知の値の処理（Trino版と同様）
        if str(value) not in value_dict:
            no_vals = feature_statistics[column].get('no_vals', len(value_dict))
            # デフォルト値として0を使用（通常は最も一般的な値）
            if 0 in value_dict.values():
                enc_value = 0
            else:
                # 0が存在しない場合は、最小インデックスを使用
                min_index = min(value_dict.values()) if value_dict else 0
                enc_value = min_index
        else:
            enc_value = value_dict[str(value)]
    else:
        raise NotImplementedError
    return enc_value


def plan_to_graph(node: PlanOperator, database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics, feature_statistics,
                  filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges, output_column_features,
                  column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                  output_column_idx, column_idx, table_idx, plan_featurization: Featurization, predicate_depths, intra_predicate_edges,
                  logical_preds, parent_node_id=None, depth=0):
    """
    引数一覧：
    1. 入力データ
        node: 現在処理中の実行計画ノード
        database_id: データベースの識別子
        db_statistics: データベースの統計情報
        feature_statistics: 特徴量の統計情報
        plan_featurization: 特徴量化の設定
    2. 出力用のリスト（参照渡しで更新）
        plan_depths: 各プランノードの深度
        plan_features: プランノードの特徴量
        plan_to_plan_edges: プランノード間のエッジ
        filter_to_plan_edges: フィルターからプランへのエッジ
        predicate_col_features: 述語の特徴量
        output_column_to_plan_edges: 出力カラムからプランへのエッジ
        output_column_features: 出力カラムの特徴量
        column_to_output_column_edges: カラムから出力カラムへのエッジ
        column_features: カラムの特徴量
        table_features: テーブルの特徴量
        table_to_plan_edges: テーブルからプランへのエッジ
        predicate_depths: 述語の深度
        intra_predicate_edges: 述語内のエッジ
        logical_preds: 論理述語かどうかのフラグ
    3. インデックス管理用辞書
        output_column_idx: 出力カラムのインデックス
        column_idx: カラムのインデックス
        table_idx: テーブルのインデックス
    4. 制御用パラメータ
        parent_node_id: 親ノードのID（再帰用）
        depth: 現在の深度（再帰用）
    
    実行プラン：
        Aggregate (depth=0)
        ├── Hash Join (depth=1)
        │   ├── Seq Scan (depth=2)
        │   └── Index Scan (depth=2)
        └── Sort (depth=1)
            └── Seq Scan (depth=2)
    のようなものが多く、depth によって planノードの階層を表現する。
    この場合：
    plan0: Aggregate
    plan1: Hash Join, Sort
    plan2: Seq Scan, Index Scan, Seq Scan
    になり、0 ← 1 ← 2 のようにプランの間にエッジが張られる。
    """
    plan_node_id = len(plan_depths)
    plan_depths.append(depth)

    # add plan features（SimpleNamespace統一により、dict/SimpleNamespace両対応）
    plan_params = node.plan_parameters
    use_dict = isinstance(plan_params, dict)
    
    # ヘルパー関数: plan_parametersから値を取得（dict/SimpleNamespace両対応）
    def get_param(key, default=None):
        return plan_params.get(key, default) if use_dict else getattr(plan_params, key, default)
    
    def has_param(key):
        return key in plan_params if use_dict else hasattr(plan_params, key)
    
    curr_plan_features = [encode(column, plan_params, feature_statistics) for column in plan_featurization.PLAN_FEATURES]
    plan_features.append(curr_plan_features)

    # encode output columns which can in turn have several columns as a product in the aggregation
    """
    output_columns は以下のような形式である。（plan_operator.py の parse_output_columns で生成される）
    output_columns = [
    {
        'aggregation': 'COUNT',  # または 'SUM', 'AVG', 'MIN', 'MAX', None
        'columns': [('table1', 'column1'), ('table2', 'column2')]  # タプルのリスト
    },
    {
        'aggregation': 'SUM',
        'columns': [('orders', 'amount')]
    },
    # ... 他の出力カラム
]
    """
    output_columns = get_param('output_columns')
    if output_columns is not None:
        for output_column in output_columns:
            output_column_node_id = output_column_idx.get(
                (output_column.aggregation, tuple(output_column.columns), database_id))

            # if not, create
            if output_column_node_id is None:
                # encode()は既にdict/SimpleNamespace両対応なので、vars()不要
                curr_output_column_features = [encode(column, output_column, feature_statistics)
                                               for column in plan_featurization.OUTPUT_COLUMN_FEATURES]

                output_column_node_id = len(output_column_features)
                output_column_features.append(curr_output_column_features)
                # 同じ出力カラムでもデータベースが違えば別物として扱うべきなので output_column_idx はこんなに複雑な key を持つ
                output_column_idx[(output_column.aggregation, tuple(output_column.columns), database_id)] = output_column_node_id

                # featurize product of columns if there are any
                # column edge と output_column edge を作成する(カラムの統計情報と出力カラムの集約方法の関係を学習する)
                db_column_features = db_statistics[database_id].column_stats
                for column in output_column.columns:
                    column_node_id = column_idx.get((column, database_id))
                    if column_node_id is None:
                        # encode()は既にdict/SimpleNamespace両対応なので、vars()不要
                        curr_column_features = [
                            encode(feature_name, db_column_features[column], feature_statistics)
                            for feature_name in plan_featurization.COLUMN_FEATURES]
                        column_node_id = len(column_features)
                        column_features.append(curr_column_features)
                        column_idx[(column, database_id)] = column_node_id
                    column_to_output_column_edges.append((column_node_id, output_column_node_id))

            # in any case add the corresponding edge
            output_column_to_plan_edges.append((output_column_node_id, plan_node_id))

    # filter_columns (we do not reference the filter columns to columns since we anyway have to create a node per filter
    #  node)
    filter_column = get_param('filter_columns')
    if filter_column is not None:
        db_column_features = db_statistics[database_id].column_stats

        # check if node already exists in the graph
        # filter_node_id = fitler_node_idx.get((filter_column.operator, filter_column.column, database_id))

        parse_predicates(db_column_features, feature_statistics, filter_column, filter_to_plan_edges,
                         plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                         logical_preds, plan_node_id=plan_node_id)

    # tables
    table = get_param('table')
    if table is not None:
        table_node_id = table_idx.get((table, database_id))
        db_table_statistics = db_statistics[database_id].table_stats

        if table_node_id is None:
            # encode()は既にdict/SimpleNamespace両対応なので、vars()不要
            curr_table_features = [encode(feature_name, db_table_statistics[table], feature_statistics)
                                   for feature_name in plan_featurization.TABLE_FEATURES]
            table_node_id = len(table_features)
            table_features.append(curr_table_features)
            table_idx[(table, database_id)] = table_node_id

        table_to_plan_edges.append((table_node_id, plan_node_id))

    # add edge to parent
    if parent_node_id is not None:
        plan_to_plan_edges.append((plan_node_id, parent_node_id))

    # continue recursively
    for c in node.children:
        plan_to_graph(c, database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics, feature_statistics,
                      filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges, output_column_features,
                      column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                      output_column_idx, column_idx, table_idx, plan_featurization, predicate_depths,
                      intra_predicate_edges, logical_preds, parent_node_id=plan_node_id, depth=depth + 1)


def parse_predicates(db_column_features, feature_statistics, filter_column: PredicateNode, filter_to_plan_edges, plan_featurization,
                     predicate_col_features, predicate_depths, intra_predicate_edges, logical_preds, plan_node_id=None,
                     parent_filter_node_id=None, depth=0):
    """
    Recursive parsing of predicate columns

    :param db_column_features:
    :param feature_statistics:
    :param filter_column:
    :param filter_to_plan_edges:
    :param plan_featurization:
    :param plan_node_id:
    :param predicate_col_features:
    :return:
    """
    filter_node_id = len(predicate_depths)
    predicate_depths.append(depth)

    # gather features
    if filter_column.operator in {str(op) for op in list(Operator)}:
        # encode()は既にdict/SimpleNamespace両対応なので、vars()不要
        curr_filter_features = [encode(feature_name, filter_column, feature_statistics)
                                for feature_name in plan_featurization.FILTER_FEATURES]

        if filter_column.column is not None:
            curr_filter_col_feats = [
                encode(column, db_column_features[filter_column.column], feature_statistics)
                for column in plan_featurization.COLUMN_FEATURES]
        # hack for cases in which we have no base filter column (e.g., in a having clause where the column is some
        # result column of a subquery/groupby). In the future, this should be replaced by some graph model that also
        # encodes the structure of this output column
        else:
            curr_filter_col_feats = [0 for _ in plan_featurization.COLUMN_FEATURES]
        # カラムの特徴量を追加する(list がそのまま append)
        curr_filter_features += curr_filter_col_feats
        logical_preds.append(False)

    else:
        # AND, OR　などの論理述語の場合
        # 論理述語などはカラムを持たないからカラム情報を curr_filter_features に追加しない
        # encode()は既にdict/SimpleNamespace両対応なので、vars()不要
        curr_filter_features = [encode(feature_name, filter_column, feature_statistics)
                                for feature_name in plan_featurization.FILTER_FEATURES]
        logical_preds.append(True)

    predicate_col_features.append(curr_filter_features)

    # add edge either to plan or inside predicates
    if depth == 0:
        assert plan_node_id is not None
        # in any case add the corresponding edge
        filter_to_plan_edges.append((filter_node_id, plan_node_id))

    else:
        assert parent_filter_node_id is not None
        intra_predicate_edges.append((filter_node_id, parent_filter_node_id))

    # recurse
    for c in filter_column.children:
        parse_predicates(db_column_features, feature_statistics, c, filter_to_plan_edges,
                         plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                         logical_preds, parent_filter_node_id=filter_node_id, depth=depth + 1)


def postgres_plan_collator(plans, feature_statistics: dict =None, db_statistics: dict =None,
                           plan_featurization: Featurization =None):
    """
    Combines physical plans into a large graph that can be fed into ML models.
    :return:
    """

    # output:
    #   - list of labels (i.e., plan runtimes)
    #   - feature dictionaries
    #       - column_features: matrix
    #       - output_column_features: matrix
    #       - filter_column_features: matrix
    #       - plan_node_features: matrix
    #       - table_features: matrix
    #       - logical_pred_features: matrix
    #   - edges
    #       - table_to_output
    #       - column_to_output
    #       - filter_to_plan
    #       - output_to_plan
    #       - plan_to_plan
    #       - intra predicate (e.g., column to AND)
    plan_depths = []
    plan_features = []
    plan_to_plan_edges = []
    filter_to_plan_edges = []
    filter_features = []
    output_column_to_plan_edges = []
    output_column_features = []
    column_to_output_column_edges = []
    column_features = []
    table_features = []
    table_to_plan_edges = []
    labels = []
    predicate_depths = []
    intra_predicate_edges = []
    logical_preds = []

    output_column_idx = dict()
    column_idx = dict()
    table_idx = dict()

    # prepare robust encoder for the numerical fields
    add_numerical_scalers(feature_statistics)

    # iterate over plans and create lists of edges and features per node
    sample_idxs = []
    # plans: 複数のクエリの実行計画が入っている
    for sample_idx, p in plans:
        sample_idxs.append(sample_idx)
        labels.append(p.plan_runtime) # それぞれのクエリの実行時間
        plan_to_graph(p, p.database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics,
                      feature_statistics, filter_to_plan_edges, filter_features, output_column_to_plan_edges,
                      output_column_features, column_to_output_column_edges, column_features, table_features,
                      table_to_plan_edges, output_column_idx, column_idx, table_idx,
                      plan_featurization, predicate_depths, intra_predicate_edges, logical_preds)

    assert len(labels) == len(plans)
    assert len(plan_depths) == len(plan_features)

    data_dict, nodes_per_depth, plan_dict = create_node_types_per_depth(plan_depths, plan_to_plan_edges)

    # similarly create node types:
    #   pred_node_{depth}, filter column
    pred_dict = dict()
    nodes_per_pred_depth = collections.defaultdict(int)
    no_filter_columns = 0
    for pred_node, d in enumerate(predicate_depths):
        # predicate node
        if logical_preds[pred_node]:
            pred_dict[pred_node] = (nodes_per_pred_depth[d], d)
            nodes_per_pred_depth[d] += 1
        # filter column
        else:
            pred_dict[pred_node] = no_filter_columns
            no_filter_columns += 1

    adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, logical_preds, plan_dict, pred_dict,
                          pred_node_type_id)

    # we additionally have filters, tables, columns, output_columns and plan nodes as node types
    # Attention: The following code assumes that column_to_output_column_edges is not none. This
    # However might be the case for very simple queries.
    # This can be mitigated by larger batch sizes
    # However, another fix would be to add a check here and
    # adapt the Message Passing scheme for the case of missing column-nodes
    data_dict[('column', 'col_output_col', 'output_column')] = column_to_output_column_edges
    for u, v in output_column_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[('output_column', 'to_plan', f'plan{d_v}')].append((u, v_node_id))
    for u, v in table_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[('table', 'to_plan', f'plan{d_v}')].append((u, v_node_id))

    # also pass number of nodes per type
    max_depth, max_pred_depth = get_depths(plan_depths, predicate_depths)
    num_nodes_dict = {
        'column': len(column_features),
        'table': len(table_features),
        'output_column': len(output_column_features),
        'filter_column': len(logical_preds) - sum(logical_preds),
    }
    num_nodes_dict = update_node_counts(max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth,
                                        num_nodes_dict)

    # create graph
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    graph.max_depth = max_depth
    graph.max_pred_depth = max_pred_depth

    features = collections.defaultdict(list)
    features.update(dict(column=column_features, table=table_features, output_column=output_column_features,
                         filter_column=[f for f, log_pred in zip(filter_features, logical_preds) if not log_pred]))
    # sort the plan features based on the depth
    for u, plan_feat in enumerate(plan_features):
        u_node_id, d_u = plan_dict[u]
        features[f'plan{d_u}'].append(plan_feat)

    # sort the predicate features based on the depth
    for pred_node_id, pred_feat in enumerate(filter_features):
        if not logical_preds[pred_node_id]:
            continue
        node_type, _ = pred_node_type_id(logical_preds, pred_dict, pred_node_id)
        features[node_type].append(pred_feat)

    features = postprocess_feats(features, num_nodes_dict)

    # rather deal with runtimes in secs
    labels = postprocess_labels(labels)

    return graph, features, labels, sample_idxs


def postprocess_labels(labels):
    labels = np.array(labels, dtype=np.float32)
    labels /= 1000
    # we do this later
    # labels = torch.from_numpy(labels)
    return labels


def postprocess_feats(features, num_nodes_dict):
    # convert to tensors, replace nan with 0®
    for k in features.keys():
        v = features[k]
        v = np.array(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0)
        v = torch.from_numpy(v)
        features[k] = v
    # filter out any node type with zero nodes
    features = {k: v for k, v in features.items() if k in num_nodes_dict}
    return features


def update_node_counts(max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth, num_nodes_dict):
    num_nodes_dict.update({f'plan{d}': nodes_per_depth[d] for d in range(max_depth + 1)})
    num_nodes_dict.update({f'logical_pred_{d}': nodes_per_pred_depth[d] for d in range(max_pred_depth + 1)})
    # filter out any node type with zero nodes
    num_nodes_dict = {k: v for k, v in num_nodes_dict.items() if v > 0}
    return num_nodes_dict


def get_depths(plan_depths, predicate_depths):
    max_depth = max(plan_depths)
    max_pred_depth = 0
    if len(predicate_depths) > 0:
        max_pred_depth = max(predicate_depths)
    return max_depth, max_pred_depth


def adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, logical_preds, plan_dict, pred_dict,
                          pred_node_type_id_func):
    # convert to plan edges
    for u, v in filter_to_plan_edges:
        # transform plan node to right id and depth
        v_node_id, d_v = plan_dict[v]
        # transform predicate node to right node type and id
        node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)

        data_dict[(node_type, 'to_plan', f'plan{d_v}')].append((u_node_id, v_node_id))
    # convert intra predicate edges (e.g. column to AND)
    for u, v in intra_predicate_edges:
        u_node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)
        v_node_type, v_node_id = pred_node_type_id_func(logical_preds, pred_dict, v)
        data_dict[(u_node_type, 'intra_predicate', v_node_type)].append((u_node_id, v_node_id))


def create_node_types_per_depth(plan_depths, plan_to_plan_edges):
    # now create heterograph with node types: table, column, filter_column, logical_pred, output_column, plan{depth}
    # for this, first create mapping of old plan node id -> depth and node id for depth
    plan_dict = dict()
    nodes_per_depth = collections.defaultdict(int)
    for plan_node, d in enumerate(plan_depths):
        plan_dict[plan_node] = (nodes_per_depth[d], d)
        nodes_per_depth[d] += 1
    # create edge and node types depending on depth in the plan
    data_dict = collections.defaultdict(list)
    for u, v in plan_to_plan_edges:
        u_node_id, d_u = plan_dict[u]
        v_node_id, d_v = plan_dict[v]
        assert d_v < d_u
        data_dict[(f'plan{d_u}', f'intra_plan', f'plan{d_v}')].append((u_node_id, v_node_id))
    return data_dict, nodes_per_depth, plan_dict


def add_numerical_scalers(feature_statistics):
    for k, v in feature_statistics.items():
        if v.get('type') == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = v['center']
            scaler.scale_ = v['scale']
            feature_statistics[k]['scaler'] = scaler


def pred_node_type_id(logical_preds, pred_dict, u):
    if logical_preds[u]:
        u_node_id, depth = pred_dict[u]
        node_type = f'logical_pred_{depth}'
    else:
        u_node_id = pred_dict[u]
        node_type = f'filter_column'
    return node_type, u_node_id
