import collections

from cross_db_benchmark.benchmark_tools.trino.plan_operator import TrinoPlanOperator
import dgl
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from cross_db_benchmark.benchmark_tools.trino.parse_filter import PredicateNode
from cross_db_benchmark.benchmark_tools.generate_workload import Operator
from training.featurizations import Featurization
from training.preprocessing.feature_statistics import FeatureType


def encode(column, plan_params, feature_statistics):
    """特徴量をエンコードする"""
    # fallback in case actual cardinality is not in plan parameters
    if column == 'act_output_rows' and column not in plan_params:
        value = 0
    else:
        value = plan_params[column]
    
    if feature_statistics[column].get('type') == str(FeatureType.numeric):
        enc_value = feature_statistics[column]['scaler'].transform(np.array([[value]])).item()
    elif feature_statistics[column].get('type') == str(FeatureType.categorical):
        value_dict = feature_statistics[column]['value_dict']
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        
        # 未知の演算子名に対して動的にインデックスを割り当て
        if str(value) not in value_dict:
            # 新しいインデックスを割り当て（既存の最大値+1）
            max_index = max(value_dict.values()) if value_dict else -1
            value_dict[str(value)] = max_index + 1
            print(f"⚠️  未知の演算子名 '{value}' にインデックス {max_index + 1} を割り当てました")
        
        enc_value = value_dict[str(value)]
    else:
        raise NotImplementedError
    return enc_value


def plan_to_graph(node: TrinoPlanOperator, database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics, feature_statistics,
                  filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges, output_column_features,
                  column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                  output_column_idx, column_idx, table_idx, plan_featurization: Featurization, predicate_depths, intra_predicate_edges,
                  logical_preds, parent_node_id=None, depth=0):
    """
    Trinoプランをグラフに変換する
    
    引数一覧：
    1. 入力データ
        node: 現在処理中のTrino実行計画ノード
        database_id: データベースの識別子
        db_statistics: データベースの統計情報
        feature_statistics: 特徴量の統計情報
        plan_featurization: 特徴量化の設定
    2. 出力用のリスト（参照渡しで更新）
        plan_depths: 各プランノードの深度
        plan_features: プランノードの特徴量
        plan_to_plan_edges: プランノード間のエッジ
        filter_to_plan_edges: フィルターからプランへのエッジ
        predicate_col_features: 述語カラムの特徴量
        output_column_to_plan_edges: 出力カラムからプランへのエッジ
        output_column_features: 出力カラムの特徴量
        column_to_output_column_edges: カラムから出力カラムへのエッジ
        column_features: カラムの特徴量
        table_features: テーブルの特徴量
        table_to_plan_edges: テーブルからプランへのエッジ
        output_column_idx: 出力カラムのインデックス
        column_idx: カラムのインデックス
        table_idx: テーブルのインデックス
        plan_featurization: プランの特徴量化設定
        predicate_depths: 述語の深度
        intra_predicate_edges: 述語内のエッジ
        logical_preds: 論理述語
        parent_node_id: 親ノードのID
        depth: 現在の深度
    """
    
    # 現在のプランノードのIDを取得
    current_plan_id = len(plan_depths)
    plan_depths.append(depth)
    
    # プランの特徴量を抽出
    plan_feat = []
    for feat_name in plan_featurization.VARIABLES['plan']:
        if feat_name in node.plan_parameters:
            # Trinoの特徴量名をPostgreSQL互換に変換
            if feat_name == 'act_output_rows':
                # PostgreSQLのact_cardに対応
                value = node.plan_parameters[feat_name]
            elif feat_name == 'est_rows':
                # PostgreSQLのest_cardに対応
                value = node.plan_parameters[feat_name]
            else:
                value = node.plan_parameters[feat_name]
            
            enc_value = encode(feat_name, node.plan_parameters, feature_statistics)
            plan_feat.append(enc_value)
        else:
            # デフォルト値
            plan_feat.append(0.0)
    
    plan_features.append(plan_feat)
    
    # 親子関係のエッジを追加
    if parent_node_id is not None:
        plan_to_plan_edges.append((current_plan_id, parent_node_id))
    
    # テーブル情報の処理
    if 'table' in node.plan_parameters:
        table_name = node.plan_parameters['table']
        
        # テーブルのインデックスを取得または作成
        if table_name not in table_idx:
            table_idx[table_name] = len(table_features)
            
            # テーブルの特徴量を抽出
            table_feat = []
            for feat_name in plan_featurization.VARIABLES['table']:
                if feat_name in node.plan_parameters:
                    value = node.plan_parameters[feat_name]
                    enc_value = encode(feat_name, node.plan_parameters, feature_statistics)
                    table_feat.append(enc_value)
                else:
                    table_feat.append(0.0)
            
            table_features.append(table_feat)
        
        # テーブルからプランへのエッジを追加
        table_to_plan_edges.append((table_idx[table_name], current_plan_id))
    
    # カラム情報の処理
    if 'columns' in node.plan_parameters:
        columns = node.plan_parameters['columns']
        for column in columns:
            # カラムのインデックスを取得または作成
            if column not in column_idx:
                column_idx[column] = len(column_features)
                
                # カラムの特徴量を抽出（Trinoでは0に設定）
                column_feat = []
                for feat_name in plan_featurization.VARIABLES['column']:
                    column_feat.append(0.0)  # Trinoではカラム統計を取得できない
                
                column_features.append(column_feat)
    
    # 出力カラム情報の処理
    if 'output_columns' in node.plan_parameters:
        output_columns = node.plan_parameters['output_columns']
        for output_col in output_columns:
            # 出力カラムのインデックスを取得または作成
            if output_col not in output_column_idx:
                output_column_idx[output_col] = len(output_column_features)
                
                # 出力カラムの特徴量を抽出
                output_col_feat = []
                for feat_name in plan_featurization.VARIABLES['output_column']:
                    output_col_feat.append(0.0)  # デフォルト値
                
                output_column_features.append(output_col_feat)
            
            # 出力カラムからプランへのエッジを追加
            output_column_to_plan_edges.append((output_column_idx[output_col], current_plan_id))
    
    # フィルター情報の処理
    if 'filter_predicate' in node.plan_parameters and node.plan_parameters['filter_predicate']:
        filter_predicate = node.plan_parameters['filter_predicate']
        
        # フィルターの特徴量を抽出
        filter_feat = []
        for feat_name in plan_featurization.VARIABLES['filter_column']:
            if feat_name == 'operator':
                # フィルター演算子の特徴量
                filter_feat.append(0.0)  # デフォルト値
            elif feat_name == 'literal_feature':
                # リテラル値の特徴量
                filter_feat.append(0.0)  # デフォルト値
            else:
                filter_feat.append(0.0)
        
        predicate_col_features.append(filter_feat)
        filter_to_plan_edges.append((len(predicate_col_features) - 1, current_plan_id))
    
    # 動的フィルター情報の処理
    if 'dynamic_filters' in node.plan_parameters and node.plan_parameters['dynamic_filters']:
        dynamic_filters = node.plan_parameters['dynamic_filters']
        
        # 動的フィルターの特徴量を抽出
        dynamic_filter_feat = []
        for feat_name in plan_featurization.VARIABLES['filter_column']:
            dynamic_filter_feat.append(0.0)  # デフォルト値
        
        predicate_col_features.append(dynamic_filter_feat)
        filter_to_plan_edges.append((len(predicate_col_features) - 1, current_plan_id))
    
    # 子ノードを再帰的に処理
    for child in node.children:
        plan_to_graph(child, database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics,
                      feature_statistics, filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges,
                      output_column_features, column_to_output_column_edges, column_features, table_features,
                      table_to_plan_edges, output_column_idx, column_idx, table_idx,
                      plan_featurization, predicate_depths, intra_predicate_edges, logical_preds,
                      parent_node_id=current_plan_id, depth=depth + 1)


def trino_plan_collator(plans, feature_statistics: dict = None, db_statistics: dict = None,
                        plan_featurization: Featurization = None):
    """
    Trinoの物理プランを大きなグラフに結合し、MLモデルに投入できる形式にする
    
    Args:
        plans: Trinoプランのリスト
        feature_statistics: 特徴量の統計情報
        db_statistics: データベースの統計情報
        plan_featurization: プランの特徴量化設定
    
    Returns:
        graph: DGLの異種グラフ
        features: ノード特徴量の辞書
        labels: ラベル（実行時間）のリスト
        sample_idxs: サンプルインデックスのリスト
    """
    
    # 出力用のリスト
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
    
    # 数値フィールド用のロバストエンコーダーを準備
    if feature_statistics is not None:
        add_numerical_scalers(feature_statistics)
    
    # プランを反復処理し、ノードごとのエッジと特徴量のリストを作成
    sample_idxs = []
    for sample_idx, p in plans:
        sample_idxs.append(sample_idx)
        labels.append(p.plan_runtime)  # 各クエリの実行時間
        plan_to_graph(p, p.database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics,
                      feature_statistics, filter_to_plan_edges, filter_features, output_column_to_plan_edges,
                      output_column_features, column_to_output_column_edges, column_features, table_features,
                      table_to_plan_edges, output_column_idx, column_idx, table_idx,
                      plan_featurization, predicate_depths, intra_predicate_edges, logical_preds)
    
    assert len(labels) == len(plans)
    assert len(plan_depths) == len(plan_features)
    
    # 深度に基づいてノードタイプを作成
    data_dict, nodes_per_depth, plan_dict = create_node_types_per_depth(plan_depths, plan_to_plan_edges)
    
    # 述語ノードタイプを作成
    pred_dict = dict()
    nodes_per_pred_depth = collections.defaultdict(int)
    no_filter_columns = 0
    for pred_node, d in enumerate(predicate_depths):
        if logical_preds[pred_node]:
            pred_dict[pred_node] = (nodes_per_pred_depth[d], d)
            nodes_per_pred_depth[d] += 1
        else:
            pred_dict[pred_node] = no_filter_columns
            no_filter_columns += 1
    
    adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, logical_preds, plan_dict, pred_dict,
                          pred_node_type_id)
    
    # フィルター、テーブル、カラム、出力カラム、プランノードをノードタイプとして追加
    # 空でないエッジのみを追加
    if column_to_output_column_edges:
        data_dict[('column', 'col_output_col', 'output_column')] = column_to_output_column_edges
    
    for u, v in output_column_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[('output_column', 'to_plan', f'plan{d_v}')].append((u, v_node_id))
    
    for u, v in table_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[('table', 'to_plan', f'plan{d_v}')].append((u, v_node_id))
    
    # ノードタイプごとのノード数も渡す
    max_depth, max_pred_depth = get_depths(plan_depths, predicate_depths)
    num_nodes_dict = {
        'column': len(column_features),
        'table': len(table_features),
        'output_column': len(output_column_features),
        'filter_column': max(1, len(logical_preds) - sum(logical_preds)),
    }
    num_nodes_dict = update_node_counts(max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth,
                                        num_nodes_dict)
    
    # データ辞書に存在するノードタイプのみをnum_nodes_dictに含める
    used_node_types = set()
    for edge_type in data_dict.keys():
        used_node_types.add(edge_type[0])  # ソースノードタイプ
        used_node_types.add(edge_type[2])  # ターゲットノードタイプ
    
    # ノード数が0より大きいノードタイプも含める
    for node_type, count in num_nodes_dict.items():
        if count > 0:
            used_node_types.add(node_type)
    
    # 使用されるノードタイプのみをnum_nodes_dictに含める（ノード数が0でも含める）
    filtered_num_nodes_dict = {k: v for k, v in num_nodes_dict.items() if k in used_node_types}
    
    # デバッグ情報を出力
    print(f"🔍 num_nodes_dict: {num_nodes_dict}")
    print(f"🔍 used_node_types: {used_node_types}")
    print(f"🔍 filtered_num_nodes_dict: {filtered_num_nodes_dict}")
    
    # グラフを作成
    graph = dgl.heterograph(data_dict, num_nodes_dict=filtered_num_nodes_dict)
    graph.max_depth = max_depth
    graph.max_pred_depth = max_pred_depth
    
    # 特徴量を整理
    features = collections.defaultdict(list)
    
    # filter_columnの特徴量を作成
    filter_column_features = []
    if logical_preds:
        # logical_predsが存在する場合
        filter_column_features = [f for f, log_pred in zip(filter_features, logical_preds) if not log_pred]
    else:
        # logical_predsが空の場合、filter_columnノード数分のダミー特徴量を作成
        filter_column_count = max(1, len(logical_preds) - sum(logical_preds))
        if filter_column_count > 0:
            # ダミーの特徴量を作成（すべて0）
            dummy_feature = [0.0] * len(plan_featurization.FILTER_FEATURES + plan_featurization.COLUMN_FEATURES)
            filter_column_features = [dummy_feature for _ in range(filter_column_count)]
    
    features.update(dict(column=column_features, table=table_features, output_column=output_column_features,
                         filter_column=filter_column_features))
    
    # 深度に基づいてプラン特徴量をソート
    for u, plan_feat in enumerate(plan_features):
        u_node_id, d_u = plan_dict[u]
        features[f'plan{d_u}'].append(plan_feat)
    
    # 深度に基づいて述語特徴量をソート
    for pred_node_id, pred_feat in enumerate(filter_features):
        # インデックスが範囲内かチェック
        if pred_node_id >= len(logical_preds):
            continue
        
        if not logical_preds[pred_node_id]:
            continue
        node_type, _ = pred_node_type_id(logical_preds, pred_dict, pred_node_id)
        features[node_type].append(pred_feat)
    
    features = postprocess_feats(features, filtered_num_nodes_dict)
    
    # 実行時間を秒単位で処理
    labels = postprocess_labels(labels)
    
    return graph, features, labels, sample_idxs


def postprocess_labels(labels):
    """ラベルを後処理する"""
    labels = np.array(labels, dtype=np.float32)
    labels /= 1000  # ミリ秒から秒に変換
    return labels


def postprocess_feats(features, num_nodes_dict):
    """特徴量を後処理する"""
    # テンソルに変換し、nanを0に置換
    for k in features.keys():
        v = features[k]
        v = np.array(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0)
        v = torch.from_numpy(v)
        features[k] = v
    # ノード数が0のノードタイプをフィルター
    features = {k: v for k, v in features.items() if k in num_nodes_dict}
    return features


def update_node_counts(max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth, num_nodes_dict):
    """ノード数を更新する"""
    for d in range(max_depth + 1):
        num_nodes_dict[f'plan{d}'] = nodes_per_depth[d]
    for d in range(max_pred_depth + 1):
        num_nodes_dict[f'logical_pred_{d}'] = nodes_per_pred_depth[d]
    # ノード数が0のノードタイプをフィルター
    num_nodes_dict = {k: v for k, v in num_nodes_dict.items() if v > 0}
    return num_nodes_dict


def get_depths(plan_depths, predicate_depths):
    """深度を取得する"""
    max_depth = max(plan_depths)
    max_pred_depth = 0
    if len(predicate_depths) > 0:
        max_pred_depth = max(predicate_depths)
    return max_depth, max_pred_depth


def adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, logical_preds, plan_dict, pred_dict,
                          pred_node_type_id_func):
    """述語エッジを適応する"""
    # プランエッジに変換
    for u, v in filter_to_plan_edges:
        # プランノードを正しいIDと深度に変換
        v_node_id, d_v = plan_dict[v]
        # 述語ノードを正しいノードタイプとIDに変換
        node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)
        
        data_dict[(node_type, 'to_plan', f'plan{d_v}')].append((u_node_id, v_node_id))
    
    # 述語内エッジを変換（例：カラムからANDへ）
    for u, v in intra_predicate_edges:
        u_node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)
        v_node_type, v_node_id = pred_node_type_id_func(logical_preds, pred_dict, v)
        data_dict[(u_node_type, 'intra_predicate', v_node_type)].append((u_node_id, v_node_id))


def create_node_types_per_depth(plan_depths, plan_to_plan_edges):
    """深度ごとにノードタイプを作成する"""
    # 異種グラフを作成：ノードタイプ: table, column, filter_column, logical_pred, output_column, plan{depth}
    # まず、古いプランノードID -> 深度とノードIDのマッピングを作成
    plan_dict = dict()
    nodes_per_depth = collections.defaultdict(int)
    for plan_node, d in enumerate(plan_depths):
        plan_dict[plan_node] = (nodes_per_depth[d], d)
        nodes_per_depth[d] += 1
    
    # プランの深度に応じてエッジとノードタイプを作成
    data_dict = collections.defaultdict(list)
    for u, v in plan_to_plan_edges:
        u_node_id, d_u = plan_dict[u]
        v_node_id, d_v = plan_dict[v]
        assert d_v < d_u
        data_dict[(f'plan{d_u}', f'intra_plan', f'plan{d_v}')].append((u_node_id, v_node_id))
    
    return data_dict, nodes_per_depth, plan_dict


def add_numerical_scalers(feature_statistics):
    """数値スケーラーを追加する"""
    for k, v in feature_statistics.items():
        if v.get('type') == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = v['center']
            scaler.scale_ = v['scale']
            feature_statistics[k]['scaler'] = scaler


def pred_node_type_id(logical_preds, pred_dict, u):
    """述語ノードタイプIDを取得する"""
    # インデックスが範囲内かチェック
    if u >= len(logical_preds):
        # デフォルト値を使用（警告は表示しない）
        u_node_id = pred_dict.get(u, 0)
        node_type = f'filter_column'
        return node_type, u_node_id
    
    if logical_preds[u]:
        u_node_id, depth = pred_dict[u]
        node_type = f'logical_pred_{depth}'
    else:
        u_node_id = pred_dict[u]
        node_type = f'filter_column'
    return node_type, u_node_id