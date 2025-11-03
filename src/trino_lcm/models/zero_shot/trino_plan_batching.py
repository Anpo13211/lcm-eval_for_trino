import collections
import json
import os
from pathlib import Path
from typing import Dict, Optional

from cross_db_benchmark.benchmark_tools.trino.plan_operator import TrinoPlanOperator
import dgl
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from cross_db_benchmark.benchmark_tools.trino.parse_filter import PredicateNode
from cross_db_benchmark.benchmark_tools.generate_workload import LogicalOperator
from training.featurizations import Featurization
from training.preprocessing.feature_statistics import FeatureType


def load_database_statistics(
    catalog: str, 
    schema: str, 
    stats_dir: str = 'datasets_statistics',
    prefer_zero_shot: bool = True
) -> Dict[str, Dict]:
    """
    データベース統計情報を読み込む（zero-shot形式対応）
    
    Args:
        catalog: Trinoカタログ名（通常は'iceberg'）
        schema: スキーマ名（データセット名）
        stats_dir: 統計情報のルートディレクトリ
        prefer_zero_shot: Trueの場合、zero-shot形式を優先して読み込む
    
    Returns:
        Postgres形式（zero-shotモデル標準形式）の統計情報: {
            'table_stats': {table: SimpleNamespace},
            'column_stats': {(table, column): SimpleNamespace}
        }
    """
    from models.zeroshot.utils.statistics_loader import load_database_statistics_for_zeroshot
    
    # zero-shot形式から読み込んでPostgres形式に変換
    dataset = schema  # schema名がデータセット名と一致する場合
    postgres_format = load_database_statistics_for_zeroshot(
        dataset=dataset,
        prefer_zero_shot=prefer_zero_shot,
        stats_dir=stats_dir
    )
    
    # メタデータも読み込む（既存の互換性のため）
    schema_dir = Path(stats_dir) / f"{catalog}_{schema}"
    metadata_file = schema_dir / 'metadata.json'
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return {
        'table_stats': postgres_format['table_stats'],
        'column_stats': postgres_format['column_stats'],
        'metadata': metadata
    }


# encode()関数はPostgreSQL版の統一実装を使用
from models.zeroshot.postgres_plan_batching import encode


def plan_to_graph(node: TrinoPlanOperator, database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics, feature_statistics,
                  filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges, output_column_features,
                  column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                  output_column_idx, column_idx, table_idx, plan_featurization: Featurization, predicate_depths, intra_predicate_edges,
                  logical_preds, parent_node_id=None, depth=0, db_real_statistics=None):
    """
    Trinoプランをグラフに変換する
    
    この関数はPostgres版の`plan_to_graph()`と互換性を持つように設計されています。
    `db_statistics`はPostgres形式（{(table, column): SimpleNamespace}）を期待します。
    
    引数一覧：
    1. 入力データ
        node: 現在処理中のTrino実行計画ノード
        database_id: データベースの識別子
        db_statistics: データベースの統計情報（Postgres形式）
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
    
    # データベース統計を取得（Postgres形式を期待）
    # db_statistics[database_id].column_stats は {(table, column): SimpleNamespace} 形式
    db_stats = None
    if db_statistics is not None:
        if isinstance(db_statistics, dict):
            db_stats = db_statistics.get(database_id)
        else:
            db_stats = getattr(db_statistics, database_id, None)

    # Postgres形式: db_stats.column_stats は {(table, column): SimpleNamespace} 形式
    db_column_stats = getattr(db_stats, 'column_stats', None) if db_stats is not None else None
    
    # db_real_statisticsは旧形式の互換性のためのフォールバック（将来削除予定）

    # プランの特徴量を抽出（統一されたencode()を使用）
    # PostgreSQL版と同様のロジックで、plan_featurization.PLAN_FEATURESを使用
    plan_params = node.plan_parameters
    use_dict = isinstance(plan_params, dict)
    
    # ヘルパー関数: plan_parametersから値を取得（dict/SimpleNamespace両対応）
    def get_param(key, default=None):
        return plan_params.get(key, default) if use_dict else getattr(plan_params, key, default)
    
    # PostgreSQL版と同じ形式で特徴量を抽出
    # TrinoではVARIABLES['plan']の代わりにPLAN_FEATURESを使用
    plan_feat = []
    if hasattr(plan_featurization, 'PLAN_FEATURES'):
        # PostgreSQL形式
        for feat_name in plan_featurization.PLAN_FEATURES:
            enc_value = encode(feat_name, plan_params, feature_statistics)
            plan_feat.append(enc_value)
    elif hasattr(plan_featurization, 'VARIABLES') and 'plan' in plan_featurization.VARIABLES:
        # Trino形式（後方互換性）
        for feat_name in plan_featurization.VARIABLES['plan']:
            if hasattr(plan_params, feat_name) if not use_dict else feat_name in plan_params:
                enc_value = encode(feat_name, plan_params, feature_statistics)
                plan_feat.append(enc_value)
            else:
                plan_feat.append(0.0)
    else:
        raise ValueError(f"plan_featurizationにPLAN_FEATURESまたはVARIABLES['plan']が見つかりません")
    
    plan_features.append(plan_feat)
    
    # 親子関係のエッジを追加
    if parent_node_id is not None:
        plan_to_plan_edges.append((current_plan_id, parent_node_id))
    
    # テーブル情報の処理
    if hasattr(node.plan_parameters, 'table'):
        table_name = getattr(node.plan_parameters, 'table')
        
        # テーブルのインデックスを取得または作成
        if table_name not in table_idx:
            table_idx[table_name] = len(table_features)
            
            # データベース統計情報からテーブル統計を取得（Postgres形式）
            table_stats_dict = {}
            # Postgres形式: db_stats.table_stats は {table: SimpleNamespace} 形式
            if db_stats is not None:
                db_table_stats = getattr(db_stats, 'table_stats', None)
                if db_table_stats is not None and table_name in db_table_stats:
                    # Postgres形式から取得（SimpleNamespace）
                    table_stat = db_table_stats[table_name]
                    table_stats_dict = {
                        'reltuples': table_stat.reltuples if hasattr(table_stat, 'reltuples') else 0,
                        'relpages': table_stat.relpages if hasattr(table_stat, 'relpages') else 0,
                    }
            
            # フォールバック: 旧形式（db_real_statistics）の互換性
            if not table_stats_dict and db_real_statistics and 'table_stats' in db_real_statistics:
                if table_name in db_real_statistics['table_stats']:
                    table_stats = db_real_statistics['table_stats'][table_name]
                    table_stats_dict = {
                        'reltuples': table_stats.get('row_count', 0),
                        'relpages': 0,
                    }
            
            # テーブルの特徴量を抽出
            table_feat = []
            for feat_name in plan_featurization.VARIABLES['table']:
                if feat_name in table_stats_dict:
                    # 統計情報から取得
                    value = table_stats_dict[feat_name]
                    enc_value = encode(feat_name, table_stats_dict, feature_statistics)
                    table_feat.append(enc_value)
                elif hasattr(node.plan_parameters, feat_name):
                    # プランパラメータから取得（SimpleNamespace対応）
                    value = getattr(node.plan_parameters, feat_name)
                    enc_value = encode(feat_name, node.plan_parameters, feature_statistics)
                    table_feat.append(enc_value)
                else:
                    # 利用不可の場合は0
                    table_feat.append(0.0)
            
            table_features.append(table_feat)
        
        # テーブルからプランへのエッジを追加
        table_to_plan_edges.append((table_idx[table_name], current_plan_id))
    
    
    # 出力カラム情報の処理
    if hasattr(node.plan_parameters, 'output_columns'):
        output_columns = getattr(node.plan_parameters, 'output_columns')
        for output_col in output_columns:
            output_col_key = (
                output_col.get('aggregation'),
                tuple(output_col.get('columns', [])),
                database_id,
            )

            # 出力カラムのインデックスを取得または作成
            output_column_node_id = output_column_idx.get(output_col_key)
            if output_column_node_id is None:
                output_col_feat = []
                for feat_name in plan_featurization.VARIABLES['output_column']:
                    output_col_feat.append(
                        encode_or_zero(feat_name, output_col, feature_statistics)
                    )

                output_column_node_id = len(output_column_features)
                output_column_features.append(output_col_feat)
                output_column_idx[output_col_key] = output_column_node_id

            # 出力カラムからプランへのエッジを追加
            output_column_to_plan_edges.append((output_column_node_id, current_plan_id))

            # 出力カラムに関連付いた元カラムとのエッジを構築
            for column in output_col.get('columns', []):
                if isinstance(column, list):
                    column = tuple(column)
                column_key = (column, database_id)
                column_node_id = column_idx.get(column_key)

                if column_node_id is None:
                    column_stats = lookup_stats(db_column_stats, column, db_real_statistics)
                    column_params = as_dict(column_stats)

                    column_feat = []
                    for feat_name in plan_featurization.VARIABLES['column']:
                        column_feat.append(
                            encode_or_zero(feat_name, column_params, feature_statistics)
                        )

                    column_node_id = len(column_features)
                    column_features.append(column_feat)
                    column_idx[column_key] = column_node_id

                column_to_output_column_edges.append((column_node_id, output_column_node_id))
    
    # フィルター情報の処理
    filter_column = getattr(node.plan_parameters, 'filter_columns', None)
    if filter_column:
        filter_column_dict = predicate_to_dict(filter_column)
        parse_predicates(
            db_column_stats,
            feature_statistics,
            filter_column_dict,
            filter_to_plan_edges,
            plan_featurization,
            predicate_col_features,
            predicate_depths,
            intra_predicate_edges,
            logical_preds,
            plan_node_id=current_plan_id,
            db_real_statistics=db_real_statistics,
        )
    
    # 子ノードを再帰的に処理
    for child in node.children:
        plan_to_graph(child, database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics,
                      feature_statistics, filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges,
                      output_column_features, column_to_output_column_edges, column_features, table_features,
                      table_to_plan_edges, output_column_idx, column_idx, table_idx,
                      plan_featurization, predicate_depths, intra_predicate_edges, logical_preds,
                      parent_node_id=current_plan_id, depth=depth + 1, db_real_statistics=db_real_statistics)


def encode_or_zero(feature_name, params, feature_statistics):
    """指定した特徴量をエンコードする（欠損時は0を返す）"""
    if feature_statistics is None or feature_name not in feature_statistics:
        return 0.0

    if params is None:
        return 0.0

    # SimpleNamespace統一により、encode()はdict/SimpleNamespace両対応
    # PredicateNodeやSimpleNamespaceの場合は直接渡す（encode()内で処理される）
    use_dict = isinstance(params, dict) or (isinstance(params, PredicateNode) and hasattr(params, '__dict__'))
    
    # 値の存在確認（dict/SimpleNamespace両対応）
    has_value = False
    if isinstance(params, dict):
        has_value = feature_name in params
    elif isinstance(params, PredicateNode):
        # PredicateNodeはSimpleNamespaceのような属性アクセス
        has_value = hasattr(params, feature_name)
    else:
        has_value = hasattr(params, feature_name)
    
    if not has_value:
        return 0.0

    # aggregation特徴量の特別処理
    if feature_name == 'aggregation':
        # 値を取得（dict/SimpleNamespace両対応）
        if isinstance(params, dict):
            agg_value = params[feature_name]
        else:
            agg_value = getattr(params, feature_name, None)
        
        if agg_value:
            # 集約関数を数値にエンコード
            agg_encoding = {
                'Aggregator.COUNT': 0,
                'Aggregator.SUM': 1,
                'Aggregator.AVG': 2,
                'Aggregator.MIN': 3,
                'Aggregator.MAX': 4,
                None: 5  # 集約なし
            }
            enc_value = agg_encoding.get(agg_value, 5)
            return float(enc_value)
        else:
            return 5.0  # 集約なし

    try:
        # 統一されたencode()を使用（dict/SimpleNamespace両対応）
        return encode(feature_name, params, feature_statistics)
    except (KeyError, ValueError, AttributeError):
        return 0.0


def lookup_stats(stats_container, key, db_statistics=None):
    """
    統計情報を取得するユーティリティ
    
    Args:
        stats_container: Postgres形式の統計情報ソース（{(table, column): SimpleNamespace}）
        key: 検索キー（通常はカラム名のタプル (table, column)）
        db_statistics: データベース統計情報（旧形式の互換性のためのフォールバック）
    
    Returns:
        統計情報オブジェクト（SimpleNamespace）またはNone
    """
    # Postgres形式の統計情報から検索（優先）
    if stats_container is not None and key is not None:
        # stats_containerは {(table, column): SimpleNamespace} 形式
        if isinstance(stats_container, dict):
            result = stats_container.get(key)
            if result is not None:
                return result
        
        # SimpleNamespaceの場合はgetメソッドが使えないので、dictとして扱う
        if hasattr(stats_container, 'get'):
            try:
                result = stats_container.get(key)
                if result is not None:
                    return result
            except Exception:
                pass
    
    # フォールバック: 旧形式（db_statistics）の互換性
    # この部分は将来削除される可能性があります
    if db_statistics and 'column_stats' in db_statistics:
        # keyがタプルの場合、table.column形式に変換
        if isinstance(key, tuple) and len(key) >= 1:
            if len(key) == 2:
                lookup_key = f"{key[0]}.{key[1]}"
            elif len(key) == 1:
                # カラム名のみの場合、全テーブルから検索
                column_name = key[0]
                for full_col_name, col_stats in db_statistics['column_stats'].items():
                    if col_stats.get('column') == column_name:
                        return type('ColumnStats', (), col_stats)
                return None
            else:
                lookup_key = '.'.join(str(k) for k in key)
            
            if lookup_key in db_statistics['column_stats']:
                col_stats = db_statistics['column_stats'][lookup_key]
                return type('ColumnStats', (), col_stats)
    
    return None


def as_dict(obj):
    """オブジェクトを辞書に変換する"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return vars(obj)


def predicate_to_dict(predicate):
    """述語ノードを辞書形式に正規化する"""
    if isinstance(predicate, PredicateNode):
        return predicate.to_dict()
    return predicate


def parse_predicates(db_column_features, feature_statistics, filter_column, filter_to_plan_edges,
                     plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                     logical_preds, plan_node_id=None, parent_filter_node_id=None, depth=0, db_real_statistics=None):
    """述語ツリーを再帰的に解析して特徴量とエッジを構築する"""
    if filter_column is None:
        return

    filter_params = as_dict(filter_column)
    filter_node_id = len(predicate_depths)
    predicate_depths.append(depth)

    operator_value = filter_params.get('operator')
    logical_operators = {str(op) for op in list(LogicalOperator)}
    is_logical = operator_value in logical_operators
    logical_preds.append(is_logical)

    curr_filter_features = []
    for feature_name in plan_featurization.FILTER_FEATURES:
        curr_filter_features.append(encode_or_zero(feature_name, filter_params, feature_statistics))

    if not is_logical:
        column_id = filter_params.get('column')
        column_stats = lookup_stats(db_column_features, column_id, db_real_statistics)
        column_params = as_dict(column_stats)
        for feature_name in plan_featurization.COLUMN_FEATURES:
            curr_filter_features.append(encode_or_zero(feature_name, column_params, feature_statistics))

    predicate_col_features.append(curr_filter_features)

    if depth == 0:
        assert plan_node_id is not None
        filter_to_plan_edges.append((filter_node_id, plan_node_id))
    else:
        assert parent_filter_node_id is not None
        intra_predicate_edges.append((filter_node_id, parent_filter_node_id))

    for child in filter_params.get('children', []):
        child_dict = predicate_to_dict(child)
        parse_predicates(db_column_features, feature_statistics, child_dict, filter_to_plan_edges,
                         plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                         logical_preds, plan_node_id=plan_node_id, parent_filter_node_id=filter_node_id,
                         depth=depth + 1, db_real_statistics=db_real_statistics)


def trino_plan_collator(plans, feature_statistics: dict = None, db_statistics: dict = None,
                        plan_featurization: Featurization = None):
    """
    Trinoの物理プランを大きなグラフに結合し、MLモデルに投入できる形式にする
    
    Args:
        plans: Trinoプランのリスト
        feature_statistics: 特徴量の統計情報
        db_statistics: データベースの統計情報（Postgres形式）
            - db_statistics[database_id].column_stats: {(table, column): SimpleNamespace}
            - db_statistics[database_id].table_stats: {table: SimpleNamespace}
        plan_featurization: プランの特徴量化設定
    
    Returns:
        graph: DGLの異種グラフ
        features: ノード特徴量の辞書
        labels: ラベル（実行時間）のリスト
        sample_idxs: サンプルインデックスのリスト
    
    Note:
        Postgres版の`postgres_plan_collator()`と互換性を持つように設計されています。
        統計情報はPostgres形式（{(table, column): SimpleNamespace}）を期待します。
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
        
        # db_statisticsが辞書形式で、database_idをキーとして持つ場合の処理
        # db_statistics[database_id]がSimpleNamespaceの場合、そのまま使用
        # db_statisticsがNoneや空の場合は空のdictとして扱う
        db_stats_for_plan = db_statistics if db_statistics else {}
        
        plan_to_graph(p, p.database_id, plan_depths, plan_features, plan_to_plan_edges, db_stats_for_plan,
                      feature_statistics, filter_to_plan_edges, filter_features, output_column_to_plan_edges,
                      output_column_features, column_to_output_column_edges, column_features, table_features,
                      table_to_plan_edges, output_column_idx, column_idx, table_idx,
                      plan_featurization, predicate_depths, intra_predicate_edges, logical_preds, db_real_statistics=None)
    
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