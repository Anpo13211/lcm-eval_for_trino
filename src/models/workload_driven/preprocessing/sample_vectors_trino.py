"""
Trino用のsample_vec生成機能

PostgreSQLの実装を参考に、Trino用のsample_vec生成機能を実装します。
"""

import json
import os
from typing import Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.parse_run import dumper
from cross_db_benchmark.benchmark_tools.utils import load_json, load_schema_json
from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator
from cross_db_benchmark.datasets.datasets import database_list


def construct_filter_sample(table_samples: Dict[str, pd.DataFrame], 
                            col_stats: list,
                            filter_column: Any) -> pd.Series:
    """
    フィルタ条件に基づいてサンプルベクトルを構築
    
    Args:
        table_samples: テーブル名をキーとするDataFrameの辞書
        col_stats: カラム統計情報のリスト
        filter_column: フィルタ条件（PredicateNode形式）
    
    Returns:
        pd.Series: フィルタ条件にマッチするかどうかのブール値のSeries
    """
    sample_vec = None
    
    # リーフノード（演算子がLogicalOperatorでない場合）
    if filter_column.operator not in {LogicalOperator.AND, LogicalOperator.OR}:
        # カラム情報を取得
        column_id = filter_column.column
        
        # column_idがtupleの場合は、最初の要素を使用（テーブル名, カラム名の形式）
        if isinstance(column_id, tuple):
            if len(column_id) == 2:
                table_name, col_name = column_id
            elif len(column_id) == 1:
                col_name = column_id[0]
                # テーブル名はcol_statsから推測する必要がある
                table_name = None
            else:
                return pd.Series([False] * 1000)
        elif column_id is None:
            # カラムがない場合は空のベクトルを返す
            return pd.Series([False] * 1000)
        elif isinstance(column_id, int):
            # カラムIDとして扱う
            if column_id < len(col_stats):
                col_stat = col_stats[column_id]
                col_name = col_stat.attname
                table_name = col_stat.tablename
            else:
                return pd.Series([False] * 1000)
        else:
            # その他の形式はサポートしない
            return pd.Series([False] * 1000)
        
        if table_name not in table_samples:
            # テーブルサンプルがない場合は空のベクトルを返す
            return pd.Series([False] * 1000)
        
        base_sample = table_samples[table_name][col_name]
        
        # 演算子に応じてフィルタリング
        if filter_column.operator == Operator.EQ:
            sample_vec = base_sample == filter_column.literal
        elif filter_column.operator == Operator.GEQ:
            sample_vec = base_sample >= filter_column.literal
        elif filter_column.operator == Operator.LEQ:
            sample_vec = base_sample <= filter_column.literal
        elif filter_column.operator == Operator.NEQ:
            sample_vec = base_sample != filter_column.literal
        elif filter_column.operator in {Operator.LIKE, Operator.NOT_LIKE}:
            # LIKEパターンの処理
            regex = str(filter_column.literal)
            # エスケープ文字を処理
            for esc_char in ['.', '(', ')', '?', '[', ']']:
                regex = regex.replace(esc_char, f'\\{esc_char}')
            regex = regex.replace('%', '.*?')
            regex = f'^{regex}$'
            sample_vec = base_sample.astype(str).str.contains(regex, regex=True, na=False)
            if filter_column.operator == Operator.NOT_LIKE:
                sample_vec = ~sample_vec
        elif filter_column.operator == Operator.IN:
            if isinstance(filter_column.literal, list):
                sample_vec = base_sample.isin(filter_column.literal)
            else:
                sample_vec = pd.Series([False] * len(base_sample))
        elif filter_column.operator == Operator.NOT_IN:
            if isinstance(filter_column.literal, list):
                sample_vec = ~base_sample.isin(filter_column.literal)
            else:
                sample_vec = pd.Series([True] * len(base_sample))
        elif filter_column.operator == Operator.IS_NOT_NULL:
            sample_vec = ~base_sample.isna()
        elif filter_column.operator == Operator.IS_NULL:
            sample_vec = base_sample.isna()
        elif filter_column.operator == Operator.BETWEEN:
            if isinstance(filter_column.literal, (list, tuple)) and len(filter_column.literal) == 2:
                sample_vec = (base_sample >= filter_column.literal[0]) & (base_sample <= filter_column.literal[1])
            else:
                sample_vec = pd.Series([False] * len(base_sample))
        else:
            print(f"Warning: Unsupported operator {filter_column.operator}")
            sample_vec = pd.Series([False] * len(base_sample))
        
        # NaNの処理
        if sample_vec is not None:
            sample_vec[sample_vec.isna()] = False
    
    # 論理演算子（AND/OR）
    elif filter_column.operator in {LogicalOperator.AND, LogicalOperator.OR}:
        if not filter_column.children:
            sample_vec = pd.Series([False] * 1000)
        else:
            children_vec = [construct_filter_sample(table_samples, col_stats, c) 
                          for c in filter_column.children]
            sample_vec = children_vec[0]
            for c_vec in children_vec[1:]:
                if filter_column.operator == LogicalOperator.AND:
                    sample_vec &= c_vec
                elif filter_column.operator == LogicalOperator.OR:
                    sample_vec |= c_vec  # ORは|を使う（^はXOR）
    
    if sample_vec is None:
        sample_vec = pd.Series([False] * 1000)
    
    return sample_vec


def construct_filter_sample_from_dict(table_samples: Dict[str, pd.DataFrame],
                                      col_stats: list,
                                      filter_dict: dict) -> pd.Series:
    """
    辞書形式のfilter_columnsからsample_vecを生成
    
    Args:
        table_samples: テーブル名をキーとするDataFrameの辞書
        col_stats: カラム統計情報のリスト
        filter_dict: 辞書形式のfilter_columns
    
    Returns:
        pd.Series: フィルタ条件にマッチするかどうかのブール値のSeries
    """
    operator = filter_dict.get('operator')
    column = filter_dict.get('column')
    literal = filter_dict.get('literal')
    children = filter_dict.get('children', [])
    
    # 論理演算子の場合
    # operator は to_dict() で文字列化されているので、文字列として比較
    if operator == 'AND':
        if not children:
            return pd.Series([False] * 1000)
        
        children_vec = [construct_filter_sample_from_dict(table_samples, col_stats, c) 
                        for c in children]
        sample_vec = children_vec[0]
        for c_vec in children_vec[1:]:
            sample_vec &= c_vec
        
        return sample_vec
    elif operator == 'OR':
        if not children:
            return pd.Series([False] * 1000)
        
        children_vec = [construct_filter_sample_from_dict(table_samples, col_stats, c) 
                        for c in children]
        sample_vec = children_vec[0]
        for c_vec in children_vec[1:]:
            sample_vec |= c_vec
        
        return sample_vec
    
    # リーフノードの場合
    if column is None:
        return pd.Series([False] * 1000)
    
    # カラム情報を取得
    # columnが整数（カラムID）の場合
    if isinstance(column, int):
        if column < len(col_stats):
            col_stat = col_stats[column]
            col_name = col_stat.attname
            table_name = col_stat.tablename
        else:
            return pd.Series([False] * 1000)
    # columnがtupleの場合
    elif isinstance(column, tuple):
        if len(column) == 2:
            table_name, col_name = column
            # 文字列をクリーンアップ
            table_name = str(table_name).strip('"')
            col_name = str(col_name).strip('"')
        elif len(column) == 1:
            col_name = str(column[0]).strip('"')
            # テーブル名はcol_statsから推測する必要がある
            table_name = None
            for col_stat in col_stats:
                if col_stat.attname == col_name:
                    table_name = col_stat.tablename
                    break
            
            # カラム名が見つからない場合、Trinoのエイリアス（_数字のサフィックス）を削除して再試行
            if table_name is None and '_' in col_name:
                # 末尾の_数字を削除して試す
                import re
                col_name_without_suffix = re.sub(r'_\d+$', '', col_name)
                for col_stat in col_stats:
                    if col_stat.attname == col_name_without_suffix:
                        table_name = col_stat.tablename
                        col_name = col_name_without_suffix  # カラム名を更新
                        break
            
            if table_name is None:
                return pd.Series([False] * 1000)
        else:
            return pd.Series([False] * 1000)
    else:
        return pd.Series([False] * 1000)
    
    if table_name not in table_samples:
        return pd.Series([False] * 1000)
    
    if col_name not in table_samples[table_name].columns:
        return pd.Series([False] * 1000)
    
    base_sample = table_samples[table_name][col_name]
    
    # literalの値を抽出（varchar 'B' -> 'B'、bigint '343' -> '343'のような形式に対応）
    literal_value = literal
    if isinstance(literal, str):
        # varchar 'B' -> 'B'、bigint '343' -> '343'のような形式から値を抽出
        if " '" in literal:
            parts = literal.split("'")
            if len(parts) >= 2:
                literal_value = parts[1]
        elif ' "' in literal:
            parts = literal.split('"')
            if len(parts) >= 2:
                literal_value = parts[1]
        
        # 数値文字列の場合は適切な型に変換
        try:
            if '.' in str(literal_value):
                literal_value = float(literal_value)
            else:
                literal_value = int(literal_value)
        except (ValueError, TypeError):
            # 変換できない場合は文字列のまま
            pass
    
    # 演算子に応じてフィルタリング
    # operator は to_dict() で文字列化されているので、文字列として比較
    sample_vec = None
    if operator == '=' or operator == '==':
        sample_vec = base_sample == literal_value
    elif operator in {'>=', '>'}:
        sample_vec = base_sample >= literal_value
    elif operator in {'<=', '<'}:
        sample_vec = base_sample <= literal_value
    elif operator in {'!=', '<>'}:
        sample_vec = base_sample != literal_value
    elif operator in {'LIKE', 'NOT LIKE'}:
        if literal_value:
            # ワイルドカードを正規表現に変換
            regex = str(literal_value)
            for esc_char in ['.', '(', ')', '?']:
                regex = regex.replace(esc_char, f'\\{esc_char}')
            regex = regex.replace('%', '.*?')
            regex = f'^{regex}$'
            sample_vec = base_sample.astype(str).str.contains(regex, na=False)
            if operator == 'NOT LIKE':
                sample_vec = ~sample_vec
        else:
            sample_vec = pd.Series([False] * len(base_sample))
    elif operator == 'IN':
        if isinstance(literal_value, list):
            sample_vec = base_sample.isin(literal_value)
        else:
            sample_vec = pd.Series([False] * len(base_sample))
    elif operator == 'IS NOT NULL':
        sample_vec = ~base_sample.isna()
    elif operator == 'IS NULL':
        sample_vec = base_sample.isna()
    else:
        # デバッグ用: 未対応の演算子を出力（最初の数回のみ）
        if not hasattr(construct_filter_sample_from_dict, '_unsupported_op_count'):
            construct_filter_sample_from_dict._unsupported_op_count = 0
        if construct_filter_sample_from_dict._unsupported_op_count < 5:
            print(f"Warning: Unsupported operator in construct_filter_sample_from_dict: '{operator}' (type: {type(operator)})")
            construct_filter_sample_from_dict._unsupported_op_count += 1
        return pd.Series([False] * 1000)
    
    # NaNの処理
    if sample_vec is not None:
        sample_vec[sample_vec.isna()] = False
    else:
        sample_vec = pd.Series([False] * 1000)
    
    return sample_vec


def augment_sample(table_samples: Dict[str, pd.DataFrame], 
                   col_stats: list,
                   p_node: Any):
    """
    プランノードにsample_vecを追加
    
    Args:
        table_samples: テーブル名をキーとするDataFrameの辞書
        col_stats: カラム統計情報のリスト
        p_node: プランノード（TrinoPlanOperator）
    """
    filter_columns = None
    if hasattr(p_node, 'plan_parameters'):
        # plan_parametersがSimpleNamespaceか辞書か確認
        if hasattr(p_node.plan_parameters, '__dict__'):
            plan_params = vars(p_node.plan_parameters)
        elif isinstance(p_node.plan_parameters, dict):
            plan_params = p_node.plan_parameters
        else:
            plan_params = {}
        
        filter_columns = plan_params.get('filter_columns')
    
    if filter_columns is not None:
        try:
            if isinstance(filter_columns, dict):
                # 辞書形式の場合
                sample_vec_series = construct_filter_sample_from_dict(table_samples, col_stats, filter_columns)
                # バイナリベクトルに変換
                sample_vec = [1 if s_i else 0 for s_i in sample_vec_series]
                
                # 1000次元に調整
                if len(sample_vec) < 1000:
                    sample_vec = sample_vec + [0] * (1000 - len(sample_vec))
                elif len(sample_vec) > 1000:
                    sample_vec = sample_vec[:1000]
            else:
                # PredicateNode形式の場合
                sample_vec_series = construct_filter_sample(table_samples, col_stats, filter_columns)
                # バイナリベクトルに変換
                sample_vec = [1 if s_i else 0 for s_i in sample_vec_series]
                
                # 1000次元に調整
                if len(sample_vec) < 1000:
                    sample_vec = sample_vec + [0] * (1000 - len(sample_vec))
                elif len(sample_vec) > 1000:
                    sample_vec = sample_vec[:1000]
            
            # SimpleNamespaceまたは辞書に設定
            if hasattr(p_node.plan_parameters, '__dict__'):
                setattr(p_node.plan_parameters, 'sample_vec', sample_vec)
            elif isinstance(p_node.plan_parameters, dict):
                p_node.plan_parameters['sample_vec'] = sample_vec
            else:
                # 新しく辞書として設定
                p_node.plan_parameters = {'sample_vec': sample_vec}
                
        except Exception as e:
            op_name = 'Unknown'
            if hasattr(p_node, 'plan_parameters'):
                if hasattr(p_node.plan_parameters, '__dict__'):
                    op_name = vars(p_node.plan_parameters).get('op_name', 'Unknown')
                elif isinstance(p_node.plan_parameters, dict):
                    op_name = p_node.plan_parameters.get('op_name', 'Unknown')
            
            # エラー出力（最初の数回のみ）
            if not hasattr(augment_sample, '_error_count'):
                augment_sample._error_count = 0
            if augment_sample._error_count < 3:
                print(f"Warning: Failed to generate sample_vec for node {op_name}: {e}")
                import traceback
                traceback.print_exc()
                augment_sample._error_count += 1
            
            # エラー時も空のsample_vecを設定
            if hasattr(p_node.plan_parameters, '__dict__'):
                setattr(p_node.plan_parameters, 'sample_vec', [0] * 1000)
            elif isinstance(p_node.plan_parameters, dict):
                p_node.plan_parameters['sample_vec'] = [0] * 1000
    
    # 子ノードを再帰的に処理
    if hasattr(p_node, 'children') and p_node.children:
        for c in p_node.children:
            augment_sample(table_samples, col_stats, c)


def get_table_samples_from_trino(dataset: str, 
                                  database_connection: Any,
                                  no_samples: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    Trinoデータベースからテーブルサンプルを取得
    
    Args:
        dataset: データセット名
        database_connection: TrinoDatabaseConnectionインスタンス
        no_samples: 取得するサンプル数
    
    Returns:
        Dict[str, pd.DataFrame]: テーブル名をキーとするDataFrameの辞書
    """
    schema = load_schema_json(dataset, prefer_zero_shot=True)
    table_samples = {}
    
    for table_name in schema.tables:
        try:
            # Trinoからサンプルを取得
            sql = f"SELECT * FROM {table_name} TABLESAMPLE SYSTEM (10) LIMIT {no_samples}"
            # または: sql = f"SELECT * FROM {table_name} LIMIT {no_samples}"
            
            result = database_connection.submit_query(sql)
            
            if result:
                # カラム名を取得
                conn = database_connection._get_connection()
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
                column_names = [desc[0] for desc in cursor.description]
                
                # DataFrameを作成
                df = pd.DataFrame(result, columns=column_names)
                table_samples[table_name] = df
                print(f"Loaded {len(df)} samples from {table_name}")
            else:
                print(f"Warning: No data found for table {table_name}")
                table_samples[table_name] = pd.DataFrame()
                
        except Exception as e:
            print(f"Warning: Failed to load samples from {table_name}: {e}")
            table_samples[table_name] = pd.DataFrame()
    
    return table_samples


def get_table_samples_from_csv(dataset: str,
                               data_dir: Optional[str] = None,
                               no_samples: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    CSVファイルからテーブルサンプルを取得（PostgreSQL互換）
    
    Args:
        dataset: データセット名（例: 'accidents', 'scaled_financial'）
        data_dir: データディレクトリ（Noneの場合はデフォルトパスを使用）
        no_samples: 取得するサンプル数
    
    Returns:
        Dict[str, pd.DataFrame]: テーブル名をキーとするDataFrameの辞書
    """
    # Trino用のschema.json読み込みを使用（zero-shot_datasets配下を優先）
    try:
        schema = load_schema_json(dataset, prefer_zero_shot=True)
    except (FileNotFoundError, AssertionError):
        # Fallback: 旧形式のパス
        fallback_path = os.path.join('/Users/an/query_engine/lakehouse/zero-shot_datasets', dataset, 'schema.json')
        if not os.path.exists(fallback_path):
            raise
        with open(fallback_path, 'r') as f:
            s = json.load(f)
        from types import SimpleNamespace
        schema = SimpleNamespace(
            name=s.get('name', dataset),
            tables=s.get('tables', []),
            csv_kwargs=s.get('csv_kwargs', {})
        )
    table_samples = {}
    
    # データディレクトリが指定されていない場合は、デフォルトパスを使用
    if data_dir is None:
        # デフォルトのベースパス: /Users/an/query_engine/lakehouse/zero-shot_datasets/
        default_base_path = '/Users/an/query_engine/lakehouse/zero-shot_datasets'
        
        # imdb_fullの場合はimdbフォルダを使用（imdb_fullフォルダは存在しない）
        if dataset == 'imdb_full':
            dataset_dir = 'imdb'
        # scaled_*で始まるデータセット名の場合はそのまま使用
        elif dataset.startswith('scaled_'):
            dataset_dir = dataset
        else:
            # データセット名に対応するDatabaseオブジェクトを検索
            dataset_dir = dataset
            for db in database_list:
                if db.db_name == dataset:
                    # データセット名が見つかった場合、data_folderを使用
                    dataset_dir = db.data_folder
                    break
        
        data_dir = os.path.join(default_base_path, dataset_dir)
    
    print(f"Loading table samples from: {data_dir}")
    
    for table_name in schema.tables:
        table_path = os.path.join(data_dir, f'{table_name}.csv')
        if os.path.exists(table_path):
            try:
                df_sample = pd.read_csv(table_path, **vars(schema.csv_kwargs))
                if len(df_sample) > no_samples:
                    df_sample = df_sample.sample(random_state=0, n=no_samples)
                table_samples[table_name] = df_sample
                print(f"Loaded {len(df_sample)} samples from {table_path}")
            except Exception as e:
                print(f"Warning: Failed to load CSV from {table_path}: {e}")
                table_samples[table_name] = pd.DataFrame()
        else:
            print(f"Warning: CSV file not found: {table_path}")
            table_samples[table_name] = pd.DataFrame()
    
    return table_samples


def augment_sample_vectors_trino(dataset: str,
                                 plan_path: str,
                                 target_path: str,
                                 data_dir: Optional[str] = None,
                                 database_connection: Optional[Any] = None,
                                 no_samples: int = 1000):
    """
    Trinoプランにsample_vecを追加
    
    Args:
        dataset: データセット名（例: 'accidents', 'scaled_financial'）
        plan_path: パース済みプランファイルのパス
        target_path: 出力先パス
        data_dir: CSVデータディレクトリ（Noneの場合はデフォルトパスから取得）
        database_connection: TrinoDatabaseConnectionインスタンス（data_dirがNoneでデータベースから取得する場合に必要）
        no_samples: サンプル数
    """
    print("Augment Sample Vectors for Trino")
    
    if os.path.exists(target_path):
        print(f'Skip for {target_path}')
        return
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # プランファイルを読み込み
    run = load_json(plan_path)
    col_stats = run.database_stats.column_stats
    
    # テーブルサンプルを取得（CSVを優先、なければデータベースから）
    if database_connection:
        # データベース接続が提供されている場合はデータベースから取得を試みる
        try:
            table_samples = get_table_samples_from_trino(dataset, database_connection, no_samples)
        except Exception as e:
            print(f"Warning: Failed to load samples from database, trying CSV: {e}")
            table_samples = get_table_samples_from_csv(dataset, data_dir, no_samples)
    else:
        # CSVから取得（data_dirがNoneの場合はデフォルトパスを使用）
        table_samples = get_table_samples_from_csv(dataset, data_dir, no_samples)
    
    # 各プランにsample_vecを追加
    for p in tqdm(run.parsed_plans):
        augment_sample(table_samples, col_stats, p)
    
    # dumper関数を拡張してplan_runtimeとjoin_condsを保存
    def enhanced_dumper(obj):
        """plan_runtimeとjoin_condsを含むdumper"""
        try:
            return obj.toJSON()
        except:
            result = obj.__dict__.copy() if hasattr(obj, '__dict__') else obj
            # TrinoPlanOperatorの場合、plan_runtimeとjoin_condsを追加
            if hasattr(obj, 'plan_runtime'):
                result['plan_runtime'] = obj.plan_runtime
            if hasattr(obj, 'join_conds'):
                result['join_conds'] = obj.join_conds
            if hasattr(obj, 'database_id'):
                result['database_id'] = obj.database_id
            return result
    
    # 結果を保存
    with open(target_path, 'w') as outfile:
        json.dump(run, outfile, default=enhanced_dumper)
    
    print(f"Saved augmented plans to {target_path}")

