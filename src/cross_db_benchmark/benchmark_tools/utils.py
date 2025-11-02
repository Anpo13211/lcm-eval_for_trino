import json
import os
from types import SimpleNamespace


def load_schema_json(dataset, prefer_zero_shot=False):
    """
    スキーマJSONファイルを読み込む
    
    Args:
        dataset: データセット名
        prefer_zero_shot: Trueの場合、zero-shot_datasets配下を優先的に探す
    
    Returns:
        スキーマ情報（SimpleNamespace形式）
    """
    # パスの候補リスト
    paths = []
    # Docker環境とローカル環境の両方に対応
    zero_shot_base = os.getenv('ZERO_SHOT_DATASETS_DIR', '/Users/an/query_engine/lakehouse/zero-shot_datasets')
    
    if prefer_zero_shot:
        # zero-shot_datasets配下を優先
        paths.append(os.path.join(zero_shot_base, dataset, 'schema.json'))
        paths.append(os.path.join('cross_db_benchmark/datasets/', dataset, 'schema.json'))
    else:
        # 従来のパスを優先（後方互換性）
        paths.append(os.path.join('cross_db_benchmark/datasets/', dataset, 'schema.json'))
        paths.append(os.path.join(zero_shot_base, dataset, 'schema.json'))
    
    # 存在するパスを探す
    for schema_path in paths:
        if os.path.exists(schema_path):
            return load_json(schema_path)
    
    # 見つからない場合はエラー
    raise FileNotFoundError(f"Could not find schema.json. Tried:\n" + "\n".join(f"  - {p}" for p in paths))


def load_column_statistics(dataset, namespace=True, prefer_zero_shot=False):
    """
    カラム統計情報JSONファイルを読み込む
    
    Args:
        dataset: データセット名
        namespace: Trueの場合、SimpleNamespace形式で返す
        prefer_zero_shot: Trueの場合、zero-shot_datasets配下を優先的に探す
    
    Returns:
        カラム統計情報（namespace=Trueの場合はSimpleNamespace形式、Falseの場合はdict形式）
    """
    # パスの候補リスト
    paths = []
    # Docker環境とローカル環境の両方に対応
    zero_shot_base = os.getenv('ZERO_SHOT_DATASETS_DIR', '/Users/an/query_engine/lakehouse/zero-shot_datasets')
    
    if prefer_zero_shot:
        # zero-shot_datasets配下を優先
        paths.append(os.path.join(zero_shot_base, dataset, 'column_statistics.json'))
        paths.append(os.path.join('cross_db_benchmark/datasets/', dataset, 'column_statistics.json'))
    else:
        # 従来のパスを優先（後方互換性）
        paths.append(os.path.join('cross_db_benchmark/datasets/', dataset, 'column_statistics.json'))
        paths.append(os.path.join(zero_shot_base, dataset, 'column_statistics.json'))
    
    # 存在するパスを探す
    for path in paths:
        if os.path.exists(path):
            return load_json(path, namespace=namespace)
    
    # 見つからない場合はエラー
    raise FileNotFoundError(f"Could not find column_statistics.json. Tried:\n" + "\n".join(f"  - {p}" for p in paths))


def load_schema_json_trino(dataset):
    """
    Trino用のschema.json読み込み関数（後方互換性のため残す）
    zero-shot_datasets配下を優先的に探す
    """
    return load_schema_json(dataset, prefer_zero_shot=True)


def load_column_statistics_trino(dataset, namespace=True):
    """
    Trino用のcolumn_statistics.json読み込み関数（後方互換性のため残す）
    zero-shot_datasets配下を優先的に探す
    """
    return load_column_statistics(dataset, namespace=namespace, prefer_zero_shot=True)


def load_string_statistics(dataset, namespace=True):
    path = os.path.join('cross_db_benchmark/datasets/', dataset, 'string_statistics.json')
    assert os.path.exists(path), f"Could not find file ({path})"
    return load_json(path, namespace=namespace)


def load_json(path, namespace=True):
    with open(path) as json_file:
        if namespace:
            json_obj = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
        else:
            json_obj = json.load(json_file)
    return json_obj


def load_schema_sql(dataset, sql_filename):
    sql_path = os.path.join('cross_db_benchmark/datasets/', dataset, 'schema_sql', sql_filename)
    assert os.path.exists(sql_path), f"Could not find schema.sql ({sql_path})"
    with open(sql_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data
