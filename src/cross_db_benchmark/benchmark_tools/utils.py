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


def convert_trino_to_zeroshot_format(trino_stats):
    """
    Trino形式の統計情報をzero-shot形式に変換
    
    Trino形式: {("table", "column"): {stats}}
    zero-shot形式: {table: {column: {stats}}}
    
    Args:
        trino_stats: Trino形式の統計情報
    
    Returns:
        zero-shot形式の統計情報
    """
    zeroshot_format = {}
    
    for key, stats in trino_stats.items():
        # キーからテーブル名とカラム名を取得
        table = stats.get('table', 'unknown')
        column = stats.get('column', 'unknown')
        
        if table not in zeroshot_format:
            zeroshot_format[table] = {}
        
        # zero-shot形式に必要なフィールドを抽出・変換
        zeroshot_format[table][column] = {
            'datatype': stats.get('datatype', 'misc'),
            'min': stats.get('min'),
            'max': stats.get('max'),
            'percentiles': stats.get('percentiles'),
            'nan_ratio': stats.get('null_frac', 0.0),
            'num_unique': int(stats['distinct_values_count']) if stats.get('distinct_values_count') is not None else stats.get('n_distinct', -1),
            # 追加フィールド（あれば保持）
            'mean': stats.get('mean'),
        }
        
        # カテゴリカル型の場合は unique_vals を追加（もしあれば）
        if stats.get('datatype') == 'categorical' and 'unique_vals' in stats:
            zeroshot_format[table][column]['unique_vals'] = stats['unique_vals']
    
    return zeroshot_format


def load_column_statistics(dataset, namespace=True, prefer_zero_shot=False, prefer_trino=False):
    """
    カラム統計情報JSONファイルを読み込む
    
    Args:
        dataset: データセット名
        namespace: Trueの場合、SimpleNamespace形式で返す
        prefer_zero_shot: Trueの場合、zero-shot_datasets配下を優先的に探す
        prefer_trino: Trueの場合、datasets_statistics（Trino統計）を最優先
    
    Returns:
        カラム統計情報（namespace=Trueの場合はSimpleNamespace形式、Falseの場合はdict形式）
    """
    # パスの候補リスト
    paths = []
    # Docker環境とローカル環境の両方に対応
    zero_shot_base = os.getenv('ZERO_SHOT_DATASETS_DIR', '/Users/an/query_engine/lakehouse/zero-shot_datasets')
    trino_stats_dir = os.getenv('TRINO_STATS_DIR', 'datasets_statistics')
    
    # プロジェクトルートを取得（utils.pyの場所から推測）
    # src/cross_db_benchmark/benchmark_tools/utils.py から プロジェクトルートへ
    # .. を3回上がるとプロジェクトルート（lcm-eval_for_trino）
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..'))
    
    if prefer_trino:
        # Trino統計を最優先
        # 絶対パス（プロジェクトルートから）を最優先、次に相対パス
        trino_paths = [
            os.path.join(project_root, trino_stats_dir, f'iceberg_{dataset}', 'column_stats.json'),  # 絶対パス（最優先）
            os.path.abspath(os.path.join(trino_stats_dir, f'iceberg_{dataset}', 'column_stats.json')),  # 現在のディレクトリからの絶対パス
            os.path.join('..', trino_stats_dir, f'iceberg_{dataset}', 'column_stats.json'),  # srcから実行時の場合
            os.path.join(trino_stats_dir, f'iceberg_{dataset}', 'column_stats.json'),  # プロジェクトルートから実行時の場合
        ]
        # 重複を除去しつつ順序を保持
        seen = set()
        for trino_path in trino_paths:
            if trino_path not in seen:
                paths.append((trino_path, 'trino'))
                seen.add(trino_path)
        paths.append((os.path.join(zero_shot_base, dataset, 'column_statistics.json'), 'zeroshot'))
        paths.append((os.path.join(project_root, 'cross_db_benchmark/datasets/', dataset, 'column_statistics.json'), 'zeroshot'))
        paths.append((os.path.join('cross_db_benchmark/datasets/', dataset, 'column_statistics.json'), 'zeroshot'))
    elif prefer_zero_shot:
        # zero-shot_datasets配下を優先
        paths.append((os.path.join(zero_shot_base, dataset, 'column_statistics.json'), 'zeroshot'))
        # Trino統計も複数パスで試す
        trino_paths = [
            os.path.join(project_root, trino_stats_dir, f'iceberg_{dataset}', 'column_stats.json'),
            os.path.join(trino_stats_dir, f'iceberg_{dataset}', 'column_stats.json'),
            os.path.join('..', trino_stats_dir, f'iceberg_{dataset}', 'column_stats.json'),
        ]
        for trino_path in trino_paths:
            paths.append((trino_path, 'trino'))
        paths.append((os.path.join(project_root, 'cross_db_benchmark/datasets/', dataset, 'column_statistics.json'), 'zeroshot'))
        paths.append((os.path.join('cross_db_benchmark/datasets/', dataset, 'column_statistics.json'), 'zeroshot'))
    else:
        # 従来のパスを優先（後方互換性）
        paths.append((os.path.join(project_root, 'cross_db_benchmark/datasets/', dataset, 'column_statistics.json'), 'zeroshot'))
        paths.append((os.path.join('cross_db_benchmark/datasets/', dataset, 'column_statistics.json'), 'zeroshot'))
        paths.append((os.path.join(zero_shot_base, dataset, 'column_statistics.json'), 'zeroshot'))
        # Trino統計も複数パスで試す
        trino_paths = [
            os.path.join(project_root, trino_stats_dir, f'iceberg_{dataset}', 'column_stats.json'),
            os.path.join(trino_stats_dir, f'iceberg_{dataset}', 'column_stats.json'),
            os.path.join('..', trino_stats_dir, f'iceberg_{dataset}', 'column_stats.json'),
        ]
        for trino_path in trino_paths:
            paths.append((trino_path, 'trino'))
    
    # 存在するパスを探す
    for path, format_type in paths:
        if os.path.exists(path):
            print(f"✅ Loading column statistics from: {path} (format: {format_type})")
            
            if format_type == 'trino':
                # Trino形式を読み込んでzero-shot形式に変換
                trino_stats = load_json(path, namespace=False)
                zeroshot_stats = convert_trino_to_zeroshot_format(trino_stats)
                
                if namespace:
                    # dictをSimpleNamespace形式に変換
                    return json.loads(json.dumps(zeroshot_stats), object_hook=lambda d: SimpleNamespace(**d))
                else:
                    return zeroshot_stats
            else:
                # zero-shot形式をそのまま読み込む
                return load_json(path, namespace=namespace)
    
    # 見つからない場合はエラー
    tried_paths = [path for path, _ in paths]
    raise FileNotFoundError(f"Could not find column_statistics.json. Tried:\n" + "\n".join(f"  - {p}" for p in tried_paths))


def load_schema_json_trino(dataset):
    """
    Trino用のschema.json読み込み関数（後方互換性のため残す）
    zero-shot_datasets配下を優先的に探す
    """
    return load_schema_json(dataset, prefer_zero_shot=True)


def load_column_statistics_trino(dataset, namespace=True, prefer_trino=True):
    """
    Trino用のcolumn_statistics.json読み込み関数
    デフォルトでTrino統計（datasets_statistics）を優先的に使用
    
    Args:
        dataset: データセット名
        namespace: Trueの場合、SimpleNamespace形式で返す
        prefer_trino: Trueの場合、Trino統計を優先（デフォルト: True）
    """
    return load_column_statistics(dataset, namespace=namespace, prefer_trino=prefer_trino)


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
