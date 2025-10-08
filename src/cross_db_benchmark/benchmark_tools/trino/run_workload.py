"""
Trinoワークロード実行
"""

import json
import os
import time
from pathlib import Path

from cross_db_benchmark.benchmark_tools.database import ExecutionMode
from cross_db_benchmark.benchmark_tools.trino.database_connection import TrinoDatabaseConnection


def run_trino_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path, run_kwargs,
                       repetitions_per_query, timeout_sec, with_indexes=False, cap_workload=None, random_hints=None,
                       min_runtime=100, mode: ExecutionMode = ExecutionMode.JSON_OUTPUT, explain_only: bool = False):
    """Trinoワークロードを実行"""
    
    print(f"Running Trino workload: {workload_path}")
    print(f"Target path: {target_path}")
    print(f"Repetitions per query: {repetitions_per_query}")
    print(f"Timeout: {timeout_sec}s")
    print(f"Explain only: {explain_only}")
    
    # データベース接続を作成
    db_conn = TrinoDatabaseConnection(db_name=db_name, database_kwargs=database_conn_args, **database_kwarg_dict)
    
    # タイムアウトを設定
    db_conn.set_statement_timeout(timeout_sec)
    
    # ワークロードファイルを読み込み
    with open(workload_path, 'r') as f:
        workload_content = f.read()
    
    # SQLクエリを分割
    queries = []
    current_query = []
    
    for line in workload_content.split('\n'):
        line = line.strip()
        if line and not line.startswith('--'):
            current_query.append(line)
            if line.endswith(';'):
                query = ' '.join(current_query)
                if query.strip():
                    queries.append(query)
                current_query = []
    
    # クエリ数を制限
    if cap_workload:
        queries = queries[:cap_workload]
    
    print(f"Found {len(queries)} queries")
    
    # 結果を格納するリスト
    results = []
    
    for i, sql in enumerate(queries):
        print(f"Running query {i+1}/{len(queries)}")
        
        try:
            # クエリを実行して統計を収集
            query_results = db_conn.run_query_collect_statistics(
                sql=sql,
                repetitions=repetitions_per_query,
                prefix=f"q{i+1}",
                hint_validation=False,
                include_hint_notices=False,
                explain_only=explain_only
            )
            
            # 結果を追加
            for result in query_results:
                result['query_id'] = i + 1
                result['database'] = 'trino'
                results.append(result)
                
        except Exception as e:
            print(f"Query {i+1} failed: {e}")
            # エラー結果を追加
            for rep in range(repetitions_per_query):
                results.append({
                    'query_id': i + 1,
                    'sql': sql,
                    'error': str(e),
                    'repetition': rep,
                    'database': 'trino'
                })
    
    # 結果をJSONファイルに保存
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Workload completed. Results saved to: {target_path}")
    print(f"Total queries executed: {len(queries)}")
    print(f"Total results: {len(results)}")
    
    # 接続を閉じる
    db_conn.drop()
    
    return results
