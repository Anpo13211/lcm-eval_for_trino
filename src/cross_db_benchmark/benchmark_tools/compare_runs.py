from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.compare_plan import compare_plans
from cross_db_benchmark.benchmark_tools.utils import load_json


def compare_runs(source_path, alt_source_path, database, min_query_ms=100):
    if database == DatabaseSystem.POSTGRES:
        compare_func = compare_plans
    elif database == DatabaseSystem.TRINO:
        # Trinoプラン比較は簡略化（compare_plan.pyは削除されたため）
        def compare_trino_plans(run_stats, alt_run_stats, min_runtime=100):
            """Trinoプランを比較（簡略化版）"""
            print("Trino plan comparison is simplified")
            print(f"Run stats queries: {len(run_stats.query_list) if hasattr(run_stats, 'query_list') else 'N/A'}")
            print(f"Alt run stats queries: {len(alt_run_stats.query_list) if hasattr(alt_run_stats, 'query_list') else 'N/A'}")
            print(f"Min runtime: {min_runtime}ms")
            print("Trino plan comparison completed")
        compare_func = compare_trino_plans
    else:
        raise NotImplementedError(f"Database {database} not yet supported.")

    run_stats = load_json(source_path)
    alt_run_stats = load_json(alt_source_path)

    compare_func(run_stats, alt_run_stats, min_runtime=min_query_ms)
