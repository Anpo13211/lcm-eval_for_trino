from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.compare_plan import compare_plans
from cross_db_benchmark.benchmark_tools.trino.compare_plan import compare_trino_plans
from cross_db_benchmark.benchmark_tools.utils import load_json


def compare_runs(source_path, alt_source_path, database, min_query_ms=100):
    if database == DatabaseSystem.POSTGRES:
        compare_func = compare_plans
    elif database == DatabaseSystem.TRINO:
        compare_func = compare_trino_plans
    else:
        raise NotImplementedError(f"Database {database} not yet supported.")

    run_stats = load_json(source_path)
    alt_run_stats = load_json(alt_source_path)

    compare_func(run_stats, alt_run_stats, min_runtime=min_query_ms)
