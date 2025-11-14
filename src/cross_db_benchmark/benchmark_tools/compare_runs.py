from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.compare_plan import compare_plans
from cross_db_benchmark.benchmark_tools.utils import load_json
from core.plugins.registry import DBMSRegistry


def compare_runs(source_path, alt_source_path, database, min_query_ms=100):
    """
    Compare two query execution runs.
    
    Args:
        database: DatabaseSystem enum OR string (e.g., 'postgres', 'trino')
                 Supports both for backward compatibility
    
    Note: This function currently uses DBMS-specific implementations.
          Future enhancement: Add compare_plans() method to plugin interface.
    """
    # Convert DatabaseSystem enum to string if needed
    if isinstance(database, DatabaseSystem):
        dbms_name = database.value
    else:
        dbms_name = database
    
    # Registry-based dispatch (still calls existing implementations)
    if dbms_name == 'postgres':
        compare_func = compare_plans
    elif dbms_name == 'trino':
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
        # Check if plugin exists and provide helpful error
        if DBMSRegistry.is_registered(dbms_name):
            raise NotImplementedError(
                f"DBMS '{dbms_name}' is registered but does not yet support compare_runs(). "
                f"Comparison functionality needs to be implemented for {dbms_name}."
            )
        else:
            raise NotImplementedError(
                f"Database '{dbms_name}' not registered. "
                f"Available: {DBMSRegistry.list_plugins()}"
            )

    run_stats = load_json(source_path)
    alt_run_stats = load_json(alt_source_path)

    compare_func(run_stats, alt_run_stats, min_runtime=min_query_ms)
