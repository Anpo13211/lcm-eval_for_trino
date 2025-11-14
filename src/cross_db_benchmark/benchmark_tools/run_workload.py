from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.run_workload import run_pg_workload
from cross_db_benchmark.benchmark_tools.trino.run_workload import run_trino_workload
from core.plugins.registry import DBMSRegistry


def run_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path, run_kwargs,
                 repetitions_per_query, timeout_sec, mode, hints=None, with_indexes=False, cap_workload=None, explain_only: bool = False,
                 min_runtime=100):
    """
    Run a workload against a database.
    
    Args:
        database: DatabaseSystem enum OR string (e.g., 'postgres', 'trino')
                 Supports both for backward compatibility
    
    Note: This function currently delegates to DBMS-specific implementations.
          Future enhancement: Add run_workload() method to plugin interface.
    """
    # Convert DatabaseSystem enum to string if needed
    if isinstance(database, DatabaseSystem):
        dbms_name = database.value
    else:
        dbms_name = database
    
    # Registry-based dispatch (still calls existing implementations)
    if dbms_name == 'postgres':
        run_pg_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path,
                        run_kwargs, repetitions_per_query, timeout_sec, random_hints=hints, with_indexes=with_indexes,
                        cap_workload=cap_workload, min_runtime=min_runtime, mode=mode, explain_only=explain_only)
    elif dbms_name == 'trino':
        run_trino_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path,
                           run_kwargs, repetitions_per_query, timeout_sec, with_indexes=with_indexes,
                           cap_workload=cap_workload, min_runtime=min_runtime, mode=mode, explain_only=explain_only)
    else:
        # Check if plugin exists and provide helpful error
        if DBMSRegistry.is_registered(dbms_name):
            raise NotImplementedError(
                f"DBMS '{dbms_name}' is registered but does not yet support run_workload(). "
                f"This requires adding a run_workload implementation for {dbms_name}."
            )
        else:
            raise NotImplementedError(
                f"Database '{dbms_name}' not registered. "
                f"Available: {DBMSRegistry.list_plugins()}"
            )
