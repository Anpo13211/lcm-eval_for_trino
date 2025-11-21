from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from core.plugins.registry import DBMSRegistry


def run_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path, run_kwargs,
                 repetitions_per_query, timeout_sec, mode, hints=None, with_indexes=False, cap_workload=None, explain_only: bool = False,
                 min_runtime=100):
    """
    Run a workload against a database.
    
    Args:
        database: DatabaseSystem enum OR string (e.g., 'postgres', 'trino')
                 Supports both for backward compatibility
    
    Note: This function delegates to DBMS-specific implementations via the plugin system.
    """
    # Convert DatabaseSystem enum to string if needed
    if isinstance(database, DatabaseSystem):
        dbms_name = database.value
    else:
        dbms_name = database
    
    # Get plugin from registry
    try:
        plugin = DBMSRegistry.get_plugin(dbms_name)
    except KeyError:
        raise NotImplementedError(
            f"Database '{dbms_name}' not registered. "
            f"Available: {DBMSRegistry.list_plugins()}"
        )

    # Dispatch to plugin
    try:
        plugin.run_workload(
            workload_path=workload_path, 
            db_name=db_name, 
            database_conn_args=database_conn_args, 
            database_kwarg_dict=database_kwarg_dict, 
            target_path=target_path, 
            run_kwargs=run_kwargs,
            repetitions_per_query=repetitions_per_query, 
            timeout_sec=timeout_sec, 
            mode=mode, 
            hints=hints, 
            with_indexes=with_indexes, 
            cap_workload=cap_workload, 
            explain_only=explain_only, 
            min_runtime=min_runtime
        )
    except NotImplementedError:
        raise NotImplementedError(
            f"DBMS '{dbms_name}' is registered but does not implement run_workload(). "
            f"Please update the plugin implementation."
        )
