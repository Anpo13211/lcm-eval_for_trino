from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from core.plugins.registry import DBMSRegistry


def create_db_conn(database, db_name, database_conn_args, database_kwarg_dict):
    """
    Create a database connection using the plugin registry.
    
    Args:
        database: DatabaseSystem enum OR string (e.g., 'postgres', 'trino')
                 Supports both for backward compatibility
    """
    # Convert DatabaseSystem enum to string if needed
    if isinstance(database, DatabaseSystem):
        dbms_name = database.value
    else:
        dbms_name = database
    
    # Get connection factory from registry (O(1) lookup!)
    try:
        connection_class = DBMSRegistry.get_connection_factory(dbms_name)
    except KeyError:
        raise NotImplementedError(
            f"Database '{dbms_name}' not registered. "
            f"Available: {DBMSRegistry.list_plugins()}"
        )
    
    # Instantiate and return connection
    return connection_class(db_name=db_name, database_kwargs=database_conn_args, **database_kwarg_dict)


def load_database(data_dir, dataset, database, db_name, database_conn_args, database_kwarg_dict, force=False):
    db_conn = create_db_conn(database, db_name, database_conn_args, database_kwarg_dict)
    db_conn.load_database(dataset, data_dir, force=force)
