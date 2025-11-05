"""
Unified database loading using plugin system

This replaces the old load_database.py with a plugin-based implementation.

Before:
    if database == DatabaseSystem.POSTGRES:
        return PostgresDatabaseConnection(...)
    else:
        raise NotImplementedError

After:
    conn_factory = DBMSRegistry.get_connection_factory(database)
    return conn_factory(...)
"""

from core.plugins.registry import DBMSRegistry
from cross_db_benchmark.benchmark_tools.database import DatabaseSystem


def create_db_conn_unified(database: str, db_name: str, database_conn_args: dict, database_kwarg_dict: dict):
    """
    Create database connection using plugin system.
    
    This is a drop-in replacement for create_db_conn() in load_database.py
    
    Args:
        database: DatabaseSystem enum or string name
        db_name: Database name
        database_conn_args: Connection arguments
        database_kwarg_dict: Additional keyword arguments
    
    Returns:
        DatabaseConnection instance
    
    Raises:
        KeyError: If database system not registered
    """
    # Convert DatabaseSystem enum to string if needed
    if isinstance(database, DatabaseSystem):
        dbms_name = database.value if hasattr(database, 'value') else str(database)
    else:
        dbms_name = str(database).lower()
    
    # Get connection factory from registry
    conn_factory = DBMSRegistry.get_connection_factory(dbms_name)
    
    # Create connection instance
    return conn_factory(
        db_name=db_name,
        database_kwargs=database_conn_args,
        **database_kwarg_dict
    )


def load_database_unified(
    data_dir: str,
    dataset: str,
    database: str,
    db_name: str,
    database_conn_args: dict,
    database_kwarg_dict: dict,
    force: bool = False
):
    """
    Load database using plugin system.
    
    This is a drop-in replacement for load_database() in load_database.py
    
    Args:
        data_dir: Directory containing data files
        dataset: Dataset name
        database: DatabaseSystem enum or string name
        db_name: Database name
        database_conn_args: Connection arguments
        database_kwarg_dict: Additional keyword arguments
        force: Force reload even if database exists
    """
    # Create connection using plugin system
    db_conn = create_db_conn_unified(
        database, db_name, database_conn_args, database_kwarg_dict
    )
    
    # Load database
    db_conn.load_database(dataset, data_dir, force=force)
    
    return db_conn


# Backward compatibility: expose the same interface as load_database.py
create_db_conn = create_db_conn_unified
load_database = load_database_unified

