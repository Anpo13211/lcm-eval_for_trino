"""
Plugin system initialization

This module should be imported at application startup to register all available plugins.

Usage:
    # In your main.py or __init__.py
    import core.init_plugins
    
    # Or explicitly
    from core.init_plugins import initialize_plugins
    initialize_plugins()
"""

def initialize_plugins():
    """
    Initialize and register all available DBMS plugins.
    
    This function:
    1. Discovers available plugin implementations
    2. Registers them with the global registry
    3. Validates that they are properly configured
    
    Call this once at application startup.
    """
    from core.plugins.registry import DBMSRegistry
    
    print("="*60)
    print("Initializing DBMS Plugin System")
    print("="*60)
    
    # Register PostgreSQL plugin
    try:
        from plugins.postgres.plugin import register as register_postgres
        register_postgres()
    except Exception as e:
        print(f"Warning: Could not register PostgreSQL plugin: {e}")
    
    # Register Trino plugin
    try:
        from plugins.trino.plugin import register as register_trino
        register_trino()
    except Exception as e:
        print(f"Warning: Could not register Trino plugin: {e}")
    
    # Add more plugins as they become available
    # try:
    #     from plugins.mysql.plugin import register as register_mysql
    #     register_mysql()
    # except Exception as e:
    #     print(f"Warning: Could not register MySQL plugin: {e}")
    
    # Print summary
    plugins = DBMSRegistry.list_plugins()
    print(f"\nRegistered {len(plugins)} DBMS plugins:")
    for plugin_name in plugins:
        plugin = DBMSRegistry.get_plugin(plugin_name)
        metadata = plugin.get_metadata()
        print(f"  - {metadata['display_name']} ({plugin_name})")
    
    print("="*60)
    print()


# Auto-initialize when this module is imported
initialize_plugins()

