"""
Global registry for DBMS plugins

This singleton registry allows:
1. Registration of DBMS plugins at application startup
2. Lookup of parsers, converters, and connections by DBMS name
3. Discovery of available DBMS implementations

Usage:
    # Registration (typically in __init__.py or main.py)
    from core.plugins import DBMSRegistry
    from plugins.postgres import PostgreSQLPlugin
    
    DBMSRegistry.register(PostgreSQLPlugin())
    
    # Lookup
    parser = DBMSRegistry.get_parser("postgres")
    converter = DBMSRegistry.get_statistics_converter("postgres")
"""

from typing import Dict, List, Optional, Type
import threading

from .dbms_plugin import DBMSPlugin


class DBMSRegistry:
    """
    Thread-safe singleton registry for DBMS plugins.
    
    This replaces the scattered DBMS-specific imports and conditional logic
    throughout the codebase with a centralized plugin discovery system.
    
    Before:
        if database == DatabaseSystem.POSTGRES:
            return PostgresDatabaseConnection(...)
        elif database == DatabaseSystem.TRINO:
            return TrinoDatabaseConnection(...)
        else:
            raise NotImplementedError
    
    After:
        conn_factory = DBMSRegistry.get_connection_factory(database)
        return conn_factory(...)
    """
    
    _plugins: Dict[str, DBMSPlugin] = {}
    _lock = threading.Lock()
    _initialized = False
    
    @classmethod
    def register(cls, plugin: DBMSPlugin) -> None:
        """
        Register a DBMS plugin.
        
        Args:
            plugin: DBMSPlugin instance to register
            
        Raises:
            ValueError: If plugin with same name already registered
            RuntimeError: If plugin validation fails
        """
        with cls._lock:
            if plugin.name in cls._plugins:
                raise ValueError(
                    f"Plugin '{plugin.name}' is already registered. "
                    f"Use update() to replace existing plugin."
                )
            
            if not plugin.validate():
                raise RuntimeError(
                    f"Plugin '{plugin.name}' failed validation. "
                    f"Ensure all required methods are implemented."
                )
            
            cls._plugins[plugin.name] = plugin
            print(f"Registered DBMS plugin: {plugin.display_name} ({plugin.name})")
    
    @classmethod
    def update(cls, plugin: DBMSPlugin) -> None:
        """
        Update an existing plugin or register a new one.
        
        Args:
            plugin: DBMSPlugin instance
        """
        with cls._lock:
            if not plugin.validate():
                raise RuntimeError(f"Plugin '{plugin.name}' failed validation")
            
            cls._plugins[plugin.name] = plugin
            print(f"Updated DBMS plugin: {plugin.display_name} ({plugin.name})")
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a DBMS plugin.
        
        Args:
            name: Plugin name (e.g., "postgres")
        """
        with cls._lock:
            if name in cls._plugins:
                del cls._plugins[name]
                print(f"Unregistered DBMS plugin: {name}")
    
    @classmethod
    def get_plugin(cls, name: str) -> Optional[DBMSPlugin]:
        """
        Get a registered plugin by name.
        
        Args:
            name: Plugin name (e.g., "postgres", "trino")
            
        Returns:
            DBMSPlugin instance or None if not found
        """
        return cls._plugins.get(name)
    
    @classmethod
    def get_parser(cls, name: str) -> 'AbstractPlanParser':
        """
        Get plan parser for a DBMS.
        
        Args:
            name: DBMS name (e.g., "postgres")
            
        Returns:
            AbstractPlanParser instance
            
        Raises:
            KeyError: If DBMS not registered
        """
        plugin = cls._plugins.get(name)
        if plugin is None:
            raise KeyError(
                f"DBMS '{name}' not registered. "
                f"Available: {list(cls._plugins.keys())}"
            )
        return plugin.get_parser()
    
    @classmethod
    def get_statistics_converter(cls, name: str) -> 'StatisticsConverter':
        """
        Get statistics converter for a DBMS.
        
        Args:
            name: DBMS name
            
        Returns:
            StatisticsConverter instance
            
        Raises:
            KeyError: If DBMS not registered
        """
        plugin = cls._plugins.get(name)
        if plugin is None:
            raise KeyError(
                f"DBMS '{name}' not registered. "
                f"Available: {list(cls._plugins.keys())}"
            )
        return plugin.get_statistics_converter()
    
    @classmethod
    def get_connection_factory(cls, name: str) -> Type['DatabaseConnection']:
        """
        Get database connection factory for a DBMS.
        
        Args:
            name: DBMS name
            
        Returns:
            DatabaseConnection class (not instance)
            
        Raises:
            KeyError: If DBMS not registered
        """
        plugin = cls._plugins.get(name)
        if plugin is None:
            raise KeyError(
                f"DBMS '{name}' not registered. "
                f"Available: {list(cls._plugins.keys())}"
            )
        return plugin.get_connection_factory()
    
    @classmethod
    def get_operator_normalizer(cls, name: str) -> 'OperatorNormalizer':
        """
        Get operator normalizer for a DBMS.
        
        Args:
            name: DBMS name
            
        Returns:
            OperatorNormalizer instance
            
        Raises:
            KeyError: If DBMS not registered
        """
        plugin = cls._plugins.get(name)
        if plugin is None:
            raise KeyError(
                f"DBMS '{name}' not registered. "
                f"Available: {list(cls._plugins.keys())}"
            )
        return plugin.get_operator_normalizer()
    
    @classmethod
    def list_plugins(cls) -> List[str]:
        """
        List all registered DBMS plugins.
        
        Returns:
            List of plugin names
        """
        return list(cls._plugins.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a DBMS plugin is registered.
        
        Args:
            name: DBMS name
            
        Returns:
            True if registered, False otherwise
        """
        return name in cls._plugins
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered plugins (mainly for testing).
        """
        with cls._lock:
            cls._plugins.clear()
            cls._initialized = False
    
    @classmethod
    def auto_discover(cls) -> None:
        """
        Auto-discover and register available DBMS plugins.
        
        This scans for plugin implementations and registers them automatically.
        Called during application initialization.
        """
        if cls._initialized:
            return
        
        with cls._lock:
            # Import and register built-in plugins
            try:
                from plugins.postgres.plugin import PostgreSQLPlugin
                cls.register(PostgreSQLPlugin())
            except ImportError as e:
                print(f"PostgreSQL plugin not available: {e}")
            
            try:
                from plugins.trino.plugin import TrinoPlugin
                cls.register(TrinoPlugin())
            except ImportError as e:
                print(f"Trino plugin not available: {e}")
            
            # Add more plugins as they become available
            # from plugins.mysql.plugin import MySQLPlugin
            # cls.register(MySQLPlugin())
            
            cls._initialized = True
    
    @classmethod
    def get_metadata_all(cls) -> Dict[str, Dict]:
        """
        Get metadata for all registered plugins.
        
        Returns:
            Dictionary mapping plugin names to metadata
        """
        return {
            name: plugin.get_metadata()
            for name, plugin in cls._plugins.items()
        }


# Forward declarations for type hints
try:
    from cross_db_benchmark.benchmark_tools.abstract.plan_parser import AbstractPlanParser
except ImportError:
    AbstractPlanParser = None

try:
    from core.statistics.converter import StatisticsConverter
except ImportError:
    StatisticsConverter = None

try:
    from cross_db_benchmark.benchmark_tools.database import DatabaseConnection
except ImportError:
    DatabaseConnection = None

try:
    from core.operators.normalizer import OperatorNormalizer
except ImportError:
    OperatorNormalizer = None

