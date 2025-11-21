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

import importlib
import pkgutil
import sys
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

def initialize_plugins():
    """
    Initialize and register all available DBMS plugins.
    
    This function:
    1. Discovers available plugin implementations in src/plugins
    2. Registers them with the global registry
    3. Registers feature aliases provided by plugins
    
    Call this once at application startup.
    """
    from core.plugins.registry import DBMSRegistry
    
    print("="*60)
    print("Initializing DBMS Plugin System")
    print("="*60)
    
    # Dynamic discovery of plugins
    try:
        import plugins
        
        # Walk through the plugins package
        package = plugins
        prefix = package.__name__ + "."
        
        if hasattr(package, '__path__'):
            for _, name, ispkg in pkgutil.iter_modules(package.__path__, prefix):
                if ispkg:
                    # Expecting structure: plugins.<dbms>.plugin
                    try:
                        module_name = f"{name}.plugin"
                        module = importlib.import_module(module_name)
                        
                        # Check for register function
                        if hasattr(module, 'register') and callable(module.register):
                            try:
                                module.register()
                            except Exception as e:
                                print(f"Error registering plugin {name}: {e}")
                        else:
                            # Optional: Warn if no register function found, 
                            # but some packages might be helpers (e.g. 'common')
                            pass
                            
                    except ImportError as e:
                        # It's possible the package doesn't follow the .plugin convention
                        # We silently skip in that case to avoid noise for utility packages
                        # print(f"Debug: Could not import {module_name}: {e}")
                        pass
                    except Exception as e:
                        print(f"Error loading plugin module {name}: {e}")

        else:
            print("Warning: 'plugins' module has no __path__, cannot walk packages.")

    except ImportError as e:
        print(f"Warning: Could not import 'plugins' package: {e}")
        print("Ensure 'src' directory is in PYTHONPATH.")
    
    # Register feature aliases from plugins
    from core.features.registry import register_dbms_aliases
    
    for plugin_name in DBMSRegistry.list_plugins():
        try:
            plugin = DBMSRegistry.get_plugin(plugin_name)
            # Check if plugin supports get_feature_aliases
            if hasattr(plugin, 'get_feature_aliases'):
                feature_aliases = plugin.get_feature_aliases()
                if feature_aliases:
                    register_dbms_aliases(plugin_name, feature_aliases)
                    print(f"  Registered {len(feature_aliases)} feature aliases for {plugin_name}")
            else:
                # Should implement get_feature_aliases in all plugins
                print(f"  Warning: Plugin {plugin_name} does not support feature aliases yet.")
        except Exception as e:
            print(f"Warning: Could not register feature aliases for {plugin_name}: {e}")
    
    # Print summary
    plugins_list = DBMSRegistry.list_plugins()
    print(f"\nRegistered {len(plugins_list)} DBMS plugins:")
    for plugin_name in plugins_list:
        plugin = DBMSRegistry.get_plugin(plugin_name)
        metadata = plugin.get_metadata()
        print(f"  - {metadata['display_name']} ({plugin_name})")
    
    print("="*60)
    print()


# Auto-initialize when this module is imported
initialize_plugins()
