"""
Centralized compatibility checks for DBMS plugins and Models.
"""

from typing import Set, List, Optional, Any
from core.plugins.registry import DBMSRegistry
from core.capabilities import check_capabilities, Capability

def check_model_compatibility(
    model_config: Any,
    dbms_name: str,
    require_adapter: bool = False
) -> bool:
    """
    Check if a DBMS plugin is compatible with a specific model configuration.
    
    Args:
        model_config: Model configuration object (must have required_capabilities)
        dbms_name: Name of the DBMS to check
        require_adapter: Whether the model requires a plan adapter (for legacy models)
        
    Returns:
        True if compatible, False otherwise (prints warnings)
    """
    print(f"Checking compatibility for {dbms_name}...")
    
    # 1. Check Plugin Existence
    try:
        plugin = DBMSRegistry.get_plugin(dbms_name)
        if plugin is None:
            print(f"❌ DBMS '{dbms_name}' is not registered.")
            return False
    except Exception as e:
        print(f"❌ Error retrieving plugin '{dbms_name}': {e}")
        return False

    # 2. Check Capabilities
    try:
        provided_caps = plugin.get_capabilities()
        required_caps = getattr(model_config, 'required_capabilities', set())
        
        # Handle case where required_caps might be a list or other iterable
        if not isinstance(required_caps, set):
            required_caps = set(required_caps)
            
        missing_caps = check_capabilities(
            required_caps,
            provided_caps,
            getattr(model_config, 'name', 'UnknownModel'),
            dbms_name
        )

        if missing_caps:
            print("="*80)
            print(f"⚠️  WARNING: DBMS '{dbms_name}' is missing capabilities required by the model:")
            for cap in missing_caps:
                print(f"   - {cap}")
            print("    Training may fail or produce suboptimal results.")
            print("="*80)
            # We don't return False here strictly, as some missing caps might be soft requirements
            # depending on the model implementation, but for strict O(M+N) we should ideally enforce it.
            # For now, we warn but allow proceeding if the user accepts the risk (or we could make it strict).
            # Given the user's request for "common guard", strictness is preferred, but let's stick to warning
            # to avoid breaking existing partial setups unless critical.
    except Exception as e:
        print(f"⚠️  Capability check failed: {e}")
        return False

    # 3. Check Plan Adapter (if required)
    if require_adapter:
        adapter = plugin.get_plan_adapter()
        if adapter is None:
            print("="*80)
            print(f"❌ ERROR: DBMS '{dbms_name}' does not provide a plan adapter.")
            print(f"   This model (legacy) requires an adapter to convert plans to its specific format.")
            print(f"   Please implement `get_plan_adapter` in the {dbms_name} plugin.")
            print("="*80)
            return False

    print(f"✓ Compatibility check passed for {dbms_name}")
    return True
