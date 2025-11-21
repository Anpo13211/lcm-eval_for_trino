"""
Global feature registry

This module contains the canonical mapping of all logical features
to DBMS-specific attribute names.

When adding a new DBMS, simply add its attribute names to the appropriate
FeatureMapping entries. No code changes required in models.
"""

from .mapping import FeatureMapping, FeatureType, to_float, to_positive_float, to_int, normalize_operator_name


# ============================================================================
# PLAN FEATURES
# ============================================================================

PLAN_FEATURES = [
    FeatureMapping(
        logical_name="operator_type",
        feature_type=FeatureType.PLAN_OPERATOR,
        dbms_aliases={},
        default_value="unknown",
        transformation=normalize_operator_name,
        description="Type of plan operator (e.g., SeqScan, HashJoin)"
    ),
    
    FeatureMapping(
        logical_name="estimated_cardinality",
        feature_type=FeatureType.PLAN_CARDINALITY,
        dbms_aliases={},
        default_value=1.0,
        transformation=to_positive_float,
        min_value=0.0,
        description="Estimated number of rows output by this operator"
    ),
    
    FeatureMapping(
        logical_name="actual_cardinality",
        feature_type=FeatureType.PLAN_CARDINALITY,
        dbms_aliases={},
        default_value=None,
        transformation=to_positive_float,
        min_value=0.0,
        description="Actual number of rows (from EXPLAIN ANALYZE)"
    ),
    
    FeatureMapping(
        logical_name="estimated_cost",
        feature_type=FeatureType.PLAN_COST,
        dbms_aliases={},
        default_value=0.0,
        transformation=to_positive_float,
        min_value=0.0,
        description="Estimated total cost of operator"
    ),
    
    FeatureMapping(
        logical_name="startup_cost",
        feature_type=FeatureType.PLAN_COST,
        dbms_aliases={},
        default_value=0.0,
        transformation=to_positive_float,
        min_value=0.0,
        description="Estimated startup cost"
    ),
    
    FeatureMapping(
        logical_name="estimated_width",
        feature_type=FeatureType.PLAN_WIDTH,
        dbms_aliases={},
        default_value=0,
        transformation=to_int,
        min_value=0,
        description="Estimated average row width in bytes"
    ),
    
    FeatureMapping(
        logical_name="workers_planned",
        feature_type=FeatureType.PLAN_PARALLELISM,
        dbms_aliases={},
        default_value=0,
        transformation=to_int,
        min_value=0,
        description="Number of parallel workers planned"
    ),
    
    FeatureMapping(
        logical_name="workers_launched",
        feature_type=FeatureType.PLAN_PARALLELISM,
        dbms_aliases={},
        default_value=0,
        transformation=to_int,
        min_value=0,
        description="Number of parallel workers actually launched"
    ),
    
    # PostgreSQL-specific but included for compatibility
    FeatureMapping(
        logical_name="actual_children_cardinality",
        feature_type=FeatureType.PLAN_CARDINALITY,
        dbms_aliases={},
        default_value=None,
        transformation=to_positive_float,
        description="Sum of actual cardinalities of child nodes"
    ),
]


# ============================================================================
# FILTER/PREDICATE FEATURES
# ============================================================================

FILTER_FEATURES = [
    FeatureMapping(
        logical_name="filter_operator",
        feature_type=FeatureType.FILTER_OPERATOR,
        dbms_aliases={},
        default_value="=",
        description="Comparison operator (=, <, >, LIKE, etc.)"
    ),
    
    FeatureMapping(
        logical_name="literal_feature",
        feature_type=FeatureType.FILTER_LITERAL,
        dbms_aliases={},
        default_value=0.0,
        transformation=to_float,
        description="Encoded literal value in predicate"
    ),
]


# ============================================================================
# COLUMN FEATURES
# ============================================================================

COLUMN_FEATURES = [
    FeatureMapping(
        logical_name="avg_width",
        feature_type=FeatureType.COLUMN_STATISTICS,
        dbms_aliases={},
        default_value=8,
        transformation=to_int,
        min_value=0,
        description="Average column width in bytes"
    ),
    
    FeatureMapping(
        logical_name="correlation",
        feature_type=FeatureType.COLUMN_STATISTICS,
        dbms_aliases={},
        default_value=0.0,
        transformation=to_float,
        min_value=-1.0,
        max_value=1.0,
        description="Physical storage correlation (-1 to 1)"
    ),
    
    FeatureMapping(
        logical_name="data_type",
        feature_type=FeatureType.COLUMN_TYPE,
        dbms_aliases={},
        default_value="unknown",
        description="Column data type"
    ),
    
    FeatureMapping(
        logical_name="n_distinct",
        feature_type=FeatureType.COLUMN_STATISTICS,
        dbms_aliases={},
        default_value=1.0,
        transformation=to_positive_float,
        min_value=0.0,
        description="Number of distinct values"
    ),
    
    FeatureMapping(
        logical_name="null_frac",
        feature_type=FeatureType.COLUMN_STATISTICS,
        dbms_aliases={},
        default_value=0.0,
        transformation=to_float,
        min_value=0.0,
        max_value=1.0,
        description="Fraction of NULL values (0.0 to 1.0)"
    ),
]


# ============================================================================
# TABLE FEATURES
# ============================================================================

TABLE_FEATURES = [
    FeatureMapping(
        logical_name="row_count",
        feature_type=FeatureType.TABLE_STATISTICS,
        dbms_aliases={},
        default_value=0.0,
        transformation=to_positive_float,
        min_value=0.0,
        description="Total number of rows in table"
    ),
    
    FeatureMapping(
        logical_name="page_count",
        feature_type=FeatureType.TABLE_STATISTICS,
        dbms_aliases={},
        default_value=0,
        transformation=to_int,
        min_value=0,
        description="Number of storage pages"
    ),
]


# ============================================================================
# OUTPUT COLUMN FEATURES
# ============================================================================

OUTPUT_COLUMN_FEATURES = [
    FeatureMapping(
        logical_name="aggregation",
        feature_type=FeatureType.OUTPUT_AGGREGATION,
        dbms_aliases={},
        default_value="none",
        description="Aggregation function (sum, avg, count, etc.)"
    ),
]


# ============================================================================
# FEATURE REGISTRY
# ============================================================================

# Create global registry by logical name
FEATURE_REGISTRY: dict = {}

for feature in (PLAN_FEATURES + FILTER_FEATURES + COLUMN_FEATURES + 
                TABLE_FEATURES + OUTPUT_COLUMN_FEATURES):
    FEATURE_REGISTRY[feature.logical_name] = feature


# Grouped registries for convenience
FEATURE_GROUPS = {
    'plan': PLAN_FEATURES,
    'filter': FILTER_FEATURES,
    'column': COLUMN_FEATURES,
    'table': TABLE_FEATURES,
    'output_column': OUTPUT_COLUMN_FEATURES,
}


def get_feature_mapping(logical_name: str) -> FeatureMapping:
    """
    Get feature mapping by logical name.
    
    Args:
        logical_name: Logical feature name
    
    Returns:
        FeatureMapping instance
    
    Raises:
        KeyError: If feature not found
    """
    if logical_name not in FEATURE_REGISTRY:
        raise KeyError(
            f"Feature '{logical_name}' not found in registry. "
            f"Available: {list(FEATURE_REGISTRY.keys())}"
        )
    return FEATURE_REGISTRY[logical_name]


def list_features(feature_type: FeatureType = None) -> list:
    """
    List all registered features.
    
    Args:
        feature_type: Optional filter by type
    
    Returns:
        List of logical feature names
    """
    if feature_type is None:
        return list(FEATURE_REGISTRY.keys())
    
    return [
        name for name, mapping in FEATURE_REGISTRY.items()
        if mapping.feature_type == feature_type
    ]


# ============================================================================
# PLUGIN-BASED EXTENSION API
# ============================================================================

def register_dbms_aliases(dbms_name: str, feature_aliases: dict) -> None:
    """
    Register additional feature aliases for a DBMS via plugin.
    
    This allows plugins to extend the feature mapping without editing this file.
    
    Args:
        dbms_name: DBMS name (e.g., "mysql", "oracle")
        feature_aliases: Dict mapping logical_name → dbms_specific_name
                        e.g., {"operator_type": "operation_name", ...}
    
    Example:
        # In MySQLPlugin.get_feature_aliases():
        return {
            "operator_type": "select_type",
            "estimated_cardinality": "rows_estimate",
            ...
        }
        
        # Then during plugin initialization:
        register_dbms_aliases("mysql", plugin.get_feature_aliases())
    """
    for logical_name, dbms_attr in feature_aliases.items():
        if logical_name in FEATURE_REGISTRY:
            mapping = FEATURE_REGISTRY[logical_name]
            # Add or update the alias for this DBMS
            mapping.dbms_aliases[dbms_name] = dbms_attr
        else:
            # Warn about unknown logical feature
            print(f"Warning: Logical feature '{logical_name}' not found in registry. "
                  f"Skipping alias registration for {dbms_name}.")


def get_registered_dbms() -> list:
    """
    Get list of DBMS that have registered feature aliases.
    
    Returns:
        List of DBMS names with at least one feature alias
    """
    all_dbms = set()
    for mapping in FEATURE_REGISTRY.values():
        all_dbms.update(mapping.dbms_aliases.keys())
    return sorted(all_dbms)
