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
        dbms_aliases={
            "postgres": "Node Type",
            "trino": "op_name",
            "mysql": "select_type",
        },
        default_value="unknown",
        transformation=normalize_operator_name,
        description="Type of plan operator (e.g., SeqScan, HashJoin)"
    ),
    
    FeatureMapping(
        logical_name="estimated_cardinality",
        feature_type=FeatureType.PLAN_CARDINALITY,
        dbms_aliases={
            "postgres": "Plan Rows",
            "trino": "est_rows",
            "mysql": "rows",
        },
        default_value=1.0,
        transformation=to_positive_float,
        min_value=0.0,
        description="Estimated number of rows output by this operator"
    ),
    
    FeatureMapping(
        logical_name="actual_cardinality",
        feature_type=FeatureType.PLAN_CARDINALITY,
        dbms_aliases={
            "postgres": "Actual Rows",
            "trino": "act_rows",
            "mysql": "rows",
        },
        default_value=None,
        transformation=to_positive_float,
        min_value=0.0,
        description="Actual number of rows (from EXPLAIN ANALYZE)"
    ),
    
    FeatureMapping(
        logical_name="estimated_cost",
        feature_type=FeatureType.PLAN_COST,
        dbms_aliases={
            "postgres": "Total Cost",
            "trino": "est_cost",
            "mysql": "cost",
        },
        default_value=0.0,
        transformation=to_positive_float,
        min_value=0.0,
        description="Estimated total cost of operator"
    ),
    
    FeatureMapping(
        logical_name="startup_cost",
        feature_type=FeatureType.PLAN_COST,
        dbms_aliases={
            "postgres": "Startup Cost",
            "trino": "startup_cost",
        },
        default_value=0.0,
        transformation=to_positive_float,
        min_value=0.0,
        description="Estimated startup cost"
    ),
    
    FeatureMapping(
        logical_name="estimated_width",
        feature_type=FeatureType.PLAN_WIDTH,
        dbms_aliases={
            "postgres": "Plan Width",
            "trino": "est_width",
        },
        default_value=0,
        transformation=to_int,
        min_value=0,
        description="Estimated average row width in bytes"
    ),
    
    FeatureMapping(
        logical_name="workers_planned",
        feature_type=FeatureType.PLAN_PARALLELISM,
        dbms_aliases={
            "postgres": "Workers Planned",
            "trino": "workers_planned",
        },
        default_value=0,
        transformation=to_int,
        min_value=0,
        description="Number of parallel workers planned"
    ),
    
    FeatureMapping(
        logical_name="workers_launched",
        feature_type=FeatureType.PLAN_PARALLELISM,
        dbms_aliases={
            "postgres": "Workers Launched",
            "trino": "workers_launched",
        },
        default_value=0,
        transformation=to_int,
        min_value=0,
        description="Number of parallel workers actually launched"
    ),
    
    # PostgreSQL-specific but included for compatibility
    FeatureMapping(
        logical_name="actual_children_cardinality",
        feature_type=FeatureType.PLAN_CARDINALITY,
        dbms_aliases={
            "postgres": "act_children_card",
            "trino": "act_children_card",
        },
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
        dbms_aliases={
            "postgres": "operator",
            "trino": "operator",
        },
        default_value="=",
        description="Comparison operator (=, <, >, LIKE, etc.)"
    ),
    
    FeatureMapping(
        logical_name="literal_feature",
        feature_type=FeatureType.FILTER_LITERAL,
        dbms_aliases={
            "postgres": "literal_feature",
            "trino": "literal_feature",
        },
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
        dbms_aliases={
            "postgres": "avg_width",
            "trino": "avg_width",
        },
        default_value=8,
        transformation=to_int,
        min_value=0,
        description="Average column width in bytes"
    ),
    
    FeatureMapping(
        logical_name="correlation",
        feature_type=FeatureType.COLUMN_STATISTICS,
        dbms_aliases={
            "postgres": "correlation",
            # Trino doesn't provide correlation
        },
        default_value=0.0,
        transformation=to_float,
        min_value=-1.0,
        max_value=1.0,
        description="Physical storage correlation (-1 to 1)"
    ),
    
    FeatureMapping(
        logical_name="data_type",
        feature_type=FeatureType.COLUMN_TYPE,
        dbms_aliases={
            "postgres": "data_type",
            "trino": "data_type",
        },
        default_value="unknown",
        description="Column data type"
    ),
    
    FeatureMapping(
        logical_name="n_distinct",
        feature_type=FeatureType.COLUMN_STATISTICS,
        dbms_aliases={
            "postgres": "n_distinct",
            "trino": "distinct_count",
        },
        default_value=1.0,
        transformation=to_positive_float,
        min_value=0.0,
        description="Number of distinct values"
    ),
    
    FeatureMapping(
        logical_name="null_frac",
        feature_type=FeatureType.COLUMN_STATISTICS,
        dbms_aliases={
            "postgres": "null_frac",
            "trino": "null_fraction",
        },
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
        dbms_aliases={
            "postgres": "reltuples",
            "trino": "row_count",
            "mysql": "table_rows",
        },
        default_value=0.0,
        transformation=to_positive_float,
        min_value=0.0,
        description="Total number of rows in table"
    ),
    
    FeatureMapping(
        logical_name="page_count",
        feature_type=FeatureType.TABLE_STATISTICS,
        dbms_aliases={
            "postgres": "relpages",
            # Trino doesn't provide page counts
        },
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
        dbms_aliases={
            "postgres": "aggregation",
            "trino": "aggregation",
        },
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

