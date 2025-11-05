"""
Unified featurization definitions using logical feature names

This module replaces DBMS-specific featurizations (e.g., PostgresTrueCardDetail)
with logical feature names that work across all DBMS.

Key changes:
- Use logical feature names instead of DBMS-specific names
- Features are automatically mapped to DBMS-specific attributes via FeatureMapper
- Single featurization works for PostgreSQL, Trino, MySQL, etc.

Example:
    Before:
        class PostgresTrueCardDetail(Featurization):
            PLAN_FEATURES = ['Plan Rows', 'Node Type', ...]  # PostgreSQL-specific
        
        class TrinoTrueCardDetail(Featurization):
            PLAN_FEATURES = ['est_rows', 'op_name', ...]  # Trino-specific
    
    After:
        class UnifiedTrueCardDetail(Featurization):
            PLAN_FEATURES = ['estimated_cardinality', 'operator_type', ...]  # Logical names
        
        # Works for both PostgreSQL and Trino!
"""

from attr import field


class Featurization:
    """Base featurization class."""
    PLAN_FEATURES = field()
    FILTER_FEATURES = field()
    COLUMN_FEATURES = field()
    OUTPUT_COLUMN_FEATURES = field()
    TABLE_FEATURES = field()


class UnifiedTrueCardDetail(Featurization):
    """
    Detailed true cardinality model with unified logical features.
    
    This replaces PostgresTrueCardDetail and works across all DBMS.
    """
    PLAN_FEATURES = [
        'actual_cardinality',           # PostgreSQL: 'Actual Rows', Trino: 'act_rows'
        'estimated_width',               # PostgreSQL: 'Plan Width', Trino: 'est_width'
        'workers_planned',               # PostgreSQL: 'Workers Planned', Trino: 'workers_planned'
        'operator_type',                 # PostgreSQL: 'Node Type', Trino: 'op_name'
        'actual_children_cardinality',   # PostgreSQL: 'act_children_card', Trino: 'act_children_card'
    ]
    
    FILTER_FEATURES = [
        'filter_operator',               # PostgreSQL: 'operator', Trino: 'operator'
        'literal_feature',               # PostgreSQL: 'literal_feature', Trino: 'literal_feature'
    ]
    
    COLUMN_FEATURES = [
        'avg_width',                     # PostgreSQL: 'avg_width', Trino: 'avg_width'
        'correlation',                   # PostgreSQL: 'correlation', Trino: None (defaults to 0.0)
        'data_type',                     # PostgreSQL: 'data_type', Trino: 'data_type'
        'n_distinct',                    # PostgreSQL: 'n_distinct', Trino: 'distinct_count'
        'null_frac',                     # PostgreSQL: 'null_frac', Trino: 'null_fraction'
    ]
    
    OUTPUT_COLUMN_FEATURES = [
        'aggregation',                   # PostgreSQL: 'aggregation', Trino: 'aggregation'
    ]
    
    TABLE_FEATURES = [
        'row_count',                     # PostgreSQL: 'reltuples', Trino: 'row_count'
        'page_count',                    # PostgreSQL: 'relpages', Trino: None (defaults to 0)
    ]
    
    VARIABLES = {
        "column": COLUMN_FEATURES,
        "table": TABLE_FEATURES,
        "output_column": OUTPUT_COLUMN_FEATURES,
        "filter_column": FILTER_FEATURES + COLUMN_FEATURES,
        "plan": PLAN_FEATURES,
        "logical_pred": FILTER_FEATURES,
    }


class UnifiedTrueCardMedium(Featurization):
    """
    Medium complexity true cardinality model.
    
    Fewer features than Detail version for faster training.
    """
    PLAN_FEATURES = [
        'actual_cardinality',
        'estimated_width',
        'operator_type'
    ]
    
    FILTER_FEATURES = [
        'filter_operator',
        'literal_feature'
    ]
    
    COLUMN_FEATURES = [
        'avg_width',
        'data_type',
        'n_distinct'
    ]
    
    OUTPUT_COLUMN_FEATURES = [
        'aggregation'
    ]
    
    TABLE_FEATURES = [
        'row_count',
        'page_count'
    ]


class UnifiedTrueCardCoarse(Featurization):
    """
    Coarse true cardinality model with minimal features.
    """
    PLAN_FEATURES = [
        'actual_cardinality',
        'estimated_width',
        'workers_planned',
        'operator_type'
    ]
    
    FILTER_FEATURES = [
        'filter_operator',
        'literal_feature'
    ]
    
    COLUMN_FEATURES = [
        'avg_width',
        'data_type'
    ]
    
    OUTPUT_COLUMN_FEATURES = [
        'aggregation'
    ]
    
    TABLE_FEATURES = [
        'row_count',
        'page_count'
    ]


class UnifiedEstSystemCardDetail(Featurization):
    """
    Estimated system cardinality model using optimizer estimates.
    
    Uses estimated cardinality instead of actual (for workload-agnostic prediction).
    """
    PLAN_FEATURES = [
        'estimated_cardinality',         # Use estimates instead of actuals
        'estimated_width',
        'workers_planned',
        'operator_type',
        'estimated_cost',                # Include cost estimates
    ]
    
    FILTER_FEATURES = [
        'filter_operator',
        'literal_feature'
    ]
    
    COLUMN_FEATURES = [
        'avg_width',
        'correlation',
        'data_type',
        'n_distinct',
        'null_frac'
    ]
    
    OUTPUT_COLUMN_FEATURES = [
        'aggregation'
    ]
    
    TABLE_FEATURES = [
        'row_count',
        'page_count'
    ]
    
    VARIABLES = {
        "column": COLUMN_FEATURES,
        "table": TABLE_FEATURES,
        "output_column": OUTPUT_COLUMN_FEATURES,
        "filter_column": FILTER_FEATURES + COLUMN_FEATURES,
        "plan": PLAN_FEATURES,
        "logical_pred": FILTER_FEATURES,
    }


class UnifiedDACEFeaturization(Featurization):
    """
    DACE model featurization with logical feature names.
    
    Simplified features for the DACE (Deep Auto-regressive Cost Estimator) model.
    """
    PLAN_FEATURES = [
        'operator_type',
        'estimated_cost',
        'estimated_cardinality'
    ]
    
    # DACE uses fewer auxiliary features
    FILTER_FEATURES = []
    COLUMN_FEATURES = []
    OUTPUT_COLUMN_FEATURES = []
    TABLE_FEATURES = []


class UnifiedQueryFormerFeaturization(Featurization):
    """
    QueryFormer model featurization with logical feature names.
    
    QueryFormer uses a transformer-based architecture with specific features.
    """
    PLAN_FEATURES = [
        'operator_type',
        'estimated_cardinality',
        'estimated_cost',
        'estimated_width'
    ]
    
    FILTER_FEATURES = [
        'filter_operator',
        'literal_feature'
    ]
    
    COLUMN_FEATURES = [
        'data_type',
        'n_distinct',
        'null_frac'
    ]
    
    OUTPUT_COLUMN_FEATURES = [
        'aggregation'
    ]
    
    TABLE_FEATURES = [
        'row_count'
    ]


# Mapping from old featurization names to new unified ones
FEATURIZATION_MIGRATION_MAP = {
    'PostgresTrueCardDetail': UnifiedTrueCardDetail,
    'PostgresTrueCardMedium': UnifiedTrueCardMedium,
    'PostgresTrueCardCoarse': UnifiedTrueCardCoarse,
    'PostgresEstSystemCardDetail': UnifiedEstSystemCardDetail,
    'DACEFeaturization': UnifiedDACEFeaturization,
    'QueryFormerFeaturization': UnifiedQueryFormerFeaturization,
}


def get_unified_featurization(name: str) -> Featurization:
    """
    Get unified featurization by name.
    
    Args:
        name: Featurization name (old or new)
    
    Returns:
        Featurization class
    
    Raises:
        ValueError: If featurization not found
    """
    # Try new naming convention first
    unified_classes = {
        'UnifiedTrueCardDetail': UnifiedTrueCardDetail,
        'UnifiedTrueCardMedium': UnifiedTrueCardMedium,
        'UnifiedTrueCardCoarse': UnifiedTrueCardCoarse,
        'UnifiedEstSystemCardDetail': UnifiedEstSystemCardDetail,
        'UnifiedDACEFeaturization': UnifiedDACEFeaturization,
        'UnifiedQueryFormerFeaturization': UnifiedQueryFormerFeaturization,
    }
    
    if name in unified_classes:
        return unified_classes[name]
    
    # Try migration map
    if name in FEATURIZATION_MIGRATION_MAP:
        return FEATURIZATION_MIGRATION_MAP[name]
    
    raise ValueError(
        f"Featurization '{name}' not found. "
        f"Available: {list(unified_classes.keys())}"
    )


def migrate_featurization(old_featurization) -> Featurization:
    """
    Migrate old DBMS-specific featurization to unified version.
    
    Args:
        old_featurization: Old featurization class or instance
    
    Returns:
        Unified featurization class
    """
    # Get class name
    if isinstance(old_featurization, type):
        class_name = old_featurization.__name__
    else:
        class_name = old_featurization.__class__.__name__
    
    # Look up in migration map
    if class_name in FEATURIZATION_MIGRATION_MAP:
        return FEATURIZATION_MIGRATION_MAP[class_name]
    
    # If already unified, return as-is
    if class_name.startswith('Unified'):
        return old_featurization if isinstance(old_featurization, type) else old_featurization.__class__
    
    # Unknown featurization
    raise ValueError(
        f"Cannot migrate featurization '{class_name}'. "
        f"Please create a unified version manually."
    )

