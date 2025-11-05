"""
Feature mapping definitions

This module defines the mapping between logical feature names (what models expect)
and DBMS-specific attribute names (what plan parsers provide).

Example:
    Logical feature: "estimated_cardinality"
    PostgreSQL: "Plan Rows"
    Trino: "est_rows"
    MySQL: "rows"
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
from enum import Enum


class FeatureType(Enum):
    """
    Type of feature for categorization.
    """
    # Plan node features
    PLAN_OPERATOR = "plan_operator"
    PLAN_CARDINALITY = "plan_cardinality"
    PLAN_COST = "plan_cost"
    PLAN_WIDTH = "plan_width"
    PLAN_PARALLELISM = "plan_parallelism"
    
    # Predicate/filter features
    FILTER_OPERATOR = "filter_operator"
    FILTER_LITERAL = "filter_literal"
    
    # Column features
    COLUMN_STATISTICS = "column_statistics"
    COLUMN_TYPE = "column_type"
    
    # Table features
    TABLE_STATISTICS = "table_statistics"
    
    # Output column features
    OUTPUT_AGGREGATION = "output_aggregation"


@dataclass
class FeatureMapping:
    """
    Mapping from logical feature name to DBMS-specific attributes.
    
    This allows models to request features by logical name (e.g., "estimated_cardinality")
    and have them automatically resolved to DBMS-specific names:
    - PostgreSQL: "Plan Rows"
    - Trino: "est_rows"
    
    Fields:
    - logical_name: Name used by models
    - feature_type: Categorization for grouping
    - dbms_aliases: Dict[dbms_name, actual_attribute_name]
    - default_value: Value to use if attribute missing
    - transformation: Optional function to transform the value
    - description: Human-readable description
    """
    
    logical_name: str
    feature_type: FeatureType
    dbms_aliases: Dict[str, str]  # {dbms_name: attribute_name}
    default_value: Any = None
    transformation: Optional[Callable[[Any], Any]] = None
    description: str = ""
    
    # Optional: value range for validation
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def get_alias(self, dbms_name: str) -> Optional[str]:
        """
        Get DBMS-specific attribute name.
        
        Args:
            dbms_name: DBMS name (e.g., "postgres", "trino")
        
        Returns:
            Attribute name for that DBMS, or None if not defined
        """
        return self.dbms_aliases.get(dbms_name)
    
    def transform(self, value: Any) -> Any:
        """
        Apply transformation to value.
        
        Args:
            value: Raw value from plan
        
        Returns:
            Transformed value
        """
        if value is None:
            return self.default_value
        
        if self.transformation is not None:
            try:
                return self.transformation(value)
            except (ValueError, TypeError):
                return self.default_value
        
        return value
    
    def validate(self, value: Any) -> bool:
        """
        Validate that value is in acceptable range.
        
        Args:
            value: Value to validate
        
        Returns:
            True if valid, False otherwise
        """
        if value is None:
            return True
        
        if self.min_value is not None and value < self.min_value:
            return False
        
        if self.max_value is not None and value > self.max_value:
            return False
        
        return True


# Common transformation functions
def to_float(value: Any) -> float:
    """Convert value to float with error handling."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def to_positive_float(value: Any) -> float:
    """Convert value to positive float."""
    return max(0.0, to_float(value))


def to_int(value: Any) -> int:
    """Convert value to int with error handling."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def normalize_operator_name(value: str) -> str:
    """Normalize operator name to consistent format."""
    if not isinstance(value, str):
        return "unknown"
    
    # Remove extra whitespace
    normalized = " ".join(value.split())
    
    # Common normalizations
    replacements = {
        "Seq Scan": "SeqScan",
        "Index Scan": "IndexScan",
        "Index Only Scan": "IndexOnlyScan",
        "Bitmap Heap Scan": "BitmapHeapScan",
        "Bitmap Index Scan": "BitmapIndexScan",
        "Hash Join": "HashJoin",
        "Merge Join": "MergeJoin",
        "Nested Loop": "NestedLoop",
    }
    
    return replacements.get(normalized, normalized)


def log_transform(value: Any) -> float:
    """Apply log transformation for cardinality features."""
    import math
    val = to_positive_float(value)
    if val <= 0:
        return 0.0
    return math.log(max(1.0, val))

