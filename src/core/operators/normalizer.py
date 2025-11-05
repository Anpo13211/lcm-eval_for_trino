"""
Operator normalizer - converts DBMS-specific operators to logical types

This module standardizes operator names across different DBMS:
- PostgreSQL: "Seq Scan", "Parallel Seq Scan"
- Trino: "TableScan"
- MySQL: "table scan"

All map to LogicalOperator.TABLE_SCAN with metadata about parallelism.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
from abc import ABC, abstractmethod


class LogicalOperator(Enum):
    """
    Logical operator types (DBMS-independent).
    
    These represent the semantic operation being performed,
    regardless of DBMS-specific implementation details.
    """
    # Scan operators
    TABLE_SCAN = "TableScan"
    INDEX_SCAN = "IndexScan"
    INDEX_ONLY_SCAN = "IndexOnlyScan"
    BITMAP_SCAN = "BitmapScan"
    
    # Join operators
    HASH_JOIN = "HashJoin"
    MERGE_JOIN = "MergeJoin"
    NESTED_LOOP_JOIN = "NestedLoopJoin"
    
    # Aggregate operators
    AGGREGATE = "Aggregate"
    HASH_AGGREGATE = "HashAggregate"
    
    # Sort operators
    SORT = "Sort"
    
    # Utility operators
    HASH = "Hash"
    MATERIALIZE = "Materialize"
    LIMIT = "Limit"
    
    # Parallel coordination
    GATHER = "Gather"
    GATHER_MERGE = "GatherMerge"
    
    # Set operations
    UNION = "Union"
    INTERSECT = "Intersect"
    EXCEPT = "Except"
    
    # Other
    RESULT = "Result"
    SUBQUERY_SCAN = "SubqueryScan"
    CTE_SCAN = "CTEScan"
    UNKNOWN = "Unknown"


@dataclass
class NormalizedOperator:
    """
    Normalized operator with metadata.
    
    Fields:
    - logical_type: Logical operator type
    - original_name: Original DBMS-specific name
    - parallel: Whether this is a parallel variant
    - metadata: Additional operator-specific metadata
    """
    logical_type: LogicalOperator
    original_name: str
    parallel: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OperatorNormalizer(ABC):
    """
    Abstract base class for operator normalization.
    
    Each DBMS implements this to convert its specific operator names
    to LogicalOperator types.
    """
    
    def __init__(self, dbms_name: str):
        """
        Initialize normalizer.
        
        Args:
            dbms_name: Name of DBMS
        """
        self.dbms_name = dbms_name
    
    @abstractmethod
    def normalize(self, operator_name: str) -> NormalizedOperator:
        """
        Normalize DBMS-specific operator name to logical type.
        
        Args:
            operator_name: DBMS-specific operator name
        
        Returns:
            NormalizedOperator instance
        """
        pass
    
    def normalize_batch(self, operator_names: list) -> list:
        """
        Normalize multiple operator names.
        
        Args:
            operator_names: List of operator names
        
        Returns:
            List of NormalizedOperator instances
        """
        return [self.normalize(name) for name in operator_names]


class PostgreSQLOperatorNormalizer(OperatorNormalizer):
    """
    Normalizer for PostgreSQL operators.
    
    Handles PostgreSQL-specific naming conventions:
    - "Parallel" prefix for parallel operations
    - "Bitmap Heap Scan" and "Bitmap Index Scan" as separate operators
    - Various join types with "Join" suffix
    """
    
    # Mapping from PostgreSQL operator names to logical types
    OPERATOR_MAP = {
        # Scan operations
        "Seq Scan": LogicalOperator.TABLE_SCAN,
        "Parallel Seq Scan": LogicalOperator.TABLE_SCAN,
        "Index Scan": LogicalOperator.INDEX_SCAN,
        "Index Only Scan": LogicalOperator.INDEX_ONLY_SCAN,
        "Parallel Index Scan": LogicalOperator.INDEX_SCAN,
        "Parallel Index Only Scan": LogicalOperator.INDEX_ONLY_SCAN,
        "Bitmap Heap Scan": LogicalOperator.BITMAP_SCAN,
        "Parallel Bitmap Heap Scan": LogicalOperator.BITMAP_SCAN,
        "Bitmap Index Scan": LogicalOperator.BITMAP_SCAN,
        
        # Join operations
        "Hash Join": LogicalOperator.HASH_JOIN,
        "Merge Join": LogicalOperator.MERGE_JOIN,
        "Nested Loop": LogicalOperator.NESTED_LOOP_JOIN,
        "Parallel Hash Join": LogicalOperator.HASH_JOIN,
        
        # Aggregate operations
        "Aggregate": LogicalOperator.AGGREGATE,
        "HashAggregate": LogicalOperator.HASH_AGGREGATE,
        "Partial Aggregate": LogicalOperator.AGGREGATE,
        "Finalize Aggregate": LogicalOperator.AGGREGATE,
        
        # Sort
        "Sort": LogicalOperator.SORT,
        
        # Utility
        "Hash": LogicalOperator.HASH,
        "Materialize": LogicalOperator.MATERIALIZE,
        "Limit": LogicalOperator.LIMIT,
        
        # Parallel coordination
        "Gather": LogicalOperator.GATHER,
        "Gather Merge": LogicalOperator.GATHER_MERGE,
        
        # Set operations
        "Append": LogicalOperator.UNION,
        
        # Other
        "Result": LogicalOperator.RESULT,
        "Subquery Scan": LogicalOperator.SUBQUERY_SCAN,
        "CTE Scan": LogicalOperator.CTE_SCAN,
    }
    
    def __init__(self):
        super().__init__("postgres")
    
    def normalize(self, operator_name: str) -> NormalizedOperator:
        """
        Normalize PostgreSQL operator name.
        
        Args:
            operator_name: PostgreSQL operator name
        
        Returns:
            NormalizedOperator instance
        """
        if operator_name is None:
            return NormalizedOperator(
                logical_type=LogicalOperator.UNKNOWN,
                original_name="",
            )
        
        # Check for parallel prefix
        is_parallel = operator_name.startswith("Parallel ")
        
        # Look up in mapping
        logical_type = self.OPERATOR_MAP.get(operator_name, LogicalOperator.UNKNOWN)
        
        metadata = {}
        if is_parallel:
            metadata['parallel'] = True
        
        # Detect partial/finalize aggregates
        if "Partial" in operator_name:
            metadata['aggregate_phase'] = 'partial'
        elif "Finalize" in operator_name:
            metadata['aggregate_phase'] = 'finalize'
        
        return NormalizedOperator(
            logical_type=logical_type,
            original_name=operator_name,
            parallel=is_parallel,
            metadata=metadata
        )


class TrinoOperatorNormalizer(OperatorNormalizer):
    """
    Normalizer for Trino operators.
    
    Trino uses different naming conventions:
    - TableScan instead of "Seq Scan"
    - ScanProject instead of separate operators
    - Exchange for distributed operations
    """
    
    OPERATOR_MAP = {
        # Scan operations
        "TableScan": LogicalOperator.TABLE_SCAN,
        "ScanProject": LogicalOperator.TABLE_SCAN,
        "IndexScan": LogicalOperator.INDEX_SCAN,
        
        # Join operations
        "Join": LogicalOperator.HASH_JOIN,  # Default join type
        "HashJoin": LogicalOperator.HASH_JOIN,
        "MergeJoin": LogicalOperator.MERGE_JOIN,
        "NestedLoopJoin": LogicalOperator.NESTED_LOOP_JOIN,
        
        # Aggregate
        "Aggregate": LogicalOperator.AGGREGATE,
        "PartialAggregate": LogicalOperator.AGGREGATE,
        "FinalAggregate": LogicalOperator.AGGREGATE,
        
        # Sort
        "Sort": LogicalOperator.SORT,
        "TopN": LogicalOperator.SORT,
        
        # Utility
        "Project": LogicalOperator.RESULT,
        "Filter": LogicalOperator.RESULT,
        "Limit": LogicalOperator.LIMIT,
        
        # Distributed operations (Trino-specific)
        "Exchange": LogicalOperator.GATHER,
        "RemoteSource": LogicalOperator.GATHER,
        
        # Set operations
        "Union": LogicalOperator.UNION,
        
        # Other
        "Output": LogicalOperator.RESULT,
        "Values": LogicalOperator.RESULT,
    }
    
    def __init__(self):
        super().__init__("trino")
    
    def normalize(self, operator_name: str) -> NormalizedOperator:
        """
        Normalize Trino operator name.
        
        Args:
            operator_name: Trino operator name
        
        Returns:
            NormalizedOperator instance
        """
        if operator_name is None:
            return NormalizedOperator(
                logical_type=LogicalOperator.UNKNOWN,
                original_name="",
            )
        
        # Look up in mapping
        logical_type = self.OPERATOR_MAP.get(operator_name, LogicalOperator.UNKNOWN)
        
        metadata = {}
        
        # Detect aggregate phases
        if "Partial" in operator_name:
            metadata['aggregate_phase'] = 'partial'
        elif "Final" in operator_name:
            metadata['aggregate_phase'] = 'final'
        
        # Detect distributed operations
        if operator_name in ("Exchange", "RemoteSource"):
            metadata['distributed'] = True
        
        return NormalizedOperator(
            logical_type=logical_type,
            original_name=operator_name,
            parallel=False,  # Trino doesn't use "Parallel" prefix
            metadata=metadata
        )

