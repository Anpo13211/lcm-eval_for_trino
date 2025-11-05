"""
Standardized statistics schema (DBMS-agnostic)

This module defines a unified representation of database statistics
that is independent of any specific DBMS implementation.

All DBMS-specific statistics must be converted to this standard format
before being consumed by models.

Design principles:
1. Nullable fields - not all DBMS provide all statistics
2. Logical naming - use domain concepts, not DBMS-specific terminology
3. Type safety - use enums and typed fields where possible
4. Extensible - allow DBMS-specific extensions via metadata dict
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Tuple
from enum import Enum


class DataType(Enum):
    """
    Logical data types (DBMS-independent).
    
    Maps DBMS-specific types to logical categories:
    - PostgreSQL: integer, bigint, text, varchar, timestamp, etc.
    - Trino: integer, bigint, varchar, timestamp, etc.
    - MySQL: int, bigint, varchar, datetime, etc.
    """
    # Numeric types
    INTEGER = "integer"
    BIGINT = "bigint"
    SMALLINT = "smallint"
    NUMERIC = "numeric"
    DECIMAL = "decimal"
    REAL = "real"
    DOUBLE = "double"
    
    # String types
    TEXT = "text"
    VARCHAR = "varchar"
    CHAR = "char"
    
    # Date/time types
    DATE = "date"
    TIME = "time"
    TIMESTAMP = "timestamp"
    INTERVAL = "interval"
    
    # Boolean
    BOOLEAN = "boolean"
    
    # Binary
    BYTEA = "bytea"
    
    # Special types
    JSON = "json"
    ARRAY = "array"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_postgres(cls, pg_type: str) -> 'DataType':
        """Convert PostgreSQL type to logical type."""
        mapping = {
            'integer': cls.INTEGER,
            'bigint': cls.BIGINT,
            'smallint': cls.SMALLINT,
            'numeric': cls.NUMERIC,
            'decimal': cls.DECIMAL,
            'real': cls.REAL,
            'double precision': cls.DOUBLE,
            'text': cls.TEXT,
            'character varying': cls.VARCHAR,
            'varchar': cls.VARCHAR,
            'character': cls.CHAR,
            'date': cls.DATE,
            'time': cls.TIME,
            'timestamp': cls.TIMESTAMP,
            'timestamp without time zone': cls.TIMESTAMP,
            'interval': cls.INTERVAL,
            'boolean': cls.BOOLEAN,
            'bytea': cls.BYTEA,
            'json': cls.JSON,
            'jsonb': cls.JSON,
        }
        return mapping.get(pg_type.lower(), cls.UNKNOWN)
    
    @classmethod
    def from_trino(cls, trino_type: str) -> 'DataType':
        """Convert Trino type to logical type."""
        mapping = {
            'integer': cls.INTEGER,
            'bigint': cls.BIGINT,
            'smallint': cls.SMALLINT,
            'decimal': cls.DECIMAL,
            'real': cls.REAL,
            'double': cls.DOUBLE,
            'varchar': cls.VARCHAR,
            'char': cls.CHAR,
            'date': cls.DATE,
            'time': cls.TIME,
            'timestamp': cls.TIMESTAMP,
            'boolean': cls.BOOLEAN,
            'varbinary': cls.BYTEA,
            'json': cls.JSON,
            'array': cls.ARRAY,
        }
        return mapping.get(trino_type.lower(), cls.UNKNOWN)
    
    def is_numeric(self) -> bool:
        """Check if type is numeric."""
        return self in {
            self.INTEGER, self.BIGINT, self.SMALLINT,
            self.NUMERIC, self.DECIMAL, self.REAL, self.DOUBLE
        }
    
    def is_string(self) -> bool:
        """Check if type is string-like."""
        return self in {self.TEXT, self.VARCHAR, self.CHAR}
    
    def is_temporal(self) -> bool:
        """Check if type is temporal."""
        return self in {self.DATE, self.TIME, self.TIMESTAMP, self.INTERVAL}


@dataclass
class ColumnStats:
    """
    Standardized column statistics.
    
    All fields are Optional because different DBMS provide different statistics.
    Models should gracefully handle missing values.
    
    Field descriptions:
    - row_count: Total number of rows in the table (redundant with TableStats, but convenient)
    - distinct_count: Number of distinct values (cardinality)
    - null_fraction: Fraction of NULL values (0.0 to 1.0)
    - avg_width: Average storage width in bytes
    - data_type: Logical data type
    - min_value: Minimum value (for numeric/temporal types)
    - max_value: Maximum value (for numeric/temporal types)
    - correlation: Physical storage correlation (-1.0 to 1.0, PostgreSQL-specific)
    - most_common_values: List of most common values (PostgreSQL MCV)
    - most_common_freqs: Frequencies of most common values
    """
    
    # Basic statistics (available in most DBMS)
    row_count: Optional[float] = None
    distinct_count: Optional[float] = None  # n_distinct in PostgreSQL
    null_fraction: Optional[float] = None
    avg_width: Optional[int] = None
    data_type: Optional[DataType] = None
    
    # Extended statistics (may not be available in all DBMS)
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    correlation: Optional[float] = None  # PostgreSQL-specific
    
    # Histogram/distribution (PostgreSQL-specific)
    most_common_values: Optional[list] = None
    most_common_freqs: Optional[list] = None
    histogram_bounds: Optional[list] = None
    
    # DBMS-specific metadata (extensibility point)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Check if statistics have minimum required fields."""
        return (
            self.row_count is not None and
            self.distinct_count is not None and
            self.null_fraction is not None and
            self.data_type is not None
        )
    
    def get_effective_distinct_count(self) -> float:
        """
        Get effective distinct count (accounting for nulls).
        
        Returns:
            Distinct count or fallback estimate
        """
        if self.distinct_count is not None:
            return max(1.0, float(self.distinct_count))
        
        # Fallback: estimate from row_count and null_fraction
        if self.row_count is not None and self.null_fraction is not None:
            non_null_rows = self.row_count * (1.0 - self.null_fraction)
            return max(1.0, non_null_rows * 0.1)  # Assume 10% distinct
        
        return 1.0
    
    def get_selectivity(self, value: Any = None) -> float:
        """
        Estimate selectivity for equality predicate.
        
        Args:
            value: Optional value to check in MCV list
            
        Returns:
            Estimated selectivity (0.0 to 1.0)
        """
        if self.distinct_count is not None and self.row_count is not None:
            if self.distinct_count > 0:
                return 1.0 / self.distinct_count
        
        return 0.01  # Default 1% selectivity


@dataclass
class TableStats:
    """
    Standardized table statistics.
    
    Field descriptions:
    - row_count: Total number of rows (tuples)
    - page_count: Number of storage pages
    - avg_row_width: Average row width in bytes
    - size_bytes: Total table size in bytes
    - index_size_bytes: Total index size in bytes
    """
    
    # Basic statistics
    row_count: float
    page_count: Optional[int] = None  # relpages in PostgreSQL
    avg_row_width: Optional[int] = None
    
    # Storage statistics
    size_bytes: Optional[int] = None
    index_size_bytes: Optional[int] = None
    
    # DBMS-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Check if statistics have minimum required fields."""
        return self.row_count is not None
    
    def get_estimated_size_mb(self) -> float:
        """
        Estimate table size in MB.
        
        Returns:
            Size in megabytes
        """
        if self.size_bytes is not None:
            return self.size_bytes / (1024 * 1024)
        
        # Estimate from page_count (assuming 8KB pages like PostgreSQL)
        if self.page_count is not None:
            return (self.page_count * 8192) / (1024 * 1024)
        
        # Estimate from row_count and avg_row_width
        if self.row_count is not None and self.avg_row_width is not None:
            estimated_bytes = self.row_count * self.avg_row_width
            return estimated_bytes / (1024 * 1024)
        
        return 0.0


@dataclass
class StandardizedStatistics:
    """
    Complete standardized statistics for a database.
    
    This is the canonical format that all models consume.
    DBMS plugins must convert their native statistics to this format.
    
    Structure:
    - column_stats: {(table_name, column_name): ColumnStats}
    - table_stats: {table_name: TableStats}
    - metadata: Additional database-level information
    
    Usage:
        # Get statistics for a specific column
        stats = standardized_stats.column_stats.get(("users", "id"))
        if stats and stats.is_complete():
            selectivity = stats.get_selectivity()
        
        # Get statistics for a table
        table = standardized_stats.table_stats.get("users")
        if table:
            size_mb = table.get_estimated_size_mb()
    """
    
    column_stats: Dict[Tuple[str, str], ColumnStats] = field(default_factory=dict)
    table_stats: Dict[str, TableStats] = field(default_factory=dict)
    
    # Database-level metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_column_stats(self, table: str, column: str) -> Optional[ColumnStats]:
        """
        Get column statistics with fallback to table-level info.
        
        Args:
            table: Table name
            column: Column name
            
        Returns:
            ColumnStats if available, None otherwise
        """
        return self.column_stats.get((table, column))
    
    def get_table_stats(self, table: str) -> Optional[TableStats]:
        """
        Get table statistics.
        
        Args:
            table: Table name
            
        Returns:
            TableStats if available, None otherwise
        """
        return self.table_stats.get(table)
    
    def has_column_stats(self, table: str, column: str) -> bool:
        """Check if column statistics are available."""
        return (table, column) in self.column_stats
    
    def has_table_stats(self, table: str) -> bool:
        """Check if table statistics are available."""
        return table in self.table_stats
    
    def validate(self) -> bool:
        """
        Validate that statistics are well-formed.
        
        Returns:
            True if valid, False otherwise
        """
        # Check that all tables referenced in column_stats have table_stats
        referenced_tables = set(table for table, _ in self.column_stats.keys())
        
        for table in referenced_tables:
            if table not in self.table_stats:
                print(f"Warning: Column stats reference table '{table}' with no table stats")
                return False
        
        # Check that all column stats have valid row counts
        for (table, column), col_stats in self.column_stats.items():
            table_stats = self.table_stats.get(table)
            
            if col_stats.row_count is not None and table_stats is not None:
                if abs(col_stats.row_count - table_stats.row_count) > 1e-6:
                    print(
                        f"Warning: Row count mismatch for {table}.{column}: "
                        f"column={col_stats.row_count}, table={table_stats.row_count}"
                    )
        
        return True
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate a summary of available statistics.
        
        Returns:
            Dictionary with summary information
        """
        return {
            'num_tables': len(self.table_stats),
            'num_columns': len(self.column_stats),
            'tables': list(self.table_stats.keys()),
            'complete_columns': sum(
                1 for stats in self.column_stats.values() if stats.is_complete()
            ),
            'total_rows': sum(
                stats.row_count for stats in self.table_stats.values()
            ),
        }

