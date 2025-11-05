"""
Statistics converter interface

Each DBMS plugin implements this interface to convert DBMS-specific
statistics to the standardized format.

Example implementations:
- PostgreSQLStatisticsConverter: Converts pg_stats to StandardizedStatistics
- TrinoStatisticsConverter: Converts Trino SHOW STATS to StandardizedStatistics
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import re

from .schema import StandardizedStatistics, ColumnStats, TableStats, DataType


class StatisticsConverter(ABC):
    """
    Abstract base class for statistics conversion.
    
    Each DBMS implements this to transform native statistics
    into StandardizedStatistics format.
    
    Implementation cost per DBMS: ~100-200 lines
    """
    
    def __init__(self, dbms_name: str):
        """
        Initialize converter.
        
        Args:
            dbms_name: Name of the DBMS (e.g., "postgres", "trino")
        """
        self.dbms_name = dbms_name
    
    @abstractmethod
    def convert(self, raw_statistics: Any) -> StandardizedStatistics:
        """
        Convert DBMS-specific statistics to standardized format.
        
        Args:
            raw_statistics: DBMS-specific statistics object
                - For PostgreSQL: dict with 'column_stats' and 'table_stats'
                - For Trino: dict with similar structure but different schema
                - For others: DBMS-specific format
        
        Returns:
            StandardizedStatistics instance
        """
        pass
    
    @abstractmethod
    def convert_data_type(self, dbms_type: str) -> DataType:
        """
        Convert DBMS-specific data type to logical DataType.
        
        Args:
            dbms_type: DBMS-specific type name
        
        Returns:
            Logical DataType enum value
        """
        pass
    
    def parse_numeric_value(self, value: Any) -> Optional[float]:
        """
        Parse numeric value with fallback for string representations.
        
        Args:
            value: Value to parse (int, float, str, or None)
        
        Returns:
            Parsed float or None
        """
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Handle special values
            if value.lower() in ('null', 'none', ''):
                return None
            
            # Handle negative values with special PostgreSQL notation
            # e.g., "-1" for n_distinct means percentage
            try:
                return float(value)
            except ValueError:
                return None
        
        return None
    
    def normalize_null_fraction(self, null_frac: Any) -> Optional[float]:
        """
        Normalize null fraction to [0.0, 1.0] range.
        
        Args:
            null_frac: Null fraction value
        
        Returns:
            Normalized value between 0.0 and 1.0, or None
        """
        val = self.parse_numeric_value(null_frac)
        if val is None:
            return None
        
        # Clamp to valid range
        return max(0.0, min(1.0, val))
    
    def normalize_distinct_count(
        self,
        distinct_count: Any,
        row_count: Optional[float]
    ) -> Optional[float]:
        """
        Normalize distinct count handling special PostgreSQL semantics.
        
        PostgreSQL uses negative n_distinct to represent percentage:
        - n_distinct > 0: Actual distinct count
        - n_distinct < 0: |n_distinct| = fraction of row_count
        
        Args:
            distinct_count: Raw distinct count value
            row_count: Total row count
        
        Returns:
            Absolute distinct count, or None
        """
        val = self.parse_numeric_value(distinct_count)
        if val is None:
            return None
        
        # Handle PostgreSQL negative n_distinct notation
        if val < 0 and row_count is not None:
            # -0.5 means 50% of rows are distinct
            return abs(val) * row_count
        
        return abs(val)
    
    def create_empty_statistics(self) -> StandardizedStatistics:
        """
        Create empty StandardizedStatistics.
        
        Returns:
            Empty StandardizedStatistics instance
        """
        return StandardizedStatistics(
            column_stats={},
            table_stats={},
            metadata={'dbms': self.dbms_name}
        )


class PostgreSQLStatisticsConverter(StatisticsConverter):
    """
    Converter for PostgreSQL statistics.
    
    Converts from PostgreSQL's pg_stats and pg_class format
    to StandardizedStatistics.
    
    Input format (from PostgresDatabaseConnection.collect_db_statistics):
    {
        'column_stats': {
            (table, column): SimpleNamespace(
                avg_width=...,
                correlation=...,
                data_type=...,
                n_distinct=...,
                null_frac=...,
            )
        },
        'table_stats': {
            table: SimpleNamespace(
                reltuples=...,
                relpages=...,
            )
        }
    }
    """
    
    def __init__(self):
        super().__init__("postgres")
    
    def convert(self, raw_statistics: Any) -> StandardizedStatistics:
        """
        Convert PostgreSQL statistics to standardized format.
        
        Args:
            raw_statistics: Dictionary with 'column_stats' and 'table_stats'
        
        Returns:
            StandardizedStatistics instance
        """
        result = self.create_empty_statistics()
        
        if raw_statistics is None:
            return result
        
        # Handle both dict and object with attributes
        if isinstance(raw_statistics, dict):
            raw_column_stats = raw_statistics.get('column_stats', {})
            raw_table_stats = raw_statistics.get('table_stats', {})
        else:
            raw_column_stats = getattr(raw_statistics, 'column_stats', {})
            raw_table_stats = getattr(raw_statistics, 'table_stats', {})
        
        # Convert table statistics first (needed for row_count)
        for table_name, table_data in raw_table_stats.items():
            result.table_stats[table_name] = self._convert_table_stats(
                table_name, table_data
            )
        
        # Convert column statistics
        for (table, column), column_data in raw_column_stats.items():
            table_row_count = None
            if table in result.table_stats:
                table_row_count = result.table_stats[table].row_count
            
            result.column_stats[(table, column)] = self._convert_column_stats(
                table, column, column_data, table_row_count
            )
        
        return result
    
    def _convert_column_stats(
        self,
        table: str,
        column: str,
        data: Any,
        table_row_count: Optional[float]
    ) -> ColumnStats:
        """Convert single column statistics."""
        # Handle both dict and SimpleNamespace
        if isinstance(data, dict):
            get_val = lambda k, default=None: data.get(k, default)
        else:
            get_val = lambda k, default=None: getattr(data, k, default)
        
        # Get row count
        row_count = table_row_count
        
        # Get distinct count (handle negative values)
        distinct_count = self.normalize_distinct_count(
            get_val('n_distinct'),
            row_count
        )
        
        return ColumnStats(
            row_count=row_count,
            distinct_count=distinct_count,
            null_fraction=self.normalize_null_fraction(get_val('null_frac')),
            avg_width=self.parse_numeric_value(get_val('avg_width')),
            data_type=self.convert_data_type(get_val('data_type', 'unknown')),
            correlation=self.parse_numeric_value(get_val('correlation')),
            metadata={
                'dbms': 'postgres',
                'table': table,
                'column': column,
            }
        )
    
    def _convert_table_stats(self, table: str, data: Any) -> TableStats:
        """Convert single table statistics."""
        # Handle both dict and SimpleNamespace
        if isinstance(data, dict):
            get_val = lambda k, default=None: data.get(k, default)
        else:
            get_val = lambda k, default=None: getattr(data, k, default)
        
        row_count = self.parse_numeric_value(get_val('reltuples', 0)) or 0.0
        page_count_raw = self.parse_numeric_value(get_val('relpages'))
        page_count = int(page_count_raw) if page_count_raw is not None else None
        
        return TableStats(
            row_count=row_count,
            page_count=page_count,
            metadata={
                'dbms': 'postgres',
                'table': table,
            }
        )
    
    def convert_data_type(self, dbms_type: str) -> DataType:
        """Convert PostgreSQL data type to logical type."""
        if dbms_type is None:
            return DataType.UNKNOWN
        return DataType.from_postgres(str(dbms_type))


class TrinoStatisticsConverter(StatisticsConverter):
    """
    Converter for Trino statistics.
    
    Converts from Trino's SHOW STATS format to StandardizedStatistics.
    
    Note: Trino provides fewer statistics than PostgreSQL:
    - No correlation
    - No MCV (most common values)
    - Basic distinct counts and null fractions
    """
    
    def __init__(self):
        super().__init__("trino")
    
    def convert(self, raw_statistics: Any) -> StandardizedStatistics:
        """
        Convert Trino statistics to standardized format.
        
        Args:
            raw_statistics: Dictionary with Trino statistics
        
        Returns:
            StandardizedStatistics instance
        """
        result = self.create_empty_statistics()
        
        if raw_statistics is None:
            return result
        
        # Handle both dict and object with attributes
        if isinstance(raw_statistics, dict):
            raw_column_stats = raw_statistics.get('column_stats', {})
            raw_table_stats = raw_statistics.get('table_stats', {})
        else:
            raw_column_stats = getattr(raw_statistics, 'column_stats', {})
            raw_table_stats = getattr(raw_statistics, 'table_stats', {})
        
        # Convert table statistics
        for table_name, table_data in raw_table_stats.items():
            result.table_stats[table_name] = self._convert_table_stats(
                table_name, table_data
            )
        
        # Convert column statistics
        for (table, column), column_data in raw_column_stats.items():
            table_row_count = None
            if table in result.table_stats:
                table_row_count = result.table_stats[table].row_count
            
            result.column_stats[(table, column)] = self._convert_column_stats(
                table, column, column_data, table_row_count
            )
        
        return result
    
    def _convert_column_stats(
        self,
        table: str,
        column: str,
        data: Any,
        table_row_count: Optional[float]
    ) -> ColumnStats:
        """Convert single column statistics."""
        # Handle both dict and SimpleNamespace
        if isinstance(data, dict):
            get_val = lambda k, default=None: data.get(k, default)
        else:
            get_val = lambda k, default=None: getattr(data, k, default)
        
        return ColumnStats(
            row_count=table_row_count,
            distinct_count=self.parse_numeric_value(get_val('distinct_count')),
            null_fraction=self.normalize_null_fraction(get_val('null_fraction')),
            avg_width=self.parse_numeric_value(get_val('avg_width')),
            data_type=self.convert_data_type(get_val('data_type', 'unknown')),
            correlation=None,  # Trino doesn't provide correlation
            metadata={
                'dbms': 'trino',
                'table': table,
                'column': column,
            }
        )
    
    def _convert_table_stats(self, table: str, data: Any) -> TableStats:
        """Convert single table statistics."""
        # Handle both dict and SimpleNamespace
        if isinstance(data, dict):
            get_val = lambda k, default=None: data.get(k, default)
        else:
            get_val = lambda k, default=None: getattr(data, k, default)
        
        row_count = self.parse_numeric_value(get_val('row_count', 0)) or 0.0
        
        return TableStats(
            row_count=row_count,
            page_count=None,  # Trino doesn't provide page counts
            metadata={
                'dbms': 'trino',
                'table': table,
            }
        )
    
    def convert_data_type(self, dbms_type: str) -> DataType:
        """Convert Trino data type to logical type."""
        if dbms_type is None:
            return DataType.UNKNOWN
        return DataType.from_trino(str(dbms_type))

