"""
DBMS Plugin interface definition

Each DBMS implementation provides:
1. Plan parser (AbstractPlanParser)
2. Statistics converter (converts DBMS-specific stats to StandardizedStatistics)
3. Database connection factory (DatabaseConnection)
"""

from abc import ABC, abstractmethod
from typing import Type, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DBMSPlugin(ABC):
    """
    Base class for DBMS plugins.
    
    Each DBMS (PostgreSQL, Trino, MySQL, etc.) implements this interface
    to provide:
    - Plan parsing capability
    - Statistics conversion to standardized format
    - Database connection management
    
    Implementation cost per DBMS: ~400 lines
    """
    
    name: str  # e.g., "postgres", "trino"
    display_name: str  # e.g., "PostgreSQL", "Trino"
    
    @abstractmethod
    def get_parser(self) -> 'AbstractPlanParser':
        """
        Returns a plan parser for this DBMS.
        
        The parser must implement AbstractPlanParser interface and be able to:
        - Parse raw EXPLAIN ANALYZE output
        - Generate AbstractPlanOperator tree structure
        
        Returns:
            AbstractPlanParser instance
        """
        pass
    
    @abstractmethod
    def get_statistics_converter(self) -> 'StatisticsConverter':
        """
        Returns a statistics converter for this DBMS.
        
        The converter transforms DBMS-specific statistics format
        to StandardizedStatistics format.
        
        Returns:
            StatisticsConverter instance
        """
        pass
    
    @abstractmethod
    def get_connection_factory(self) -> Type['DatabaseConnection']:
        """
        Returns a database connection class for this DBMS.
        
        Returns:
            DatabaseConnection subclass (not instance)
        """
        pass
    
    @abstractmethod
    def get_operator_normalizer(self) -> 'OperatorNormalizer':
        """
        Returns an operator normalizer for this DBMS.
        
        The normalizer converts DBMS-specific operator names
        to logical operator types (e.g., "Parallel Seq Scan" -> "TableScan" + parallel=True)
        
        Returns:
            OperatorNormalizer instance
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns plugin metadata.
        
        Returns:
            Dictionary with plugin information
        """
        return {
            'name': self.name,
            'display_name': self.display_name,
            'version': getattr(self, 'version', '1.0.0'),
            'description': getattr(self, 'description', ''),
        }
    
    def validate(self) -> bool:
        """
        Validates that the plugin is properly configured.
        
        Returns:
            True if plugin is valid, False otherwise
        """
        try:
            # Check that all required components can be instantiated
            self.get_parser()
            self.get_statistics_converter()
            self.get_connection_factory()
            self.get_operator_normalizer()
            return True
        except NotImplementedError:
            return False
        except Exception as e:
            print(f"Plugin validation failed for {self.name}: {e}")
            return False


# Forward declarations for type hints
# These will be properly imported from their respective modules
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

