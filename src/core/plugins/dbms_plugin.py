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
        
        The parser should inherit from:
            core.parsers.base.AbstractPlanParser (preferred)
        or:
            cross_db_benchmark.benchmark_tools.abstract.plan_parser.AbstractPlanParser (legacy)
        
        Returns:
            AbstractPlanParser instance
        
        Example:
            def get_parser(self):
                from core.parsers.adapter import wrap_legacy_parser
                from my_dbms.parser import MyDBMSParser
                return wrap_legacy_parser(MyDBMSParser(), self.name)
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
    
    def get_plan_collator(self) -> Optional[Any]:
        """
        Returns a plan collator function for training (optional).
        
        Plugins can override this to provide DBMS-specific collation logic.
        If not provided, training code should fall back to unified_plan_collator.
        
        Returns:
            Collator function or None
        """
        return None
    
    def get_feature_aliases(self) -> Optional[Dict[str, str]]:
        """
        Returns feature name aliases for this DBMS (optional).
        
        Maps DBMS-specific feature names to standardized names.
        
        Returns:
            Dictionary mapping DBMS features to standard features, or None
        """
        return None
    
    def get_plan_adapter(self) -> Optional[Any]:
        """
        Returns a plan adapter function for legacy models (optional).
        
        Some models (e.g., QPPNet, QueryFormer) use model-specific plan formats.
        Plugins can provide adapters to convert DBMS-specific plans to these formats.
        
        Returns:
            Adapter function (plan → dict) or None
            
        Example:
            def adapt_to_qppnet(plan):
                # Convert Trino plan to QPPNet format
                return {'node_type': ..., 'children': ...}
        """
        return None
    
    def run_workload(self, workload_path, db_name, database_conn_args, database_kwarg_dict, target_path, run_kwargs,
                     repetitions_per_query, timeout_sec, mode, hints=None, with_indexes=False, cap_workload=None, 
                     explain_only: bool = False, min_runtime=100):
        """
        Run a workload against the database.
        
        Plugins should implement this method to support benchmark execution.
        
        Args:
            workload_path: Path to workload queries
            db_name: Database name
            database_conn_args: Connection arguments
            database_kwarg_dict: Additional connection kwargs
            target_path: Path to save results
            run_kwargs: Run configuration
            repetitions_per_query: Number of repetitions
            timeout_sec: Timeout in seconds
            mode: Execution mode
            hints: Query hints (optional)
            with_indexes: Whether indexes are used
            cap_workload: Max number of queries
            explain_only: If True, only run EXPLAIN
            min_runtime: Minimum runtime to capture
        """
        raise NotImplementedError(f"run_workload not implemented for {self.name}")

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
    # Try new unified parser first
    from core.parsers.base import AbstractPlanParser
except ImportError:
    try:
        # Fall back to legacy parser
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

