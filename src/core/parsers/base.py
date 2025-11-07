"""
Abstract Plan Parser - Unified Parser Interface

This is the canonical abstract base class for all DBMS plan parsers.
All database-specific parsers must inherit from this class.

This replaces the scattered parser definitions with a single unified interface
managed within src/core/.

Design principles:
1. Database-agnostic interface
2. Type-safe return values
3. Minimal required methods
4. Extensible for DBMS-specific features
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class PlanParseResult:
    """
    Standardized result from parsing a plan file.
    
    This provides a clean, type-safe interface for parse results.
    """
    parsed_plans: List[Any]  # List of AbstractPlanOperator instances
    runtimes: List[float]    # Runtime for each plan (milliseconds)
    planning_times: Optional[List[float]] = None  # Planning time (milliseconds)
    metadata: Optional[Dict[str, Any]] = None      # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AbstractPlanParser(ABC):
    """
    Abstract base class for database query plan parsers.
    
    All database-specific parsers (PostgreSQL, Trino, MySQL, etc.) must inherit
    from this class and implement the required abstract methods.
    
    This class defines the minimal interface needed for:
    1. Parsing EXPLAIN ANALYZE output files
    2. Parsing individual plan texts
    3. Extracting execution statistics
    
    Implementation cost per DBMS: ~200-300 lines
    
    Attributes:
        database_type (str): The type of database (e.g., 'postgres', 'trino')
    
    Example:
        class PostgreSQLParser(AbstractPlanParser):
            def __init__(self):
                super().__init__("postgres")
            
            def parse_explain_analyze_file(self, file_path, ...):
                # Implementation
                pass
            
            def parse_raw_plan(self, plan_text, ...):
                # Implementation
                pass
    """
    
    def __init__(self, database_type: str):
        """
        Initialize the plan parser.
        
        Args:
            database_type: The type of database system (e.g., 'postgres', 'trino')
        """
        self.database_type = database_type
    
    @abstractmethod
    def parse_explain_analyze_file(
        self,
        file_path: str,
        min_runtime: float = 0,
        max_runtime: float = float('inf'),
        **kwargs
    ) -> Tuple[List[Any], List[float]]:
        """
        Parse EXPLAIN ANALYZE output from a file.
        
        This is the primary interface used by training scripts to load
        query plans from disk.
        
        Args:
            file_path: Path to file containing EXPLAIN ANALYZE output
            min_runtime: Minimum runtime threshold in milliseconds (default: 0)
            max_runtime: Maximum runtime threshold in milliseconds (default: inf)
            **kwargs: Additional database-specific parameters
        
        Returns:
            tuple: (parsed_plans, runtimes) where:
                - parsed_plans: List of AbstractPlanOperator instances
                - runtimes: List of execution times in milliseconds
        
        Example:
            >>> parser = TrinoPlanParser()
            >>> plans, runtimes = parser.parse_explain_analyze_file("plans.txt")
            >>> print(f"Loaded {len(plans)} plans")
            >>> print(f"Average runtime: {np.mean(runtimes):.2f}ms")
        
        Notes:
            - Plans with runtime outside [min_runtime, max_runtime] are filtered
            - Each parsed plan should have plan_runtime attribute set
            - Return empty lists if file cannot be parsed
        """
        pass
    
    @abstractmethod
    def parse_raw_plan(
        self,
        plan_text: str,
        analyze: bool = True,
        **kwargs
    ) -> Tuple[Any, float, float]:
        """
        Parse raw EXPLAIN ANALYZE text output from the database.
        
        This method parses a single plan from raw text (as opposed to a file
        containing multiple plans).
        
        Args:
            plan_text: Raw plan text from EXPLAIN ANALYZE
            analyze: Whether to extract execution statistics (default: True)
            **kwargs: Additional database-specific parameters
        
        Returns:
            tuple: (root_operator, execution_time, planning_time) where:
                - root_operator: AbstractPlanOperator tree root
                - execution_time: Execution time in milliseconds
                - planning_time: Planning time in milliseconds
        
        Example:
            >>> parser = PostgreSQLParser()
            >>> plan_text = "Seq Scan on users  (cost=0.00..1.10 rows=10 ...)"
            >>> root, exec_time, plan_time = parser.parse_raw_plan(plan_text)
            >>> print(f"Root operator: {root.op_name}")
            >>> print(f"Execution: {exec_time}ms, Planning: {plan_time}ms")
        
        Notes:
            - Should handle both EXPLAIN and EXPLAIN ANALYZE output
            - If analyze=False, execution_time may be 0
            - Should raise ValueError for invalid plan text
        """
        pass
    
    def parse_multiple_plans(
        self,
        plan_texts: List[str],
        analyze: bool = True,
        **kwargs
    ) -> PlanParseResult:
        """
        Parse multiple plans at once (convenience method).
        
        This is a convenience wrapper around parse_raw_plan() for batch processing.
        
        Args:
            plan_texts: List of raw plan texts
            analyze: Whether to extract execution statistics
            **kwargs: Additional parameters
        
        Returns:
            PlanParseResult with all parsed plans
        
        Example:
            >>> parser = TrinoPlanParser()
            >>> result = parser.parse_multiple_plans(plan_texts)
            >>> print(f"Parsed {len(result.parsed_plans)} plans")
        """
        parsed_plans = []
        runtimes = []
        planning_times = []
        
        for plan_text in plan_texts:
            try:
                root, exec_time, plan_time = self.parse_raw_plan(
                    plan_text, analyze=analyze, **kwargs
                )
                if root is not None:
                    parsed_plans.append(root)
                    runtimes.append(exec_time)
                    planning_times.append(plan_time)
            except Exception as e:
                # Log error but continue with other plans
                print(f"Warning: Failed to parse plan: {e}")
                continue
        
        return PlanParseResult(
            parsed_plans=parsed_plans,
            runtimes=runtimes,
            planning_times=planning_times,
            metadata={'num_failed': len(plan_texts) - len(parsed_plans)}
        )
    
    def get_statistics(
        self,
        parsed_plans: List[Any]
    ) -> Dict[str, Any]:
        """
        Extract statistics from parsed plans.
        
        This is an optional method that can be overridden to provide
        additional statistics about parsed plans.
        
        Args:
            parsed_plans: List of parsed AbstractPlanOperator instances
        
        Returns:
            Dictionary with statistics (implementation-specific)
        
        Example:
            >>> parser = PostgreSQLParser()
            >>> stats = parser.get_statistics(plans)
            >>> print(stats)
            {'num_plans': 100, 'avg_depth': 5.2, 'num_operators': 523}
        """
        return {
            'num_plans': len(parsed_plans),
            'database_type': self.database_type,
        }
    
    def validate_plan(self, plan: Any) -> bool:
        """
        Validate that a parsed plan is well-formed.
        
        This is an optional method for plan validation.
        
        Args:
            plan: Parsed plan operator
        
        Returns:
            True if valid, False otherwise
        
        Example:
            >>> parser = TrinoPlanParser()
            >>> is_valid = parser.validate_plan(root_operator)
            >>> if not is_valid:
            >>>     print("Warning: Invalid plan structure")
        """
        if plan is None:
            return False
        
        # Basic validation: check for required attributes
        required_attrs = ['op_name', 'children']
        return all(hasattr(plan, attr) for attr in required_attrs)
    
    def __repr__(self) -> str:
        """String representation of parser."""
        return f"{self.__class__.__name__}(database_type='{self.database_type}')"


# Compatibility layer: make this available as if imported from abstract module
# This allows gradual migration from old imports
try:
    from cross_db_benchmark.benchmark_tools.abstract import plan_parser as _old_parser
    
    # If old module exists, we're in compatibility mode
    # New code should use this module, but old imports still work
    _COMPATIBILITY_MODE = True
except ImportError:
    _COMPATIBILITY_MODE = False


def get_parser_for_dbms(dbms_name: str) -> AbstractPlanParser:
    """
    Factory function to get parser for a specific DBMS.
    
    This is a convenience function that uses the plugin registry.
    
    Args:
        dbms_name: DBMS name (e.g., "postgres", "trino")
    
    Returns:
        AbstractPlanParser instance
    
    Example:
        >>> parser = get_parser_for_dbms("trino")
        >>> plans, runtimes = parser.parse_explain_analyze_file("plans.txt")
    """
    from core.plugins.registry import DBMSRegistry
    return DBMSRegistry.get_parser(dbms_name)

