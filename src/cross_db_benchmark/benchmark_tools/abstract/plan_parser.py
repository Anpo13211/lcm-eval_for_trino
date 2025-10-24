"""
Abstract base class for database query plan parsers
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class AbstractPlanParser(ABC):
    """
    Abstract base class for query plan parsers.
    
    All database-specific parsers (PostgreSQL, Trino, MySQL, etc.) must inherit
    from this class and implement the required abstract methods.
    
    Attributes:
        database_type (str): The type of database (e.g., 'postgres', 'trino', 'mysql')
    """
    
    def __init__(self, database_type: str):
        """
        Initialize the plan parser.
        
        Args:
            database_type: The type of database system (e.g., 'postgres', 'trino')
        """
        self.database_type = database_type
    
    @abstractmethod
    def parse_plans(
        self,
        run_stats: Any,
        min_runtime: float = 100,
        max_runtime: float = 30000,
        parse_baseline: bool = False,
        cap_queries: Optional[int] = None,
        parse_join_conds: bool = False,
        include_zero_card: bool = False,
        explain_only: bool = False,
        **kwargs
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Parse multiple query plans from run statistics.
        
        This method should:
        1. Extract plans from run_stats
        2. Filter based on runtime constraints
        3. Parse each plan into AbstractPlanOperator instances
        4. Collect statistics
        5. Return results in standardized format
        
        Args:
            run_stats: Database run statistics object containing query plans
            min_runtime: Minimum runtime threshold in milliseconds (default: 100)
            max_runtime: Maximum runtime threshold in milliseconds (default: 30000)
            parse_baseline: Whether to parse baseline information (default: False)
            cap_queries: Maximum number of queries to parse (default: None)
            parse_join_conds: Whether to parse join conditions (default: False)
            include_zero_card: Whether to include zero cardinality plans (default: False)
            explain_only: Whether to only parse EXPLAIN plans without execution (default: False)
            **kwargs: Additional database-specific parameters
        
        Returns:
            tuple: (parsed_runs, stats) where:
                - parsed_runs (dict): Contains:
                    * 'parsed_plans': List[AbstractPlanOperator] - Parsed plan operators
                    * 'database_stats': Database statistics object
                    * 'run_kwargs': Runtime arguments
                - stats (dict): Contains:
                    * 'runtimes': Runtime statistics
                    * 'no_tables': Number of tables per query
                    * 'no_filters': Number of filters per query
        """
        pass
    
    @abstractmethod
    def parse_single_plan(
        self,
        plan_text: str,
        analyze: bool = True,
        parse: bool = True,
        **kwargs
    ) -> Optional[Any]:
        """
        Parse a single query plan from text.
        
        This method should:
        1. Parse the plan text structure
        2. Extract timing information (if analyze=True)
        3. Create AbstractPlanOperator tree
        4. Return the root operator
        
        Args:
            plan_text: Raw plan text from the database
            analyze: Whether to extract execution statistics (default: True)
            parse: Whether to parse the plan structure (default: True)
            **kwargs: Additional database-specific parameters
        
        Returns:
            Optional[AbstractPlanOperator]: Root operator of the parsed plan,
                                           or None if parsing fails
        
        Example:
            >>> parser = TrinoPlanParser()
            >>> root = parser.parse_single_plan(plan_text)
            >>> if root:
            ...     print(f"Root operator: {root.op_name}")
        """
        pass
    
    def get_statistics(self, parsed_plans: List[Any]) -> Dict[str, Any]:
        """
        Calculate statistics from parsed plans.
        
        Args:
            parsed_plans: List of parsed AbstractPlanOperator instances
        
        Returns:
            dict: Statistics including:
                - total_plans: Number of plans
                - avg_runtime: Average runtime in milliseconds
                - min_runtime: Minimum runtime in milliseconds
                - max_runtime: Maximum runtime in milliseconds
                - median_runtime: Median runtime in milliseconds
        """
        if not parsed_plans:
            return {
                'total_plans': 0,
                'avg_runtime': 0.0,
                'min_runtime': 0.0,
                'max_runtime': 0.0,
                'median_runtime': 0.0
            }
        
        runtimes = []
        for plan in parsed_plans:
            if hasattr(plan, 'plan_runtime'):
                runtimes.append(plan.plan_runtime)
        
        if not runtimes:
            return {
                'total_plans': len(parsed_plans),
                'avg_runtime': 0.0,
                'min_runtime': 0.0,
                'max_runtime': 0.0,
                'median_runtime': 0.0
            }
        
        return {
            'total_plans': len(parsed_plans),
            'avg_runtime': float(np.mean(runtimes)),
            'min_runtime': float(np.min(runtimes)),
            'max_runtime': float(np.max(runtimes)),
            'median_runtime': float(np.median(runtimes))
        }
    
    def validate_parsed_plans(self, parsed_plans: List[Any]) -> List[str]:
        """
        Validate that parsed plans conform to requirements.
        
        Args:
            parsed_plans: List of parsed AbstractPlanOperator instances
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        if not isinstance(parsed_plans, list):
            errors.append(f"parsed_plans must be a list, got {type(parsed_plans)}")
            return errors
        
        for i, plan in enumerate(parsed_plans):
            # Check if plan has required attributes
            if not hasattr(plan, 'op_name'):
                errors.append(f"Plan {i}: missing 'op_name' attribute")
            
            if not hasattr(plan, 'est_card'):
                errors.append(f"Plan {i}: missing 'est_card' attribute")
            
            if not hasattr(plan, 'act_card'):
                errors.append(f"Plan {i}: missing 'act_card' attribute")
            
            if not hasattr(plan, 'database_type'):
                errors.append(f"Plan {i}: missing 'database_type' attribute")
            elif plan.database_type != self.database_type:
                errors.append(
                    f"Plan {i}: database_type mismatch - expected '{self.database_type}', "
                    f"got '{plan.database_type}'"
                )
            
            # Validate numeric values
            if hasattr(plan, 'est_card') and plan.est_card is not None:
                if not isinstance(plan.est_card, (int, float)) or plan.est_card < 0:
                    errors.append(f"Plan {i}: est_card must be non-negative number")
            
            if hasattr(plan, 'act_card') and plan.act_card is not None:
                if not isinstance(plan.act_card, (int, float)) or plan.act_card < 0:
                    errors.append(f"Plan {i}: act_card must be non-negative number")
        
        return errors
    
    def filter_by_runtime(
        self,
        plans: List[Any],
        runtimes: List[float],
        min_runtime: float = 100,
        max_runtime: float = 30000
    ) -> tuple[List[Any], List[float]]:
        """
        Filter plans based on runtime thresholds.
        
        Args:
            plans: List of plan operators
            runtimes: List of runtimes corresponding to each plan
            min_runtime: Minimum runtime threshold in milliseconds
            max_runtime: Maximum runtime threshold in milliseconds
        
        Returns:
            tuple: (filtered_plans, filtered_runtimes)
        """
        filtered_plans = []
        filtered_runtimes = []
        
        for plan, runtime in zip(plans, runtimes):
            if min_runtime <= runtime <= max_runtime:
                filtered_plans.append(plan)
                filtered_runtimes.append(runtime)
        
        return filtered_plans, filtered_runtimes
    
    def collect_operator_statistics(
        self,
        parsed_plans: List[Any]
    ) -> Dict[str, int]:
        """
        Collect statistics about operator usage across plans.
        
        Args:
            parsed_plans: List of parsed AbstractPlanOperator instances
        
        Returns:
            dict: Mapping of operator names to their occurrence count
        """
        operator_counts = {}
        
        def count_operators(node):
            """Recursively count operators in the plan tree"""
            if hasattr(node, 'op_name') and node.op_name:
                op_name = node.op_name
                operator_counts[op_name] = operator_counts.get(op_name, 0) + 1
            
            if hasattr(node, 'children'):
                for child in node.children:
                    count_operators(child)
        
        for plan in parsed_plans:
            count_operators(plan)
        
        return operator_counts
    
    def print_summary(
        self,
        parsed_runs: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> None:
        """
        Print a summary of parsed plans and statistics.
        
        Args:
            parsed_runs: Parsed runs dictionary from parse_plans()
            stats: Statistics dictionary from parse_plans()
        """
        parsed_plans = parsed_runs.get('parsed_plans', [])
        
        print(f"\n{'='*60}")
        print(f"Plan Parser Summary ({self.database_type.upper()})")
        print(f"{'='*60}")
        print(f"Total plans parsed: {len(parsed_plans)}")
        
        if parsed_plans:
            plan_stats = self.get_statistics(parsed_plans)
            print(f"\nRuntime Statistics:")
            print(f"  Average: {plan_stats['avg_runtime']:.2f} ms")
            print(f"  Median:  {plan_stats['median_runtime']:.2f} ms")
            print(f"  Min:     {plan_stats['min_runtime']:.2f} ms")
            print(f"  Max:     {plan_stats['max_runtime']:.2f} ms")
            
            operator_stats = self.collect_operator_statistics(parsed_plans)
            if operator_stats:
                print(f"\nTop Operators:")
                sorted_ops = sorted(
                    operator_stats.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                for op_name, count in sorted_ops:
                    percentage = (count / len(parsed_plans)) * 100
                    print(f"  {op_name:30s}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"{'='*60}\n")
    
    def __repr__(self) -> str:
        """String representation of the parser"""
        return f"{self.__class__.__name__}(database_type='{self.database_type}')"

