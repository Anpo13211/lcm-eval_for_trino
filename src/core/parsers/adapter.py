"""
Parser Adapter - Bridge between old and new parser interfaces

This module provides adapters to use existing parsers with the new
unified AbstractPlanParser interface.

This allows gradual migration without breaking existing code.
"""

from typing import Any, List, Tuple
from .base import AbstractPlanParser


class LegacyParserAdapter(AbstractPlanParser):
    """
    Adapter to wrap legacy parsers in the new AbstractPlanParser interface.
    
    This allows existing parsers (from cross_db_benchmark/benchmark_tools)
    to work with the new unified interface.
    
    Usage:
        # Wrap existing parser
        legacy_parser = existing_parser_instance
        unified_parser = LegacyParserAdapter(legacy_parser, "postgres")
        
        # Now use unified interface
        plans, runtimes = unified_parser.parse_explain_analyze_file("plans.txt")
    """
    
    def __init__(self, legacy_parser: Any, database_type: str):
        """
        Initialize adapter.
        
        Args:
            legacy_parser: Existing parser instance
            database_type: DBMS type
        """
        super().__init__(database_type)
        self.legacy_parser = legacy_parser
    
    def parse_explain_analyze_file(
        self,
        file_path: str,
        min_runtime: float = 0,
        max_runtime: float = float('inf'),
        **kwargs
    ) -> Tuple[List[Any], List[float]]:
        """
        Parse EXPLAIN ANALYZE file using legacy parser.
        
        This adapts the legacy parse_plans() or similar method to the
        new interface.
        """
        # Try to use parse_plans method if available
        if hasattr(self.legacy_parser, 'parse_plans'):
            # Legacy method signature
            result = self.legacy_parser.parse_plans(
                file_path,
                min_runtime=min_runtime,
                max_runtime=max_runtime,
                **kwargs
            )
            
            # Extract plans and runtimes from legacy result format
            if isinstance(result, dict):
                parsed_plans = result.get('parsed_plans', [])
                runtimes = result.get('runtimes', [])
            elif isinstance(result, tuple):
                parsed_plans, runtimes = result[0], result[1]
            else:
                parsed_plans, runtimes = result, []
            
            return parsed_plans, runtimes
        
        # Fallback: try to read file and parse each plan
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Try parse_raw_plan
            if hasattr(self.legacy_parser, 'parse_raw_plan'):
                root, exec_time, plan_time = self.legacy_parser.parse_raw_plan(content)
                return [root], [exec_time]
        except Exception as e:
            print(f"Warning: Legacy parser failed: {e}")
            return [], []
    
    def parse_raw_plan(
        self,
        plan_text: str,
        analyze: bool = True,
        **kwargs
    ) -> Tuple[Any, float, float]:
        """
        Parse raw plan using legacy parser.
        """
        if hasattr(self.legacy_parser, 'parse_raw_plan'):
            return self.legacy_parser.parse_raw_plan(
                plan_text,
                analyze=analyze,
                **kwargs
            )
        
        # Fallback
        if hasattr(self.legacy_parser, 'parse_single_plan'):
            result = self.legacy_parser.parse_single_plan(plan_text, **kwargs)
            return result, 0.0, 0.0
        
        raise NotImplementedError(
            f"Legacy parser {type(self.legacy_parser)} does not implement "
            "parse_raw_plan or parse_single_plan"
        )


def wrap_legacy_parser(parser: Any, database_type: str) -> AbstractPlanParser:
    """
    Convenience function to wrap legacy parser.
    
    Args:
        parser: Legacy parser instance
        database_type: DBMS type
    
    Returns:
        AbstractPlanParser compatible wrapper
    
    Example:
        >>> from cross_db_benchmark.benchmark_tools.postgres import parse_plan
        >>> legacy = parse_plan.PostgresPlanParser()
        >>> unified = wrap_legacy_parser(legacy, "postgres")
        >>> plans, runtimes = unified.parse_explain_analyze_file("plans.txt")
    """
    return LegacyParserAdapter(parser, database_type)

