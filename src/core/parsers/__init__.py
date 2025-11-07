"""
Core Parser Module

Unified parser interface for all DBMS.
This module provides the abstract base class that all DBMS parsers must implement.

Usage:
    # For new implementations, inherit from AbstractPlanParser:
    from core.parsers import AbstractPlanParser
    
    class MyDBMSParser(AbstractPlanParser):
        def __init__(self):
            super().__init__("mydbms")
        
        def parse_explain_analyze_file(self, file_path, ...):
            # Implementation
            pass
    
    # For legacy parsers, use the adapter:
    from core.parsers.adapter import wrap_legacy_parser
    unified_parser = wrap_legacy_parser(legacy_parser, "postgres")
"""

from .base import AbstractPlanParser, PlanParseResult, get_parser_for_dbms
from .adapter import LegacyParserAdapter, wrap_legacy_parser

__all__ = [
    'AbstractPlanParser',
    'PlanParseResult',
    'LegacyParserAdapter',
    'wrap_legacy_parser',
    'get_parser_for_dbms',
]

