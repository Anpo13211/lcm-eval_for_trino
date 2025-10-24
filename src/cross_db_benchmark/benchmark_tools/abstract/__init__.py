"""
Abstract base classes for database query plan parsing

This module provides abstract base classes that all database-specific
implementations must inherit from to ensure consistency across different
database systems (PostgreSQL, Trino, MySQL, etc.).

Key classes:
    - AbstractPlanOperator: Base class for query plan operators
    - AbstractPlanParser: Base class for plan parsers
"""

from .plan_operator import AbstractPlanOperator
from .plan_parser import AbstractPlanParser

__all__ = [
    'AbstractPlanOperator',
    'AbstractPlanParser',
]


