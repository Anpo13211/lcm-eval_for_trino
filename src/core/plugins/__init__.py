"""
Plugin system for DBMS extensibility
"""

from .dbms_plugin import DBMSPlugin
from .registry import DBMSRegistry

__all__ = ['DBMSPlugin', 'DBMSRegistry']

