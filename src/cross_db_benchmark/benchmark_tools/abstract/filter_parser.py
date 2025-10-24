"""
Abstract filter parser for unified filter/predicate parsing across database systems.

This module provides abstract base classes for filter parsing that enable
O(M+N) implementation cost by unifying common parsing logic while allowing
database-specific operator handling.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any
from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator


class AbstractPredicateNode(ABC):
    """
    Abstract base class for predicate nodes across database systems.
    
    Provides unified interface for parsing database-specific filter conditions
    while maintaining database-specific operator handling.
    """
    
    def __init__(self, text: str, children: List['AbstractPredicateNode']):
        self.text = text
        self.children = children
        self.column: Optional[Union[str, tuple]] = None
        self.operator: Optional[Union[Operator, LogicalOperator]] = None
        self.literal: Optional[Any] = None
        self.filter_feature: Optional[int] = None
    
    def __str__(self):
        return self.to_tree_rep(depth=0)
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return dict(
            column=self.column,
            operator=str(self.operator),
            literal=self.literal,
            literal_feature=self.filter_feature,
            children=[c.to_dict() for c in self.children]
        )
    
    def lookup_columns(self, plan, **kwargs):
        """Lookup column IDs in the plan"""
        if self.column is not None:
            self.column = plan.lookup_column_id(self.column, **kwargs)
        for c in self.children:
            c.lookup_columns(plan, **kwargs)
    
    def parse_lines_recursively(self, parse_baseline: bool = False):
        """
        Recursively parse all nodes in the tree.
        
        Common logic for baseline filtering is implemented here,
        while database-specific parsing is delegated to parse_lines.
        """
        self.parse_lines(parse_baseline=parse_baseline)
        for c in self.children:
            c.parse_lines_recursively(parse_baseline=parse_baseline)
        
        # Common baseline filtering logic
        if parse_baseline:
            self.children = [c for c in self.children if
                           c.operator in {LogicalOperator.AND, LogicalOperator.OR,
                                        Operator.IS_NOT_NULL, Operator.IS_NULL}
                           or c.literal is not None]
    
    @abstractmethod
    def parse_lines(self, parse_baseline: bool = False):
        """
        Parse the current node's text into operator, column, and literal.
        
        This method must be implemented by each database system to handle
        database-specific operators and syntax.
        
        Args:
            parse_baseline: Whether to parse for baseline statistics
        """
        pass
    
    def to_tree_rep(self, depth: int = 0) -> str:
        """Generate tree representation"""
        rep_text = '\n' + ''.join(['\t'] * depth)
        rep_text += self.text
        
        for c in self.children:
            rep_text += c.to_tree_rep(depth=depth + 1)
        
        return rep_text


class AbstractFilterParser(ABC):
    """
    Abstract base class for filter parsing across database systems.
    
    Provides unified interface for parsing filter conditions while allowing
    database-specific predicate node implementations.
    """
    
    def __init__(self, database_type: str):
        self.database_type = database_type
    
    @abstractmethod
    def create_predicate_node(self, text: str, children: List[AbstractPredicateNode]) -> AbstractPredicateNode:
        """
        Create a database-specific predicate node.
        
        Args:
            text: The text content of the node
            children: Child nodes
            
        Returns:
            Database-specific predicate node instance
        """
        pass
    
    def parse_recursively(self, filter_cond: str, offset: int, _class: type = None) -> tuple[AbstractPredicateNode, int]:
        """
        Recursively parse filter conditions with unified bracket handling.
        
        This method provides common logic for handling nested parentheses
        and quotes, delegating node creation to the database-specific implementation.
        
        Args:
            filter_cond: The filter condition string
            offset: Current parsing position
            _class: Predicate node class (uses create_predicate_node if None)
            
        Returns:
            Tuple of (parsed_node, next_offset)
        """
        if _class is None:
            _class = self.create_predicate_node
        
        escaped = False
        node_text = ''
        children = []
        
        while True:
            if offset >= len(filter_cond):
                return _class(node_text, children), offset
            
            if filter_cond[offset] == '(' and not escaped:
                child_node, offset = self.parse_recursively(filter_cond, offset + 1, _class=_class)
                children.append(child_node)
            elif filter_cond[offset] == ')' and not escaped:
                return _class(node_text, children), offset
            elif filter_cond[offset] == "'":
                escaped = not escaped
                node_text += "'"
            else:
                node_text += filter_cond[offset]
            offset += 1
    
    def parse_filter(self, filter_cond: str, parse_baseline: bool = False) -> Optional[AbstractPredicateNode]:
        """
        Parse a filter condition into a predicate tree.
        
        Common logic for filter parsing is implemented here,
        while database-specific node creation is delegated to create_predicate_node.
        
        Args:
            filter_cond: The filter condition string
            parse_baseline: Whether to parse for baseline statistics
            
        Returns:
            Parsed predicate tree or None if parsing fails
        """
        if not filter_cond:
            return None
        
        parse_tree, _ = self.parse_recursively(filter_cond, offset=0)
        
        # Handle the case where parse_recursively returns a root node with children
        if len(parse_tree.children) == 1:
            parse_tree = parse_tree.children[0]
        elif len(parse_tree.children) == 0:
            # Single condition without parentheses
            pass
        else:
            # Multiple conditions at root level - wrap in AND
            return None  # This case needs special handling
        
        parse_tree.parse_lines_recursively(parse_baseline=parse_baseline)
        
        # Common validation logic
        if parse_tree.operator not in {LogicalOperator.AND, LogicalOperator.OR, 
                                     Operator.IS_NOT_NULL, Operator.IS_NULL} \
                and parse_tree.literal is None:
            return None
        if parse_tree.operator in {LogicalOperator.AND, LogicalOperator.OR} and len(parse_tree.children) == 0:
            return None
        
        return parse_tree
