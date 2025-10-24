"""
Abstract base class for query plan operators

All database-specific plan operator implementations must inherit from
AbstractPlanOperator to ensure a unified interface across different
database systems.

This module defines the contract that all implementations must follow,
using PostgreSQL-compatible feature names as the standard.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class AbstractPlanOperator(ABC, dict):
    """
    Abstract base class for query plan operators
    
    All DBMS-specific implementations must inherit from this class and
    provide plan information using PostgreSQL-compatible feature names.
    
    This class inherits from both ABC (for abstract methods) and dict
    (for compatibility with existing serialization and usage patterns).
    
    ===== Required Features (PostgreSQL-based) =====
    The following features must be provided by all DBMS implementations:
    
    - op_name: str           Operator name (e.g., "Seq Scan", "Hash Join")
    - est_card: float        Estimated cardinality (number of rows)
    - act_card: float        Actual cardinality (number of rows)
    - est_width: float       Estimated row width (bytes)
    - workers_planned: int   Number of planned parallel workers
    - act_children_card: float  Product of actual cardinalities of children
    - est_children_card: float  Product of estimated cardinalities of children
    
    ===== Optional Features =====
    Provide these features only if available in the DBMS:
    
    - est_cost: Optional[float]         Estimated cost (PostgreSQL format)
    - est_startup_cost: Optional[float] Estimated startup cost
    - act_time: Optional[float]         Actual execution time (ms)
    - table: Optional[str]              Table name
    - columns: Optional[List[str]]      List of columns
    - output_columns: Optional[List[Dict]]  Output column information
    - filter_columns: Optional[Any]     Filter conditions
    
    ===== Access Methods =====
    Features can be accessed in two ways for backward compatibility:
    
    1. Via properties (recommended for new code):
       plan.op_name
       plan.est_card
    
    2. Via plan_parameters dict (for existing code):
       plan.plan_parameters['op_name']
       plan.plan_parameters['est_card']
    
    Both methods access the same underlying data.
    """
    
    def __init__(self, plain_content=None, children=None, plan_parameters=None, plan_runtime=0):
        """
        Initialize the base plan operator
        
        Args:
            plain_content: Raw plan text lines (for parsing)
            children: List of child operators
            plan_parameters: Dictionary of plan parameters (optional)
            plan_runtime: Execution time in milliseconds
        
        Note:
            Subclasses MUST call super().__init__()
        """
        super().__init__()
        # Make dict attributes accessible as object attributes
        self.__dict__ = self
        
        # === Core data structures ===
        self.plain_content = plain_content if plain_content is not None else []
        self.children = list(children) if children is not None else []
        self.plan_parameters = plan_parameters if plan_parameters is not None else {}
        self.plan_runtime = plan_runtime
        
        # === Metadata ===
        self.database_type = ""        # "postgres", "trino", etc.
        self.database_id = ""          # Database instance ID
        self.sql: Optional[str] = None # SQL query string
        
        # Initialize default values in plan_parameters if not present
        if 'op_name' not in self.plan_parameters:
            self.plan_parameters['op_name'] = 'Unknown'
        if 'est_card' not in self.plan_parameters:
            self.plan_parameters['est_card'] = 0.0
        if 'act_card' not in self.plan_parameters:
            self.plan_parameters['act_card'] = 0.0
        if 'est_width' not in self.plan_parameters:
            self.plan_parameters['est_width'] = 0.0
        if 'workers_planned' not in self.plan_parameters:
            self.plan_parameters['workers_planned'] = 0
        if 'act_children_card' not in self.plan_parameters:
            self.plan_parameters['act_children_card'] = 1.0
        if 'est_children_card' not in self.plan_parameters:
            self.plan_parameters['est_children_card'] = 1.0
    
    # ==========================================
    # Required feature properties
    # ==========================================
    
    @property
    def op_name(self) -> str:
        """Operator name"""
        return self.plan_parameters.get('op_name', 'Unknown')
    
    @op_name.setter
    def op_name(self, value: str):
        self.plan_parameters['op_name'] = value
    
    @property
    def est_card(self) -> float:
        """Estimated cardinality (number of rows)"""
        return self.plan_parameters.get('est_card', 0.0)
    
    @est_card.setter
    def est_card(self, value: float):
        self.plan_parameters['est_card'] = float(value)
    
    @property
    def act_card(self) -> float:
        """Actual cardinality (number of rows)"""
        return self.plan_parameters.get('act_card', 0.0)
    
    @act_card.setter
    def act_card(self, value: float):
        self.plan_parameters['act_card'] = float(value)
    
    @property
    def est_width(self) -> float:
        """Estimated row width (bytes)"""
        return self.plan_parameters.get('est_width', 0.0)
    
    @est_width.setter
    def est_width(self, value: float):
        self.plan_parameters['est_width'] = float(value)
    
    @property
    def workers_planned(self) -> int:
        """Number of planned parallel workers"""
        return self.plan_parameters.get('workers_planned', 0)
    
    @workers_planned.setter
    def workers_planned(self, value: int):
        self.plan_parameters['workers_planned'] = int(value)
    
    @property
    def act_children_card(self) -> float:
        """Product of actual cardinalities of children nodes"""
        return self.plan_parameters.get('act_children_card', 1.0)
    
    @act_children_card.setter
    def act_children_card(self, value: float):
        self.plan_parameters['act_children_card'] = float(value)
    
    @property
    def est_children_card(self) -> float:
        """Product of estimated cardinalities of children nodes"""
        return self.plan_parameters.get('est_children_card', 1.0)
    
    @est_children_card.setter
    def est_children_card(self, value: float):
        self.plan_parameters['est_children_card'] = float(value)
    
    # ==========================================
    # Optional feature properties
    # ==========================================
    
    @property
    def est_cost(self) -> Optional[float]:
        """Estimated cost (PostgreSQL format, None if unavailable)"""
        return self.plan_parameters.get('est_cost')
    
    @est_cost.setter
    def est_cost(self, value: Optional[float]):
        if value is not None:
            self.plan_parameters['est_cost'] = float(value)
        else:
            self.plan_parameters['est_cost'] = None
    
    @property
    def est_startup_cost(self) -> Optional[float]:
        """Estimated startup cost (None if unavailable)"""
        return self.plan_parameters.get('est_startup_cost')
    
    @est_startup_cost.setter
    def est_startup_cost(self, value: Optional[float]):
        if value is not None:
            self.plan_parameters['est_startup_cost'] = float(value)
        else:
            self.plan_parameters['est_startup_cost'] = None
    
    @property
    def act_time(self) -> Optional[float]:
        """Actual execution time in milliseconds (None if unavailable)"""
        return self.plan_parameters.get('act_time')
    
    @act_time.setter
    def act_time(self, value: Optional[float]):
        if value is not None:
            self.plan_parameters['act_time'] = float(value)
        else:
            self.plan_parameters['act_time'] = None
    
    @property
    def table(self) -> Optional[str]:
        """Table name"""
        return self.plan_parameters.get('table')
    
    @table.setter
    def table(self, value: Optional[str]):
        self.plan_parameters['table'] = value
    
    @property
    def columns(self) -> Optional[List[str]]:
        """List of columns"""
        return self.plan_parameters.get('columns')
    
    @columns.setter
    def columns(self, value: Optional[List[str]]):
        self.plan_parameters['columns'] = value
    
    @property
    def output_columns(self) -> Optional[List[Dict]]:
        """Output column information"""
        return self.plan_parameters.get('output_columns')
    
    @output_columns.setter
    def output_columns(self, value: Optional[List[Dict]]):
        self.plan_parameters['output_columns'] = value
    
    @property
    def filter_columns(self) -> Optional[Any]:
        """Filter conditions"""
        return self.plan_parameters.get('filter_columns')
    
    @filter_columns.setter
    def filter_columns(self, value: Optional[Any]):
        self.plan_parameters['filter_columns'] = value
    
    # ==========================================
    # Abstract methods (must be implemented)
    # ==========================================
    
    @abstractmethod
    def parse_lines(self, alias_dict: Optional[Dict] = None, **kwargs) -> None:
        """
        Extract features from DBMS-specific raw plan text
        
        【Required Implementation】
        1. Extract operator name and set to self.plan_parameters['op_name']
        2. Extract cardinality and set to self.plan_parameters['est_card'], etc.
        3. Extract other features and set to corresponding keys
        4. Map DBMS-specific names to unified names
           Example: Trino's est_rows → 'est_card'
        
        Args:
            alias_dict: Dictionary of table aliases (optional)
            **kwargs: DBMS-specific additional parameters
                     Common kwargs:
                     - parse_baseline: bool
                     - parse_join_conds: bool
        
        Raises:
            NotImplementedError: If not implemented
        
        Example:
            def parse_lines(self, alias_dict=None, **kwargs):
                # Parse Trino format
                params = self._parse_trino_text()
                
                # Map to unified names (store in plan_parameters)
                self.plan_parameters.update({
                    'op_name': params['op_name'],
                    'est_card': params['est_rows'],  # Trino → unified
                    'act_card': params['act_output_rows'],  # Trino → unified
                })
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.parse_lines() must be implemented"
        )
    
    @abstractmethod
    def parse_columns_bottom_up(
        self, 
        column_id_mapping: Dict,
        partial_column_name_mapping: Dict,
        table_id_mapping: Dict,
        **kwargs
    ) -> set:
        """
        Match column information with statistics in bottom-up manner
        
        【Required Implementation】
        1. Process children nodes recursively
        2. Calculate and set children cardinalities
           - self.plan_parameters['act_children_card']
           - self.plan_parameters['est_children_card']
        3. Resolve output_columns
        4. Resolve filter_columns
        5. Return set of tables processed
        
        Args:
            column_id_mapping: Dictionary mapping (table, column) -> column_id
            partial_column_name_mapping: Dictionary mapping column -> {table}
            table_id_mapping: Dictionary mapping table -> table_id
            **kwargs: DBMS-specific additional parameters
                     Common kwargs:
                     - alias_dict: Dict[str, str]
        
        Returns:
            Set of table names used in this node
        
        Raises:
            NotImplementedError: If not implemented
        
        Example:
            def parse_columns_bottom_up(self, column_id_mapping, ...):
                # Process children first
                node_tables = set()
                for child in self.children:
                    node_tables.update(child.parse_columns_bottom_up(...))
                
                # Calculate children cardinalities
                if self.children:
                    act_cards = [c.act_card for c in self.children]
                    self.plan_parameters['act_children_card'] = prod(act_cards)
                
                # Resolve columns
                # ...
                
                return node_tables
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.parse_columns_bottom_up() must be implemented"
        )
    
    # ==========================================
    # Common methods (can be overridden)
    # ==========================================
    
    def parse_lines_recursively(self, **kwargs) -> None:
        """
        Parse lines for this node and all children recursively
        
        Args:
            **kwargs: Arguments to pass to parse_lines()
        """
        self.parse_lines(**kwargs)
        for child in self.children:
            if hasattr(child, 'parse_lines_recursively'):
                child.parse_lines_recursively(**kwargs)
    
    def merge_recursively(self, other: 'AbstractPlanOperator') -> None:
        """
        Merge information from another plan operator tree
        
        This is useful when combining EXPLAIN and EXPLAIN ANALYZE results.
        
        Args:
            other: Another plan operator tree to merge from
        """
        # Merge plan_parameters (keep existing values, add new ones)
        for key, value in other.plan_parameters.items():
            if key not in self.plan_parameters or self.plan_parameters[key] is None:
                self.plan_parameters[key] = value
        
        # Merge children recursively
        if len(self.children) == len(other.children):
            for i, child in enumerate(self.children):
                if hasattr(child, 'merge_recursively'):
                    child.merge_recursively(other.children[i])
    
    def min_card(self) -> float:
        """
        Get minimum cardinality in this subtree
        
        Returns:
            Minimum act_card value (excluding zero/negative)
        """
        min_val = self.act_card if self.act_card > 0 else float('inf')
        
        for child in self.children:
            if hasattr(child, 'min_card'):
                child_min = child.min_card()
                if child_min < min_val:
                    min_val = child_min
        
        return min_val if min_val != float('inf') else 0.0
    
    def validate(self) -> List[str]:
        """
        Validate that required features are properly set
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check required features
        if not self.op_name or self.op_name == "Unknown":
            errors.append("op_name is not set")
        
        if self.est_card < 0:
            errors.append(f"est_card is negative: {self.est_card}")
        
        if self.act_card < 0:
            errors.append(f"act_card is negative: {self.act_card}")
        
        if self.est_width < 0:
            errors.append(f"est_width is negative: {self.est_width}")
        
        if self.workers_planned < 0:
            errors.append(f"workers_planned is negative: {self.workers_planned}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get plan information as dictionary (for debugging/serialization)
        
        Returns:
            Dictionary containing all features
        """
        return {
            'database_type': self.database_type,
            'database_id': self.database_id,
            'op_name': self.op_name,
            'est_card': self.est_card,
            'act_card': self.act_card,
            'est_width': self.est_width,
            'workers_planned': self.workers_planned,
            'act_children_card': self.act_children_card,
            'est_children_card': self.est_children_card,
            'est_cost': self.est_cost,
            'est_startup_cost': self.est_startup_cost,
            'act_time': self.act_time,
            'table': self.table,
            'columns': self.columns,
            'num_children': len(self.children),
            'plan_runtime': self.plan_runtime,
        }
    
    def __repr__(self):
        """String representation of the operator"""
        return (
            f"{self.__class__.__name__}("
            f"db={self.database_type}, "
            f"op={self.op_name}, "
            f"est_card={self.est_card:.0f}, "
            f"act_card={self.act_card:.0f}, "
            f"children={len(self.children)})"
        )
