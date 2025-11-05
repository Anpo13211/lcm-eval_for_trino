"""
Feature mapper - converts logical feature names to DBMS-specific values

This is the runtime component that uses FeatureMapping definitions
to extract and transform feature values from plan nodes.

Usage:
    mapper = FeatureMapper("postgres")
    
    # Extract single feature
    est_card = mapper.get_feature(
        logical_name="estimated_cardinality",
        plan_params=node.plan_parameters
    )
    
    # Extract multiple features
    features = mapper.extract_features(
        logical_names=["estimated_cardinality", "operator_type"],
        plan_params=node.plan_parameters
    )
"""

from typing import Any, Dict, List, Optional, Union
import logging

from .registry import FEATURE_REGISTRY, get_feature_mapping
from .mapping import FeatureMapping


logger = logging.getLogger(__name__)


class FeatureMapper:
    """
    Runtime feature extraction and transformation.
    
    This class provides the actual mapping logic that:
    1. Looks up DBMS-specific attribute names from logical names
    2. Extracts values from plan_parameters (dict or SimpleNamespace)
    3. Applies transformations
    4. Returns default values for missing attributes
    
    Implementation:
        mapper = FeatureMapper("postgres")
        value = mapper.get_feature("estimated_cardinality", plan_params)
    """
    
    def __init__(self, dbms_name: str, strict: bool = False):
        """
        Initialize mapper for a specific DBMS.
        
        Args:
            dbms_name: Name of DBMS (e.g., "postgres", "trino")
            strict: If True, raise exceptions for missing features.
                   If False, return default values (recommended for production)
        """
        self.dbms_name = dbms_name
        self.strict = strict
        self._cache: Dict[str, Optional[str]] = {}
    
    def get_feature(
        self,
        logical_name: str,
        plan_params: Any,
        default: Any = None
    ) -> Any:
        """
        Extract single feature value from plan parameters.
        
        Args:
            logical_name: Logical feature name (e.g., "estimated_cardinality")
            plan_params: Plan parameters (dict or SimpleNamespace)
            default: Override default value (optional)
        
        Returns:
            Extracted and transformed feature value
        
        Raises:
            KeyError: If feature not found and strict=True
        """
        # Look up feature mapping
        try:
            mapping = get_feature_mapping(logical_name)
        except KeyError:
            if self.strict:
                raise
            logger.warning(f"Feature '{logical_name}' not in registry")
            return default
        
        # Get DBMS-specific attribute name (with caching)
        cache_key = logical_name
        if cache_key not in self._cache:
            self._cache[cache_key] = mapping.get_alias(self.dbms_name)
        
        attr_name = self._cache[cache_key]
        
        if attr_name is None:
            # This DBMS doesn't provide this feature
            result = default if default is not None else mapping.default_value
            if self.strict:
                raise KeyError(
                    f"Feature '{logical_name}' not available for DBMS '{self.dbms_name}'"
                )
            return result
        
        # Extract value from plan_params
        raw_value = self._get_value_from_params(plan_params, attr_name)
        
        if raw_value is None:
            return default if default is not None else mapping.default_value
        
        # Apply transformation
        transformed_value = mapping.transform(raw_value)
        
        # Validate if needed
        if not mapping.validate(transformed_value):
            logger.warning(
                f"Feature '{logical_name}' value {transformed_value} "
                f"out of range [{mapping.min_value}, {mapping.max_value}]"
            )
            return default if default is not None else mapping.default_value
        
        return transformed_value
    
    def extract_features(
        self,
        logical_names: List[str],
        plan_params: Any,
        as_dict: bool = False
    ) -> Union[List[Any], Dict[str, Any]]:
        """
        Extract multiple features at once.
        
        Args:
            logical_names: List of logical feature names
            plan_params: Plan parameters (dict or SimpleNamespace)
            as_dict: If True, return dict; if False, return list
        
        Returns:
            List of values (order matches logical_names) or dict
        """
        if as_dict:
            return {
                name: self.get_feature(name, plan_params)
                for name in logical_names
            }
        else:
            return [
                self.get_feature(name, plan_params)
                for name in logical_names
            ]
    
    def extract_feature_group(
        self,
        group_name: str,
        plan_params: Any,
        as_dict: bool = False
    ) -> Union[List[Any], Dict[str, Any]]:
        """
        Extract all features from a predefined group.
        
        Args:
            group_name: Group name ('plan', 'filter', 'column', 'table', 'output_column')
            plan_params: Plan parameters
            as_dict: If True, return dict; if False, return list
        
        Returns:
            List of values or dict
        """
        from .registry import FEATURE_GROUPS
        
        if group_name not in FEATURE_GROUPS:
            raise ValueError(
                f"Unknown feature group '{group_name}'. "
                f"Available: {list(FEATURE_GROUPS.keys())}"
            )
        
        feature_mappings = FEATURE_GROUPS[group_name]
        logical_names = [fm.logical_name for fm in feature_mappings]
        
        return self.extract_features(logical_names, plan_params, as_dict=as_dict)
    
    def _get_value_from_params(self, params: Any, attr_name: str) -> Any:
        """
        Extract value from plan_parameters (dict or SimpleNamespace).
        
        Args:
            params: Plan parameters (various formats)
            attr_name: Attribute name to extract
        
        Returns:
            Extracted value or None
        """
        if params is None:
            return None
        
        # Handle dict
        if isinstance(params, dict):
            return params.get(attr_name)
        
        # Handle SimpleNamespace or object with attributes
        if hasattr(params, '__dict__'):
            return getattr(params, attr_name, None)
        
        # Handle vars() dict
        try:
            params_dict = vars(params)
            return params_dict.get(attr_name)
        except TypeError:
            pass
        
        return None
    
    def get_available_features(self) -> List[str]:
        """
        Get list of features available for this DBMS.
        
        Returns:
            List of logical feature names
        """
        available = []
        for logical_name, mapping in FEATURE_REGISTRY.items():
            if mapping.get_alias(self.dbms_name) is not None:
                available.append(logical_name)
        return available
    
    def get_missing_features(self, required_features: List[str]) -> List[str]:
        """
        Get list of required features not available for this DBMS.
        
        Args:
            required_features: List of required feature names
        
        Returns:
            List of missing feature names
        """
        available = set(self.get_available_features())
        return [f for f in required_features if f not in available]
    
    def create_feature_vector(
        self,
        logical_names: List[str],
        plan_params: Any,
        flatten: bool = True
    ) -> List[float]:
        """
        Create a numeric feature vector for ML models.
        
        This is a convenience method that:
        1. Extracts features
        2. Encodes categorical features as one-hot or indices
        3. Returns flat numeric vector
        
        Args:
            logical_names: List of feature names
            plan_params: Plan parameters
            flatten: If True, flatten nested structures
        
        Returns:
            Flat list of numeric values
        """
        values = self.extract_features(logical_names, plan_params)
        
        # Convert to numeric (this is a simplified version)
        numeric_values = []
        for value in values:
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
            elif isinstance(value, bool):
                numeric_values.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # For strings, would need feature_statistics for encoding
                # For now, just use hash
                numeric_values.append(float(hash(value) % 10000))
            else:
                numeric_values.append(0.0)
        
        return numeric_values
    
    def validate_plan_params(self, plan_params: Any) -> bool:
        """
        Validate that plan_params has expected structure.
        
        Args:
            plan_params: Plan parameters to validate
        
        Returns:
            True if valid, False otherwise
        """
        if plan_params is None:
            return False
        
        # Check that it's dict-like or has attributes
        if not isinstance(plan_params, dict) and not hasattr(plan_params, '__dict__'):
            return False
        
        return True


class BatchFeatureMapper:
    """
    Batch feature extraction for multiple plan nodes.
    
    This is more efficient than creating individual FeatureMapper
    for each node when processing large plan trees.
    """
    
    def __init__(self, dbms_name: str, strict: bool = False):
        """
        Initialize batch mapper.
        
        Args:
            dbms_name: DBMS name
            strict: Strict mode flag
        """
        self.mapper = FeatureMapper(dbms_name, strict=strict)
    
    def extract_batch(
        self,
        logical_names: List[str],
        plan_params_list: List[Any]
    ) -> List[List[Any]]:
        """
        Extract features from multiple plan nodes.
        
        Args:
            logical_names: List of feature names
            plan_params_list: List of plan_parameters objects
        
        Returns:
            List of feature vectors (one per plan node)
        """
        return [
            self.mapper.extract_features(logical_names, params)
            for params in plan_params_list
        ]
    
    def extract_tree(
        self,
        logical_names: List[str],
        root_node: Any
    ) -> List[List[Any]]:
        """
        Extract features from entire plan tree.
        
        Args:
            logical_names: List of feature names
            root_node: Root of plan tree (with .children)
        
        Returns:
            List of feature vectors for all nodes in tree
        """
        result = []
        
        def traverse(node):
            if hasattr(node, 'plan_parameters'):
                features = self.mapper.extract_features(
                    logical_names,
                    node.plan_parameters
                )
                result.append(features)
            
            if hasattr(node, 'children'):
                for child in node.children:
                    traverse(child)
        
        traverse(root_node)
        return result

