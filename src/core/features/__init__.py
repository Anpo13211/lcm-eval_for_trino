"""
Feature mapping system for cross-database support
"""

from .mapping import FeatureMapping, FeatureType
from .mapper import FeatureMapper
from .registry import FEATURE_REGISTRY

__all__ = [
    'FeatureMapping',
    'FeatureType',
    'FeatureMapper',
    'FEATURE_REGISTRY',
]

