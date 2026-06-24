"""可扩展特征提取框架。"""

from analysis.features.base import FeatureContext, FeatureExtractor, FeatureSet
from analysis.features.registry import FeatureRegistry, default_registry
from analysis.features.selection import FeatureSelection, load_feature_selection
from analysis.features.skeleton import SkeletonFeatureExtractor
from analysis.features.spatial import BoxSpatialFeatureExtractor
from analysis.features.temporal import TemporalFeatureExtractor

__all__ = [
    "FeatureContext",
    "FeatureExtractor",
    "FeatureSet",
    "FeatureRegistry",
    "FeatureSelection",
    "default_registry",
    "load_feature_selection",
    "SkeletonFeatureExtractor",
    "BoxSpatialFeatureExtractor",
    "TemporalFeatureExtractor",
]
