"""可扩展特征提取框架。"""

from analysis.features.base import FeatureContext, FeatureExtractor, FeatureSet
from analysis.features.registry import FeatureRegistry, default_registry
from analysis.features.skeleton import SkeletonFeatureExtractor
from analysis.features.spatial import BoxSpatialFeatureExtractor
from analysis.features.temporal import TemporalFeatureExtractor

__all__ = [
    "FeatureContext",
    "FeatureExtractor",
    "FeatureSet",
    "FeatureRegistry",
    "default_registry",
    "SkeletonFeatureExtractor",
    "BoxSpatialFeatureExtractor",
    "TemporalFeatureExtractor",
]
