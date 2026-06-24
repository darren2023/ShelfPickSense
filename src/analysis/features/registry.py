"""特征提取器注册表。"""

from __future__ import annotations

from analysis.features.base import FeatureContext, FeatureExtractor, FeatureSet, PerBoxFeatureSet
from analysis.features.skeleton import SkeletonFeatureExtractor
from analysis.features.spatial import BoxSpatialFeatureExtractor
from analysis.features.temporal import TemporalFeatureExtractor
from analysis.records import FramePersons, RecordData


class FeatureRegistry:
    """管理并组合多个 FeatureExtractor。"""

    def __init__(self, extractors: list[FeatureExtractor] | None = None) -> None:
        self._extractors: list[FeatureExtractor] = list(extractors or [])

    def register(self, extractor: FeatureExtractor) -> None:
        self._extractors.append(extractor)

    @property
    def extractors(self) -> list[FeatureExtractor]:
        return list(self._extractors)

    def extract_frame_features_from_context(self, ctx: FeatureContext) -> FeatureSet:
        merged: dict[str, float] = {}
        for ext in self._extractors:
            for k, v in ext.extract_frame(ctx).items():
                merged[f"{ext.name}.{k}"] = float(v)
        return FeatureSet(record_id=ctx.record.record_id, frame_idx=ctx.frame.frame_idx, features=merged)

    def extract_frame_features(self, record: RecordData, frame: FramePersons) -> FeatureSet:
        ctx = FeatureContext.from_record(record, frame)
        return self.extract_frame_features_from_context(ctx)

    def extract_per_box_features_from_context(self, ctx: FeatureContext) -> list[PerBoxFeatureSet]:
        per_box: dict[str, dict[str, float]] = {}
        for ext in self._extractors:
            box_feats = ext.extract_per_box(ctx)
            for token, feats in box_feats.items():
                bucket = per_box.setdefault(token, {})
                for k, v in feats.items():
                    bucket[f"{ext.name}.{k}"] = float(v)

        return [
            PerBoxFeatureSet(
                record_id=ctx.record.record_id,
                frame_idx=ctx.frame.frame_idx,
                box_token=token,
                features=feats,
            )
            for token, feats in sorted(per_box.items())
        ]

    def extract_per_box_features(
        self, record: RecordData, frame: FramePersons
    ) -> list[PerBoxFeatureSet]:
        ctx = FeatureContext.from_record(record, frame)
        return self.extract_per_box_features_from_context(ctx)

    def frame_feature_names(self, record: RecordData) -> list[str]:
        frames = record.frames()
        if not frames:
            return []
        sample = self.extract_frame_features(record, frames[0])
        return sample.names()

    def per_box_feature_names(self, record: RecordData) -> list[str]:
        frames = record.frames()
        if not frames:
            return []
        samples = self.extract_per_box_features(record, frames[0])
        if not samples:
            return []
        return sorted(samples[0].features.keys())


def default_registry() -> FeatureRegistry:
    reg = FeatureRegistry()
    reg.register(SkeletonFeatureExtractor())
    reg.register(BoxSpatialFeatureExtractor())
    reg.register(TemporalFeatureExtractor())
    return reg
