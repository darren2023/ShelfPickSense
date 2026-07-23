"""特征提取器注册表。"""

from __future__ import annotations

from analysis.features.base import FeatureContext, FeatureExtractor, FeatureSet, PerBoxFeatureSet
from analysis.features.layout import BoxLayoutFeatureExtractor
from analysis.features.rule_engine import RuleEngineFeatureExtractor
from analysis.features.speed import SpeedFeatureExtractor
from analysis.features.stance import StanceFeatureExtractor
from analysis.features.skeleton import SkeletonFeatureExtractor
from analysis.features.spatial import BoxSpatialFeatureExtractor
from analysis.features.temporal import TemporalFeatureExtractor
from analysis.features.tracking import person_track_id, sorted_persons
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

    def frame_feature_names(self) -> list[str]:
        names: list[str] = []
        for ext in self._extractors:
            names.extend(f"{ext.name}.{name}" for name in ext.frame_feature_names())
        return list(dict.fromkeys(names))

    def per_box_schema_feature_names(self) -> list[str]:
        names: list[str] = []
        for ext in self._extractors:
            names.extend(f"{ext.name}.{name}" for name in ext.per_box_feature_names())
        return list(dict.fromkeys(names))

    def select_extractors_for_features(
        self,
        *,
        frame_feature_names: list[str] | None = None,
        box_feature_names: list[str] | None = None,
    ) -> "FeatureRegistry":
        if frame_feature_names is None and box_feature_names is None:
            return self
        prefixes: set[str] = set()
        for names in (frame_feature_names, box_feature_names):
            for name in names or []:
                prefix, _, _ = name.partition(".")
                if prefix:
                    prefixes.add(prefix)
        return FeatureRegistry([ext for ext in self._extractors if ext.name in prefixes])

    def extract_frame_feature_groups_from_context(self, ctx: FeatureContext) -> list[FeatureSet]:
        groups: list[FeatureSet] = []
        for person in sorted_persons(ctx.frame):
            track_id = person_track_id(person)
            person_ctx = ctx.for_person(person, track_id)
            features: dict[str, float] = {}
            for ext in self._extractors:
                for k, v in ext.extract_frame(person_ctx).items():
                    features[f"{ext.name}.{k}"] = float(v)
            groups.append(
                FeatureSet(
                    record_id=ctx.record.record_id,
                    frame_idx=ctx.frame.frame_idx,
                    person_track_id=track_id,
                    features=features,
                )
            )
        return groups

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

    def per_box_feature_names(self, record: RecordData) -> list[str]:
        return self.per_box_schema_feature_names()


def default_registry() -> FeatureRegistry:
    reg = FeatureRegistry()
    reg.register(SkeletonFeatureExtractor())
    reg.register(SpeedFeatureExtractor())
    reg.register(StanceFeatureExtractor())
    reg.register(BoxSpatialFeatureExtractor())
    reg.register(BoxLayoutFeatureExtractor())
    reg.register(TemporalFeatureExtractor())
    reg.register(RuleEngineFeatureExtractor())
    return reg
