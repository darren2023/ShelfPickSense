"""货框几何布局特征（货架侧 / 推导层列 / 货架层列统计）。"""

from __future__ import annotations

from analysis.box_layout import BoxNumericCode, compute_shelf_layout_stats
from analysis.features.base import FeatureContext, FeatureExtractor


class BoxLayoutFeatureExtractor(FeatureExtractor):
    """
    基于 box_layout 推导的 per-box 布局特征。

    与 annotation 中的 layer/column 不同，此处为几何计算得到的 layout_layer/layout_column。
    """

    name = "layout"

    def extract_frame(self, ctx: FeatureContext) -> dict[str, float]:
        return {}

    def per_box_feature_names(self) -> list[str]:
        return [
            "shelf_side",
            "layout_layer",
            "layout_column",
            "shelf_layer_count",
            "shelf_column_count_mean",
        ]

    def extract_per_box(self, ctx: FeatureContext) -> dict[str, dict[str, float]]:
        layout: dict[str, BoxNumericCode] = ctx.record.box_layout
        if not layout:
            return {}

        shelf_stats = ctx.record.shelf_layout_stats or compute_shelf_layout_stats(layout)
        result: dict[str, dict[str, float]] = {}

        for token in ctx.box_tokens:
            entry = layout.get(token)
            if entry is None:
                continue
            stats = shelf_stats.get(entry.shelf_code or "_default")
            result[token] = {
                "shelf_side": float(entry.shelf_side),
                "layout_layer": float(entry.layer),
                "layout_column": float(entry.column),
                "shelf_layer_count": float(stats.layer_count if stats else 0),
                "shelf_column_count_mean": float(stats.column_count_mean if stats else 0.0),
            }
        return result
