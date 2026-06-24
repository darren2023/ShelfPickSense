"""特征相关性分析。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analysis.dataset import Dataset, load_dataset


@dataclass
class CorrelationAnalysisResult:
    input_source: str
    output_dir: str
    method: str
    threshold: float
    frame_count: int
    box_sample_count: int
    outputs: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "input_source": self.input_source,
            "output_dir": self.output_dir,
            "method": self.method,
            "threshold": self.threshold,
            "frame_count": self.frame_count,
            "box_sample_count": self.box_sample_count,
            "outputs": self.outputs,
        }


def _frame_dataframe(dataset: Dataset) -> pd.DataFrame:
    rows = []
    for sample in dataset.frame_samples:
        row = {
            "record_id": sample.record_id,
            "frame_idx": sample.frame_idx,
            "is_picking": int(sample.is_picking),
        }
        row.update(dict(zip(dataset.frame_feature_names, sample.x.tolist(), strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _box_dataframe(dataset: Dataset) -> pd.DataFrame:
    rows = []
    for sample in dataset.box_samples:
        row = {
            "record_id": sample.record_id,
            "frame_idx": sample.frame_idx,
            "box_token": sample.box_token,
            "is_target": int(sample.is_target),
        }
        row.update(dict(zip(dataset.box_feature_names, sample.x.tolist(), strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_matrix(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if df.empty or not feature_names:
        return pd.DataFrame()
    return df[feature_names].apply(pd.to_numeric, errors="coerce")


def _target_correlations(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    target_col: str,
    method: str,
) -> pd.DataFrame:
    if df.empty or not feature_names or target_col not in df:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation", "non_null_count"])

    target = pd.to_numeric(df[target_col], errors="coerce")
    rows = []
    for feature in feature_names:
        values = pd.to_numeric(df[feature], errors="coerce")
        pair = pd.concat([values, target], axis=1).dropna()
        if len(pair) < 2 or pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
            corr = 0.0
        else:
            corr_value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
            corr = 0.0 if pd.isna(corr_value) else float(corr_value)
        rows.append(
            {
                "feature": feature,
                "correlation": corr,
                "abs_correlation": abs(corr),
                "non_null_count": int(len(pair)),
            }
        )
    return pd.DataFrame(rows).sort_values(["abs_correlation", "feature"], ascending=[False, True])


def _high_correlation_pairs(
    corr: pd.DataFrame,
    *,
    threshold: float,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            value = float(value)
            abs_value = abs(value)
            if abs_value < threshold:
                continue
            rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "correlation": value,
                    "abs_correlation": abs_value,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "abs_correlation"])
    return result.sort_values(["abs_correlation", "feature_a", "feature_b"], ascending=[False, True, True]).head(top_n)


def _format_float(value: object, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def _short_text(value: object, limit: int = 42) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _rel_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _corr_color(value: float) -> str:
    value = max(-1.0, min(1.0, value))
    if value >= 0:
        base = 255 - int(130 * value)
        return f"rgb(255,{base},{base})"
    base = 255 - int(130 * abs(value))
    return f"rgb({base},{base},255)"


def _write_target_bar_svg(path: Path, data: pd.DataFrame, *, title: str, top_n: int = 15) -> None:
    rows = data.head(top_n).copy()
    width = 1000
    row_h = 30
    top = 58
    center_x = 550
    bar_scale = 360
    height = top + max(len(rows), 1) * row_h + 42
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="24" y="32" class="title">{escape(title)}</text>',
        f'<line x1="{center_x}" y1="{top - 22}" x2="{center_x}" y2="{height - 25}" class="axis"/>',
        f'<text x="{center_x - bar_scale}" y="{top - 28}" text-anchor="middle">-1</text>',
        f'<text x="{center_x}" y="{top - 28}" text-anchor="middle">0</text>',
        f'<text x="{center_x + bar_scale}" y="{top - 28}" text-anchor="middle">1</text>',
    ]
    if rows.empty:
        parts.append(f'<text x="24" y="{top + 22}">无可用数据</text>')
    for idx, row in enumerate(rows.itertuples(index=False)):
        y = top + idx * row_h
        corr = float(getattr(row, "correlation", 0.0) or 0.0)
        bar_w = abs(corr) * bar_scale
        x = center_x if corr >= 0 else center_x - bar_w
        color = "#d95f5f" if corr >= 0 else "#5f7fd9"
        parts.extend(
            [
                f'<text x="24" y="{y + 18}">{escape(_short_text(getattr(row, "feature", ""), 44))}</text>',
                f'<rect x="{x:.2f}" y="{y + 5}" width="{bar_w:.2f}" height="18" fill="{color}" opacity="0.85"/>',
                f'<text x="{center_x + bar_scale + 18}" y="{y + 18}">{_format_float(corr)}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_pairs_bar_svg(path: Path, data: pd.DataFrame, *, title: str, top_n: int = 15) -> None:
    rows = data.head(top_n).copy()
    width = 1100
    row_h = 30
    top = 58
    label_w = 520
    bar_scale = 420
    height = top + max(len(rows), 1) * row_h + 42
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#555;stroke-width:1}</style>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="24" y="32" class="title">{escape(title)}</text>',
        f'<line x1="{label_w}" y1="{top - 22}" x2="{label_w + bar_scale}" y2="{top - 22}" class="axis"/>',
        f'<text x="{label_w}" y="{top - 28}" text-anchor="middle">0</text>',
        f'<text x="{label_w + bar_scale}" y="{top - 28}" text-anchor="middle">1</text>',
    ]
    if rows.empty:
        parts.append(f'<text x="24" y="{top + 22}">无超过阈值的高相关特征对</text>')
    for idx, row in enumerate(rows.itertuples(index=False)):
        y = top + idx * row_h
        value = float(getattr(row, "abs_correlation", 0.0) or 0.0)
        label = f"{getattr(row, 'feature_a', '')} ↔ {getattr(row, 'feature_b', '')}"
        parts.extend(
            [
                f'<text x="24" y="{y + 18}">{escape(_short_text(label, 70))}</text>',
                f'<rect x="{label_w}" y="{y + 5}" width="{value * bar_scale:.2f}" height="18" fill="#7a68a6" opacity="0.85"/>',
                f'<text x="{label_w + bar_scale + 18}" y="{y + 18}">{_format_float(getattr(row, "correlation", 0.0))}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _select_heatmap_features(corr: pd.DataFrame, target_corr: pd.DataFrame, top_n: int = 15) -> list[str]:
    if corr.empty:
        return []
    selected = [f for f in target_corr.head(top_n)["feature"].tolist() if f in corr.columns]
    if selected:
        return selected[:top_n]
    return list(corr.columns[:top_n])


def _write_heatmap_svg(path: Path, corr: pd.DataFrame, target_corr: pd.DataFrame, *, title: str, top_n: int = 15) -> None:
    features = _select_heatmap_features(corr, target_corr, top_n=top_n)
    cell = 28
    left = 210
    top = 200
    width = left + max(len(features), 1) * cell + 60
    height = top + max(len(features), 1) * cell + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;font-size:12px}.title{font-size:20px;font-weight:700}.cell{stroke:#fff;stroke-width:1}</style>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="24" y="32" class="title">{escape(title)}</text>',
        '<text x="24" y="56">颜色：红色为正相关，蓝色为负相关，颜色越深绝对相关越强。</text>',
    ]
    if not features:
        parts.append(f'<text x="24" y="{top + 22}">无可用数据</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return
    for idx, feature in enumerate(features):
        x = left + idx * cell + 14
        parts.append(
            f'<text x="{x}" y="{top - 10}" text-anchor="end" transform="rotate(-45 {x} {top - 10})">{escape(_short_text(feature, 28))}</text>'
        )
        y = top + idx * cell + 18
        parts.append(f'<text x="{left - 8}" y="{y}" text-anchor="end">{escape(_short_text(feature, 28))}</text>')
    for row_idx, row_feature in enumerate(features):
        for col_idx, col_feature in enumerate(features):
            value = corr.loc[row_feature, col_feature]
            value = 0.0 if pd.isna(value) else float(value)
            x = left + col_idx * cell
            y = top + row_idx * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" class="cell" fill="{_corr_color(value)}">'
                f"<title>{escape(row_feature)} / {escape(col_feature)}: {_format_float(value)}</title></rect>"
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _prepare_pca_matrix(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    matrix = _feature_matrix(df, feature_names)
    if matrix.empty:
        return pd.DataFrame(), []
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    usable_features = [col for col in matrix.columns if matrix[col].nunique(dropna=False) > 1]
    if not usable_features:
        return pd.DataFrame(), []
    return matrix[usable_features], usable_features


def _compute_pca_analysis(
    *,
    prefix: str,
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    output_dir: Path,
    outputs: dict[str, str],
    max_components: int = 5,
) -> dict[str, pd.DataFrame]:
    matrix, usable_features = _prepare_pca_matrix(df, feature_names)
    empty_variance = pd.DataFrame(columns=["component", "explained_variance_ratio", "cumulative_variance_ratio"])
    empty_loadings = pd.DataFrame(columns=["feature", "PC1", "PC2"])
    empty_projection = pd.DataFrame(columns=["record_id", "frame_idx", target_col, "PC1", "PC2"])
    if matrix.empty or len(matrix) < 2:
        variance_df = empty_variance
        loadings_df = empty_loadings
        projection_df = empty_projection
    else:
        component_count = min(max_components, len(usable_features), len(matrix))
        scaled = StandardScaler().fit_transform(matrix)
        pca = PCA(n_components=component_count)
        projection = pca.fit_transform(scaled)
        components = [f"PC{i + 1}" for i in range(component_count)]
        variance_df = pd.DataFrame(
            {
                "component": components,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
            }
        )
        loadings_df = pd.DataFrame(pca.components_.T, columns=components)
        loadings_df.insert(0, "feature", usable_features)
        projection_df = pd.DataFrame(projection[:, : min(2, component_count)], columns=components[: min(2, component_count)])
        if "PC2" not in projection_df.columns:
            projection_df["PC2"] = 0.0
        meta_cols = [col for col in ("record_id", "frame_idx", "box_token", target_col) if col in df.columns]
        projection_df = pd.concat([df[meta_cols].reset_index(drop=True), projection_df[["PC1", "PC2"]]], axis=1)

    variance_path = output_dir / f"{prefix}_pca_explained_variance.csv"
    loadings_path = output_dir / f"{prefix}_pca_loadings.csv"
    projection_path = output_dir / f"{prefix}_pca_projection.csv"
    variance_df.to_csv(variance_path, index=False, encoding="utf-8-sig")
    loadings_df.to_csv(loadings_path, index=False, encoding="utf-8-sig")
    projection_df.to_csv(projection_path, index=False, encoding="utf-8-sig")
    outputs[f"{prefix}_pca_explained_variance"] = str(variance_path)
    outputs[f"{prefix}_pca_loadings"] = str(loadings_path)
    outputs[f"{prefix}_pca_projection"] = str(projection_path)
    return {
        "variance": variance_df,
        "loadings": loadings_df,
        "projection": projection_df,
    }


def _write_pca_variance_svg(path: Path, variance: pd.DataFrame, *, title: str) -> None:
    width = 760
    height = 360
    left = 70
    top = 58
    plot_w = 620
    plot_h = 230
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="24" y="32" class="title">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
    ]
    if variance.empty:
        parts.append(f'<text x="24" y="{top + 40}">无可用 PCA 数据</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return
    rows = variance.head(10).copy()
    bar_gap = 12
    bar_w = max(24, (plot_w - bar_gap * (len(rows) + 1)) / max(len(rows), 1))
    points = []
    for idx, row in enumerate(rows.itertuples(index=False)):
        ratio = float(getattr(row, "explained_variance_ratio", 0.0) or 0.0)
        cumulative = float(getattr(row, "cumulative_variance_ratio", 0.0) or 0.0)
        x = left + bar_gap + idx * (bar_w + bar_gap)
        bar_h = ratio * plot_h
        y = top + plot_h - bar_h
        cx = x + bar_w / 2
        cy = top + plot_h - cumulative * plot_h
        points.append((cx, cy))
        parts.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#5f9ed1" opacity="0.85"/>',
                f'<text x="{cx:.2f}" y="{top + plot_h + 22}" text-anchor="middle">{escape(str(getattr(row, "component", "")))}</text>',
                f'<text x="{cx:.2f}" y="{max(y - 6, top + 12):.2f}" text-anchor="middle">{_format_float(ratio, 2)}</text>',
            ]
        )
    if len(points) > 1:
        path_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(f'<polyline points="{path_points}" fill="none" stroke="#d95f5f" stroke-width="2"/>')
    for x, y in points:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#d95f5f"/>')
    parts.append(f'<text x="{left}" y="{height - 18}">蓝柱：单个主成分解释方差；红线：累计解释方差。</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_pca_scatter_svg(path: Path, projection: pd.DataFrame, *, title: str, target_col: str) -> None:
    width = 760
    height = 520
    left = 80
    top = 58
    plot_w = 620
    plot_h = 380
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#555;stroke-width:1}.grid{stroke:#e5e5e5;stroke-width:1}</style>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="24" y="32" class="title">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#ddd"/>',
    ]
    if projection.empty or "PC1" not in projection or "PC2" not in projection:
        parts.append(f'<text x="24" y="{top + 40}">无可用 PCA 投影数据</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return
    x_values = pd.to_numeric(projection["PC1"], errors="coerce")
    y_values = pd.to_numeric(projection["PC2"], errors="coerce")
    valid = projection[x_values.notna() & y_values.notna()].copy()
    if valid.empty:
        parts.append(f'<text x="24" y="{top + 40}">无可用 PCA 投影数据</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return
    x_values = pd.to_numeric(valid["PC1"], errors="coerce")
    y_values = pd.to_numeric(valid["PC2"], errors="coerce")
    xmin, xmax = float(x_values.min()), float(x_values.max())
    ymin, ymax = float(y_values.min()), float(y_values.max())
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    sample = valid if len(valid) <= 1500 else valid.sample(1500, random_state=42)
    for row in sample.itertuples(index=False):
        x = float(getattr(row, "PC1"))
        y = float(getattr(row, "PC2"))
        sx = left + (x - xmin) / (xmax - xmin) * plot_w
        sy = top + plot_h - (y - ymin) / (ymax - ymin) * plot_h
        target = getattr(row, target_col, 0) if target_col in sample.columns else 0
        color = "#d95f5f" if int(float(target or 0)) == 1 else "#5f7fd9"
        parts.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="3" fill="{color}" opacity="0.58"/>')
    parts.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 35}" text-anchor="middle">PC1</text>',
            f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle">PC2</text>',
            f'<circle cx="{left}" cy="{height - 18}" r="4" fill="#d95f5f"/><text x="{left + 10}" y="{height - 14}">{target_col}=1</text>',
            f'<circle cx="{left + 130}" cy="{height - 18}" r="4" fill="#5f7fd9"/><text x="{left + 140}" y="{height - 14}">{target_col}=0</text>',
        ]
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _prepare_pca_matrix(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    matrix = _feature_matrix(df, feature_names)
    if matrix.empty:
        return pd.DataFrame(), []
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    usable_features = [col for col in matrix.columns if matrix[col].nunique(dropna=False) > 1]
    if not usable_features:
        return pd.DataFrame(), []
    return matrix[usable_features], usable_features


def _compute_pca_analysis(
    *,
    prefix: str,
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    output_dir: Path,
    outputs: dict[str, str],
    max_components: int = 5,
) -> dict[str, pd.DataFrame]:
    matrix, usable_features = _prepare_pca_matrix(df, feature_names)
    empty_variance = pd.DataFrame(columns=["component", "explained_variance_ratio", "cumulative_variance_ratio"])
    empty_loadings = pd.DataFrame(columns=["feature", "PC1", "PC2"])
    empty_projection = pd.DataFrame(columns=["record_id", "frame_idx", target_col, "PC1", "PC2"])
    if matrix.empty or len(matrix) < 2:
        variance_df = empty_variance
        loadings_df = empty_loadings
        projection_df = empty_projection
    else:
        component_count = min(max_components, len(usable_features), len(matrix))
        scaled = StandardScaler().fit_transform(matrix)
        pca = PCA(n_components=component_count)
        projection = pca.fit_transform(scaled)
        components = [f"PC{i + 1}" for i in range(component_count)]
        variance_df = pd.DataFrame(
            {
                "component": components,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
            }
        )
        loadings_df = pd.DataFrame(pca.components_.T, columns=components)
        loadings_df.insert(0, "feature", usable_features)
        projection_df = pd.DataFrame(projection[:, : min(2, component_count)], columns=components[: min(2, component_count)])
        if "PC2" not in projection_df.columns:
            projection_df["PC2"] = 0.0
        meta_cols = [col for col in ("record_id", "frame_idx", "box_token", target_col) if col in df.columns]
        projection_df = pd.concat([df[meta_cols].reset_index(drop=True), projection_df[["PC1", "PC2"]]], axis=1)

    variance_path = output_dir / f"{prefix}_pca_explained_variance.csv"
    loadings_path = output_dir / f"{prefix}_pca_loadings.csv"
    projection_path = output_dir / f"{prefix}_pca_projection.csv"
    variance_df.to_csv(variance_path, index=False, encoding="utf-8-sig")
    loadings_df.to_csv(loadings_path, index=False, encoding="utf-8-sig")
    projection_df.to_csv(projection_path, index=False, encoding="utf-8-sig")
    outputs[f"{prefix}_pca_explained_variance"] = str(variance_path)
    outputs[f"{prefix}_pca_loadings"] = str(loadings_path)
    outputs[f"{prefix}_pca_projection"] = str(projection_path)
    return {"variance": variance_df, "loadings": loadings_df, "projection": projection_df}


def _write_pca_variance_svg(path: Path, variance: pd.DataFrame, *, title: str) -> None:
    width = 760
    height = 360
    left = 70
    top = 58
    plot_w = 620
    plot_h = 230
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}.axis{stroke:#555;stroke-width:1}</style>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="24" y="32" class="title">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
    ]
    if variance.empty:
        parts.append(f'<text x="24" y="{top + 40}">无可用 PCA 数据</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return
    rows = variance.head(10).copy()
    bar_gap = 12
    bar_w = max(24, (plot_w - bar_gap * (len(rows) + 1)) / max(len(rows), 1))
    points = []
    for idx, row in enumerate(rows.itertuples(index=False)):
        ratio = float(getattr(row, "explained_variance_ratio", 0.0) or 0.0)
        cumulative = float(getattr(row, "cumulative_variance_ratio", 0.0) or 0.0)
        x = left + bar_gap + idx * (bar_w + bar_gap)
        bar_h = ratio * plot_h
        y = top + plot_h - bar_h
        cx = x + bar_w / 2
        cy = top + plot_h - cumulative * plot_h
        points.append((cx, cy))
        parts.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#5f9ed1" opacity="0.85"/>',
                f'<text x="{cx:.2f}" y="{top + plot_h + 22}" text-anchor="middle">{escape(str(getattr(row, "component", "")))}</text>',
                f'<text x="{cx:.2f}" y="{max(y - 6, top + 12):.2f}" text-anchor="middle">{_format_float(ratio, 2)}</text>',
            ]
        )
    if len(points) > 1:
        path_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(f'<polyline points="{path_points}" fill="none" stroke="#d95f5f" stroke-width="2"/>')
    for x, y in points:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#d95f5f"/>')
    parts.append(f'<text x="{left}" y="{height - 18}">蓝柱：单个主成分解释方差；红线：累计解释方差。</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_pca_scatter_svg(path: Path, projection: pd.DataFrame, *, title: str, target_col: str) -> None:
    width = 760
    height = 520
    left = 80
    top = 58
    plot_w = 620
    plot_h = 380
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;font-size:13px}.title{font-size:20px;font-weight:700}</style>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="24" y="32" class="title">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#ddd"/>',
    ]
    if projection.empty or "PC1" not in projection or "PC2" not in projection:
        parts.append(f'<text x="24" y="{top + 40}">无可用 PCA 投影数据</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return
    x_values = pd.to_numeric(projection["PC1"], errors="coerce")
    y_values = pd.to_numeric(projection["PC2"], errors="coerce")
    valid = projection[x_values.notna() & y_values.notna()].copy()
    if valid.empty:
        parts.append(f'<text x="24" y="{top + 40}">无可用 PCA 投影数据</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return
    x_values = pd.to_numeric(valid["PC1"], errors="coerce")
    y_values = pd.to_numeric(valid["PC2"], errors="coerce")
    xmin, xmax = float(x_values.min()), float(x_values.max())
    ymin, ymax = float(y_values.min()), float(y_values.max())
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    sample = valid if len(valid) <= 1500 else valid.sample(1500, random_state=42)
    for row in sample.itertuples(index=False):
        x = float(getattr(row, "PC1"))
        y = float(getattr(row, "PC2"))
        sx = left + (x - xmin) / (xmax - xmin) * plot_w
        sy = top + plot_h - (y - ymin) / (ymax - ymin) * plot_h
        target = getattr(row, target_col, 0) if target_col in sample.columns else 0
        color = "#d95f5f" if int(float(target or 0)) == 1 else "#5f7fd9"
        parts.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="3" fill="{color}" opacity="0.58"/>')
    parts.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 35}" text-anchor="middle">PC1</text>',
            f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle">PC2</text>',
            f'<circle cx="{left}" cy="{height - 18}" r="4" fill="#d95f5f"/><text x="{left + 10}" y="{height - 14}">{target_col}=1</text>',
            f'<circle cx="{left + 130}" cy="{height - 18}" r="4" fill="#5f7fd9"/><text x="{left + 140}" y="{height - 14}">{target_col}=0</text>',
        ]
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _markdown_table(df: pd.DataFrame, columns: list[str], *, limit: int = 10) -> str:
    if df.empty:
        return "无可用数据。\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, sep]
    for row in df.head(limit).itertuples(index=False):
        values = []
        for column in columns:
            value = getattr(row, column)
            if isinstance(value, float):
                values.append(_format_float(value))
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def _feature_value_stats(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    matrix = _feature_matrix(df, feature_names)
    rows = []
    for feature in feature_names:
        if feature not in matrix:
            continue
        values = pd.to_numeric(matrix[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "non_null_count": int(values.notna().sum()),
                "unique_count": int(values.nunique(dropna=True)),
                "std": 0.0 if pd.isna(values.std()) else float(values.std()),
            }
        )
    return pd.DataFrame(rows)


def _low_value_feature_hints(
    *,
    df: pd.DataFrame,
    feature_names: list[str],
    target_corr: pd.DataFrame,
    high_pairs: pd.DataFrame,
    low_corr_threshold: float = 0.02,
) -> dict[str, pd.DataFrame]:
    stats = _feature_value_stats(df, feature_names)
    if stats.empty:
        empty_constant = pd.DataFrame(columns=["feature", "unique_count", "std", "non_null_count"])
        empty_low = pd.DataFrame(columns=["feature", "correlation", "abs_correlation", "non_null_count"])
        empty_redundant = pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "abs_correlation", "suggested_drop"])
        return {"constant": empty_constant, "low_corr": empty_low, "redundant": empty_redundant}

    constant = stats[(stats["unique_count"] <= 1) | (stats["std"] <= 1e-12)].copy()
    constant = constant.sort_values(["unique_count", "feature"])[["feature", "unique_count", "std", "non_null_count"]]

    constant_features = set(constant["feature"].tolist())
    low_corr = target_corr[
        (target_corr["abs_correlation"] <= low_corr_threshold) & (~target_corr["feature"].isin(constant_features))
    ].copy()
    low_corr = low_corr.sort_values(["abs_correlation", "feature"])

    corr_map = dict(zip(target_corr["feature"], target_corr["abs_correlation"], strict=False))
    redundant_rows = []
    for row in high_pairs.itertuples(index=False):
        feature_a = str(getattr(row, "feature_a"))
        feature_b = str(getattr(row, "feature_b"))
        corr_a = float(corr_map.get(feature_a, 0.0) or 0.0)
        corr_b = float(corr_map.get(feature_b, 0.0) or 0.0)
        suggested_drop = feature_a if corr_a < corr_b else feature_b
        redundant_rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "correlation": float(getattr(row, "correlation", 0.0) or 0.0),
                "abs_correlation": float(getattr(row, "abs_correlation", 0.0) or 0.0),
                "suggested_drop": suggested_drop,
            }
        )
    redundant = pd.DataFrame(redundant_rows)
    if redundant.empty:
        redundant = pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "abs_correlation", "suggested_drop"])
    return {"constant": constant, "low_corr": low_corr, "redundant": redundant}


def _write_low_value_hints(
    *,
    prefix: str,
    output_dir: Path,
    hints: dict[str, pd.DataFrame],
    outputs: dict[str, str],
) -> None:
    constant_path = output_dir / f"{prefix}_low_value_constant_features.csv"
    low_corr_path = output_dir / f"{prefix}_low_value_low_target_correlation.csv"
    redundant_path = output_dir / f"{prefix}_low_value_redundant_pairs.csv"
    hints["constant"].to_csv(constant_path, index=False, encoding="utf-8-sig")
    hints["low_corr"].to_csv(low_corr_path, index=False, encoding="utf-8-sig")
    hints["redundant"].to_csv(redundant_path, index=False, encoding="utf-8-sig")
    outputs[f"{prefix}_low_value_constant_features"] = str(constant_path)
    outputs[f"{prefix}_low_value_low_target_correlation"] = str(low_corr_path)
    outputs[f"{prefix}_low_value_redundant_pairs"] = str(redundant_path)


def _write_feature_analysis(
    *,
    prefix: str,
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    output_dir: Path,
    method: str,
    threshold: float,
    top_n: int,
    outputs: dict[str, str],
) -> dict[str, pd.DataFrame]:
    matrix = _feature_matrix(df, feature_names)
    corr = matrix.corr(method=method) if not matrix.empty else pd.DataFrame()
    target_corr = _target_correlations(df, feature_names, target_col=target_col, method=method)
    high_pairs = _high_correlation_pairs(corr, threshold=threshold, top_n=top_n)

    corr_path = output_dir / f"{prefix}_feature_correlation.csv"
    target_path = output_dir / f"{prefix}_target_correlation.csv"
    pairs_path = output_dir / f"{prefix}_high_correlation_pairs.csv"

    corr.to_csv(corr_path, encoding="utf-8-sig")
    target_corr.to_csv(target_path, index=False, encoding="utf-8-sig")
    high_pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")

    outputs[f"{prefix}_feature_correlation"] = str(corr_path)
    outputs[f"{prefix}_target_correlation"] = str(target_path)
    outputs[f"{prefix}_high_correlation_pairs"] = str(pairs_path)
    return {"corr": corr, "target_corr": target_corr, "high_pairs": high_pairs}


def _write_visual_report(
    *,
    result: CorrelationAnalysisResult,
    output_dir: Path,
    frame_analysis: dict[str, pd.DataFrame],
    box_analysis: dict[str, pd.DataFrame],
    frame_pca: dict[str, pd.DataFrame],
    box_pca: dict[str, pd.DataFrame],
    frame_low_value: dict[str, pd.DataFrame],
    box_low_value: dict[str, pd.DataFrame],
) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = {
        "frame_target_bar": figures_dir / "frame_target_correlation_top.svg",
        "box_target_bar": figures_dir / "box_target_correlation_top.svg",
        "frame_heatmap": figures_dir / "frame_feature_correlation_heatmap.svg",
        "box_heatmap": figures_dir / "box_feature_correlation_heatmap.svg",
        "frame_pairs": figures_dir / "frame_high_correlation_pairs.svg",
        "box_pairs": figures_dir / "box_high_correlation_pairs.svg",
        "frame_pca_variance": figures_dir / "frame_pca_explained_variance.svg",
        "box_pca_variance": figures_dir / "box_pca_explained_variance.svg",
        "frame_pca_scatter": figures_dir / "frame_pca_scatter.svg",
        "box_pca_scatter": figures_dir / "box_pca_scatter.svg",
    }
    _write_target_bar_svg(
        figure_paths["frame_target_bar"],
        frame_analysis["target_corr"],
        title="帧级特征与 is_picking 的相关性 Top 15",
    )
    _write_target_bar_svg(
        figure_paths["box_target_bar"],
        box_analysis["target_corr"],
        title="货框特征与 is_target 的相关性 Top 15",
    )
    _write_heatmap_svg(
        figure_paths["frame_heatmap"],
        frame_analysis["corr"],
        frame_analysis["target_corr"],
        title="帧级 Top 特征相关矩阵",
    )
    _write_heatmap_svg(
        figure_paths["box_heatmap"],
        box_analysis["corr"],
        box_analysis["target_corr"],
        title="货框级 Top 特征相关矩阵",
    )
    _write_pairs_bar_svg(
        figure_paths["frame_pairs"],
        frame_analysis["high_pairs"],
        title="帧级高相关特征对",
    )
    _write_pairs_bar_svg(
        figure_paths["box_pairs"],
        box_analysis["high_pairs"],
        title="货框级高相关特征对",
    )
    _write_pca_variance_svg(
        figure_paths["frame_pca_variance"],
        frame_pca["variance"],
        title="帧级 PCA 解释方差",
    )
    _write_pca_variance_svg(
        figure_paths["box_pca_variance"],
        box_pca["variance"],
        title="货框级 PCA 解释方差",
    )
    _write_pca_scatter_svg(
        figure_paths["frame_pca_scatter"],
        frame_pca["projection"],
        title="帧级 PCA PC1/PC2 投影",
        target_col="is_picking",
    )
    _write_pca_scatter_svg(
        figure_paths["box_pca_scatter"],
        box_pca["projection"],
        title="货框级 PCA PC1/PC2 投影",
        target_col="is_target",
    )

    for key, path in figure_paths.items():
        result.outputs[key] = str(path)

    report_path = output_dir / "correlation_report.md"
    result.outputs["report"] = str(report_path)
    lines = [
        "# 特征相关性分析报告",
        "",
        "## 概览",
        "",
        f"- 输入来源：`{result.input_source}`",
        f"- 相关性方法：`{result.method}`",
        f"- 高相关阈值：`{result.threshold}`",
        f"- 帧级样本数：`{result.frame_count}`",
        f"- 货框级样本数：`{result.box_sample_count}`",
        "",
        "## 帧级特征",
        "",
        "### 与 is_picking 的相关性",
        "",
        f"![帧级特征与 is_picking 的相关性]({_rel_path(figure_paths['frame_target_bar'], output_dir)})",
        "",
        _markdown_table(frame_analysis["target_corr"], ["feature", "correlation", "abs_correlation", "non_null_count"]),
        "",
        "### Top 特征相关矩阵",
        "",
        f"![帧级 Top 特征相关矩阵]({_rel_path(figure_paths['frame_heatmap'], output_dir)})",
        "",
        "### 高相关特征对",
        "",
        f"![帧级高相关特征对]({_rel_path(figure_paths['frame_pairs'], output_dir)})",
        "",
        _markdown_table(frame_analysis["high_pairs"], ["feature_a", "feature_b", "correlation", "abs_correlation"]),
        "",
        "### 低价值/冗余特征提示",
        "",
        "常量或近常量特征，通常可优先移除：",
        "",
        _markdown_table(frame_low_value["constant"], ["feature", "unique_count", "std", "non_null_count"], limit=20),
        "",
        "与 `is_picking` 几乎无相关的特征（默认 `abs(correlation) <= 0.02`）：",
        "",
        _markdown_table(frame_low_value["low_corr"], ["feature", "correlation", "abs_correlation", "non_null_count"], limit=20),
        "",
        "高度冗余特征对，`suggested_drop` 为按目标相关性较低的一侧给出的候选删除项：",
        "",
        _markdown_table(
            frame_low_value["redundant"],
            ["feature_a", "feature_b", "correlation", "abs_correlation", "suggested_drop"],
            limit=20,
        ),
        "",
        "### 主成分分析 PCA",
        "",
        f"![帧级 PCA 解释方差]({_rel_path(figure_paths['frame_pca_variance'], output_dir)})",
        "",
        _markdown_table(frame_pca["variance"], ["component", "explained_variance_ratio", "cumulative_variance_ratio"]),
        "",
        f"![帧级 PCA PC1/PC2 投影]({_rel_path(figure_paths['frame_pca_scatter'], output_dir)})",
        "",
        "PC1/PC2 载荷绝对值较大的特征：",
        "",
        _markdown_table(
            frame_pca["loadings"]
            .assign(_abs=lambda d: d.get("PC1", 0).abs() + d.get("PC2", 0).abs())
            .sort_values("_abs", ascending=False)
            .drop(columns=["_abs"]),
            [col for col in ["feature", "PC1", "PC2"] if col in frame_pca["loadings"].columns],
        ),
        "",
        "## 货框级特征",
        "",
        "### 与 is_target 的相关性",
        "",
        f"![货框特征与 is_target 的相关性]({_rel_path(figure_paths['box_target_bar'], output_dir)})",
        "",
        _markdown_table(box_analysis["target_corr"], ["feature", "correlation", "abs_correlation", "non_null_count"]),
        "",
        "### Top 特征相关矩阵",
        "",
        f"![货框级 Top 特征相关矩阵]({_rel_path(figure_paths['box_heatmap'], output_dir)})",
        "",
        "### 高相关特征对",
        "",
        f"![货框级高相关特征对]({_rel_path(figure_paths['box_pairs'], output_dir)})",
        "",
        _markdown_table(box_analysis["high_pairs"], ["feature_a", "feature_b", "correlation", "abs_correlation"]),
        "",
        "### 低价值/冗余特征提示",
        "",
        "常量或近常量特征，通常可优先移除：",
        "",
        _markdown_table(box_low_value["constant"], ["feature", "unique_count", "std", "non_null_count"], limit=20),
        "",
        "与 `is_target` 几乎无相关的特征（默认 `abs(correlation) <= 0.02`）：",
        "",
        _markdown_table(box_low_value["low_corr"], ["feature", "correlation", "abs_correlation", "non_null_count"], limit=20),
        "",
        "高度冗余特征对，`suggested_drop` 为按目标相关性较低的一侧给出的候选删除项：",
        "",
        _markdown_table(
            box_low_value["redundant"],
            ["feature_a", "feature_b", "correlation", "abs_correlation", "suggested_drop"],
            limit=20,
        ),
        "",
        "### 主成分分析 PCA",
        "",
        f"![货框级 PCA 解释方差]({_rel_path(figure_paths['box_pca_variance'], output_dir)})",
        "",
        _markdown_table(box_pca["variance"], ["component", "explained_variance_ratio", "cumulative_variance_ratio"]),
        "",
        f"![货框级 PCA PC1/PC2 投影]({_rel_path(figure_paths['box_pca_scatter'], output_dir)})",
        "",
        "PC1/PC2 载荷绝对值较大的特征：",
        "",
        _markdown_table(
            box_pca["loadings"]
            .assign(_abs=lambda d: d.get("PC1", 0).abs() + d.get("PC2", 0).abs())
            .sort_values("_abs", ascending=False)
            .drop(columns=["_abs"]),
            [col for col in ["feature", "PC1", "PC2"] if col in box_pca["loadings"].columns],
        ),
        "",
        "## 输出文件",
        "",
    ]
    for name, path in sorted(result.outputs.items()):
        lines.append(f"- `{name}`：`{_rel_path(Path(path), output_dir)}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _load_exported_feature_file(features_dir: Path, stem: str) -> pd.DataFrame:
    for suffix in ("csv", "parquet", "jsonl"):
        path = features_dir / f"{stem}.{suffix}"
        if not path.is_file():
            continue
        if suffix == "csv":
            return pd.read_csv(path)
        if suffix == "parquet":
            return pd.read_parquet(path)
        return pd.read_json(path, lines=True)
    raise FileNotFoundError(f"在 {features_dir} 下未找到 {stem}.csv/.parquet/.jsonl")


def _load_exported_feature_names(features_dir: Path) -> tuple[list[str], list[str]]:
    meta_path = features_dir / "features_meta.json"
    if not meta_path.is_file():
        return [], []
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return list(data.get("frame_feature_names") or []), list(data.get("box_feature_names") or [])


def _infer_feature_names(df: pd.DataFrame, excluded_columns: set[str]) -> list[str]:
    features = []
    for column in df.columns:
        if column in excluded_columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            features.append(str(column))
    return features


def _analyze_feature_dataframes(
    *,
    input_source: str,
    frame_df: pd.DataFrame,
    box_df: pd.DataFrame,
    frame_feature_names: list[str],
    box_feature_names: list[str],
    output_dir: Path,
    method: str = "pearson",
    threshold: float = 0.9,
    top_n: int = 100,
) -> CorrelationAnalysisResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    frame_samples_path = output_dir / "frame_feature_samples.csv"
    box_samples_path = output_dir / "box_feature_samples.csv"
    frame_df.to_csv(frame_samples_path, index=False, encoding="utf-8-sig")
    box_df.to_csv(box_samples_path, index=False, encoding="utf-8-sig")
    outputs["frame_feature_samples"] = str(frame_samples_path)
    outputs["box_feature_samples"] = str(box_samples_path)

    frame_analysis = _write_feature_analysis(
        prefix="frame",
        df=frame_df,
        feature_names=frame_feature_names,
        target_col="is_picking",
        output_dir=output_dir,
        method=method,
        threshold=threshold,
        top_n=top_n,
        outputs=outputs,
    )
    box_analysis = _write_feature_analysis(
        prefix="box",
        df=box_df,
        feature_names=box_feature_names,
        target_col="is_target",
        output_dir=output_dir,
        method=method,
        threshold=threshold,
        top_n=top_n,
        outputs=outputs,
    )
    frame_pca = _compute_pca_analysis(
        prefix="frame",
        df=frame_df,
        feature_names=frame_feature_names,
        target_col="is_picking",
        output_dir=output_dir,
        outputs=outputs,
    )
    box_pca = _compute_pca_analysis(
        prefix="box",
        df=box_df,
        feature_names=box_feature_names,
        target_col="is_target",
        output_dir=output_dir,
        outputs=outputs,
    )
    frame_low_value = _low_value_feature_hints(
        df=frame_df,
        feature_names=frame_feature_names,
        target_corr=frame_analysis["target_corr"],
        high_pairs=frame_analysis["high_pairs"],
    )
    box_low_value = _low_value_feature_hints(
        df=box_df,
        feature_names=box_feature_names,
        target_corr=box_analysis["target_corr"],
        high_pairs=box_analysis["high_pairs"],
    )
    _write_low_value_hints(
        prefix="frame",
        output_dir=output_dir,
        hints=frame_low_value,
        outputs=outputs,
    )
    _write_low_value_hints(
        prefix="box",
        output_dir=output_dir,
        hints=box_low_value,
        outputs=outputs,
    )

    result = CorrelationAnalysisResult(
        input_source=input_source,
        output_dir=str(output_dir),
        method=method,
        threshold=threshold,
        frame_count=len(frame_df),
        box_sample_count=len(box_df),
        outputs=outputs,
    )
    _write_visual_report(
        result=result,
        output_dir=output_dir,
        frame_analysis=frame_analysis,
        box_analysis=box_analysis,
        frame_pca=frame_pca,
        box_pca=box_pca,
        frame_low_value=frame_low_value,
        box_low_value=box_low_value,
    )
    summary_path = output_dir / "correlation_summary.json"
    result.outputs["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def analyze_feature_correlations(
    data_dir: Path,
    output_dir: Path,
    *,
    method: str = "pearson",
    threshold: float = 0.9,
    top_n: int = 100,
) -> CorrelationAnalysisResult:
    """从原始记录目录提取特征并分析相关性。"""
    data_dir = Path(data_dir)
    dataset = load_dataset(data_dir)
    return _analyze_feature_dataframes(
        input_source=str(data_dir),
        frame_df=_frame_dataframe(dataset),
        box_df=_box_dataframe(dataset),
        frame_feature_names=dataset.frame_feature_names,
        box_feature_names=dataset.box_feature_names,
        output_dir=output_dir,
        method=method,
        threshold=threshold,
        top_n=top_n,
    )


def analyze_exported_feature_correlations(
    features_dir: Path,
    output_dir: Path,
    *,
    method: str = "pearson",
    threshold: float = 0.9,
    top_n: int = 100,
) -> CorrelationAnalysisResult:
    """从 export-features 已导出的特征目录分析相关性。"""
    features_dir = Path(features_dir)
    frame_df = _load_exported_feature_file(features_dir, "frame_features")
    box_df = _load_exported_feature_file(features_dir, "box_features")
    frame_feature_names, box_feature_names = _load_exported_feature_names(features_dir)
    if not frame_feature_names:
        frame_feature_names = _infer_feature_names(
            frame_df,
            {"record_id", "frame_idx", "is_picking", "confirmed_box_tokens"},
        )
    if not box_feature_names:
        box_feature_names = _infer_feature_names(
            box_df,
            {"record_id", "frame_idx", "box_token", "is_target"},
        )
    return _analyze_feature_dataframes(
        input_source=str(features_dir),
        frame_df=frame_df,
        box_df=box_df,
        frame_feature_names=frame_feature_names,
        box_feature_names=box_feature_names,
        output_dir=output_dir,
        method=method,
        threshold=threshold,
        top_n=top_n,
    )
