#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict, cast


class SummaryRecord(TypedDict):
    input_type_label: str
    pipeline_id: str
    pipeline_name: str
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    mean_utility_score: float
    mean_phi_field_recall: float
    mean_field_accuracy: float
    mean_text_score: float
    mean_box_coverage: float
    mean_speed_score: float
    mean_stability_score: float
    mean_latency_ms: float


class SampleRecord(TypedDict):
    input_type_label: str
    pipeline_id: str
    pipeline_name: str
    sample_id: str
    status: str
    latency_ms: float
    utility_score: float
    phi_field_recall: float


FigureFormat = Literal["pdf", "svg", "png"]

COMPONENT_METRICS: tuple[tuple[str, str], ...] = (
    ("mean_phi_field_recall", "PHI recall"),
    ("mean_field_accuracy", "Field accuracy"),
    ("mean_text_score", "Text score"),
    ("mean_box_coverage", "Box coverage"),
    ("mean_speed_score", "Speed"),
    ("mean_stability_score", "Stability"),
)

MODALITY_COLORS: dict[str, str] = {
    "Video Frame": "#1f77b4",
    "Text Report Document": "#2ca02c",
}


def load_evaluation_jsonl(path: Path) -> tuple[list[SummaryRecord], list[SampleRecord]]:
    summaries: list[SummaryRecord] = []
    samples: list[SampleRecord] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed: object = json.loads(line)
        if not isinstance(parsed, Mapping):
            raise ValueError(f"Line {line_number} is not a JSON object")
        record = cast(Mapping[str, object], parsed)
        event = _string(record.get("event"))
        if event == "summary":
            summaries.append(_summary_record(record, line_number))
        elif event == "sample_result":
            samples.append(_sample_record(record, line_number))
    if not summaries:
        raise ValueError(f"No summary rows found in {path}")
    return summaries, samples


def _summary_record(record: Mapping[str, object], line_number: int) -> SummaryRecord:
    return {
        "input_type_label": _required_string(record, "input_type_label", line_number),
        "pipeline_id": _required_string(record, "pipeline_id", line_number),
        "pipeline_name": _required_string(record, "pipeline_name", line_number),
        "attempted": _int(record.get("attempted")),
        "succeeded": _int(record.get("succeeded")),
        "failed": _int(record.get("failed")),
        "skipped": _int(record.get("skipped")),
        "mean_utility_score": _float(record.get("mean_utility_score")),
        "mean_phi_field_recall": _float(record.get("mean_phi_field_recall")),
        "mean_field_accuracy": _float(record.get("mean_field_accuracy")),
        "mean_text_score": _float(record.get("mean_text_score")),
        "mean_box_coverage": _float(record.get("mean_box_coverage")),
        "mean_speed_score": _float(record.get("mean_speed_score")),
        "mean_stability_score": _float(record.get("mean_stability_score")),
        "mean_latency_ms": _float(record.get("mean_latency_ms")),
    }


def _sample_record(record: Mapping[str, object], line_number: int) -> SampleRecord:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Line {line_number} sample_result has no metrics object")
    metric_mapping = cast(Mapping[str, object], metrics)
    return {
        "input_type_label": _required_string(record, "input_type_label", line_number),
        "pipeline_id": _required_string(record, "pipeline_id", line_number),
        "pipeline_name": _required_string(record, "pipeline_name", line_number),
        "sample_id": _required_string(record, "sample_id", line_number),
        "status": _required_string(record, "status", line_number),
        "latency_ms": _float(record.get("latency_ms")),
        "utility_score": _float(metric_mapping.get("utility_score")),
        "phi_field_recall": _float(metric_mapping.get("phi_field_recall")),
    }


def make_figures(
    summaries: Sequence[SummaryRecord],
    samples: Sequence[SampleRecord],
    output_dir: Path,
    formats: Sequence[FigureFormat],
) -> list[Path]:
    plt, np = _load_plotting_modules()
    _configure_matplotlib(plt)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    written.extend(_plot_utility_ranking(plt, summaries, output_dir, formats))
    written.extend(_plot_metric_heatmap(plt, np, summaries, output_dir, formats))
    if samples:
        written.extend(_plot_latency_scatter(plt, summaries, samples, output_dir, formats))
    return written


def _plot_utility_ranking(
    plt: Any,
    summaries: Sequence[SummaryRecord],
    output_dir: Path,
    formats: Sequence[FigureFormat],
) -> list[Path]:
    ordered = sorted(
        summaries,
        key=lambda row: (row["input_type_label"], row["mean_utility_score"]),
    )
    labels = [f"{row['pipeline_id']}\n{row['input_type_label']}" for row in ordered]
    values = [row["mean_utility_score"] for row in ordered]
    colors = [
        MODALITY_COLORS.get(row["input_type_label"], "#7f7f7f") for row in ordered
    ]

    height = max(3.5, 0.45 * len(ordered) + 1.0)
    fig, ax = plt.subplots(figsize=(7.2, height), constrained_layout=True)
    y_positions = list(range(len(ordered)))
    ax.barh(y_positions, values, color=colors, edgecolor="#222222", linewidth=0.4)
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean anonymization utility score")
    ax.set_title("OCR backend ranking by anonymization utility")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)

    for y_position, value in zip(y_positions, values, strict=True):
        ax.text(
            min(value + 0.015, 0.98),
            y_position,
            f"{value:.2f}",
            va="center",
            ha="left" if value < 0.93 else "right",
            fontsize=8,
        )

    return _save_figure(fig, output_dir / "ocr_backend_utility_ranking", formats)


def _plot_metric_heatmap(
    plt: Any,
    np: Any,
    summaries: Sequence[SummaryRecord],
    output_dir: Path,
    formats: Sequence[FigureFormat],
) -> list[Path]:
    ordered = sorted(
        summaries,
        key=lambda row: (row["input_type_label"], row["pipeline_id"]),
    )
    row_labels = [f"{row['pipeline_id']} {row['input_type_label']}" for row in ordered]
    col_labels = [label for _, label in COMPONENT_METRICS]
    matrix = np.array(
        [[row[key] for key, _label in COMPONENT_METRICS] for row in ordered],
        dtype=float,
    )

    width = max(7.2, 0.85 * len(COMPONENT_METRICS) + 2.5)
    height = max(3.5, 0.42 * len(ordered) + 1.3)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title("Utility score components")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = float(matrix[row_index, col_index])
            text_color = "white" if value < 0.45 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.set_label("Normalized score")
    return _save_figure(fig, output_dir / "ocr_backend_component_heatmap", formats)


def _plot_latency_scatter(
    plt: Any,
    summaries: Sequence[SummaryRecord],
    samples: Sequence[SampleRecord],
    output_dir: Path,
    formats: Sequence[FigureFormat],
) -> list[Path]:
    utility_by_key = {
        (row["input_type_label"], row["pipeline_id"]): row["mean_utility_score"]
        for row in summaries
    }
    labels = sorted(
        {
            (row["input_type_label"], row["pipeline_id"], row["pipeline_name"])
            for row in samples
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for input_label, pipeline_id, pipeline_name in labels:
        group = [
            row
            for row in samples
            if row["input_type_label"] == input_label and row["pipeline_id"] == pipeline_id
        ]
        if not group:
            continue
        latencies = [max(row["latency_ms"], 0.001) for row in group]
        utilities = [row["utility_score"] for row in group]
        color = MODALITY_COLORS.get(input_label, "#7f7f7f")
        ax.scatter(
            latencies,
            utilities,
            s=26,
            alpha=0.55,
            color=color,
            edgecolor="none",
        )
        mean_latency = sum(latencies) / len(latencies)
        mean_utility = utility_by_key.get((input_label, pipeline_id), 0.0)
        ax.scatter(
            [mean_latency],
            [mean_utility],
            s=95,
            marker="D",
            color=color,
            edgecolor="#222222",
            linewidth=0.6,
            label=f"{pipeline_id} {pipeline_name}",
        )

    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Latency per sample (ms, log scale)")
    ax.set_ylabel("Anonymization utility score")
    ax.set_title("Utility and throughput trade-off")
    ax.grid(color="#dddddd", linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, loc="best")
    return _save_figure(fig, output_dir / "ocr_backend_latency_tradeoff", formats)


def _save_figure(fig: Any, base_path: Path, formats: Sequence[FigureFormat]) -> list[Path]:
    written: list[Path] = []
    for figure_format in formats:
        output_path = base_path.with_suffix(f".{figure_format}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        written.append(output_path)
    fig.clear()
    return written


def _configure_matplotlib(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _load_plotting_modules() -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Plotting requires matplotlib and numpy. Install them in your active "
            "environment, for example: uv pip install matplotlib"
        ) from exc
    return plt, np


def _parse_formats(raw: str) -> tuple[FigureFormat, ...]:
    formats: list[FigureFormat] = []
    valid = {"pdf", "svg", "png"}
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value not in valid:
            raise ValueError(f"Unsupported figure format: {value}")
        formats.append(cast(FigureFormat, value))
    if not formats:
        raise ValueError("At least one output format is required")
    return tuple(formats)


def _required_string(
    record: Mapping[str, object], key: str, line_number: int
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Line {line_number} field {key!r} must be a non-empty string")
    return value


def _string(value: object | None) -> str:
    return value if isinstance(value, str) else ""


def _float(value: object | None) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _int(value: object | None) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create publication-style figures from OCR backend matrix JSONL."
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        type=Path,
        help="JSONL output produced by manage.py evaluate_ocr_backend_matrix.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for generated figure files.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,svg,png",
        help="Comma-separated output formats: pdf,svg,png (default: pdf,svg,png).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    formats = _parse_formats(cast(str, args.formats))
    summaries, samples = load_evaluation_jsonl(cast(Path, args.input_jsonl))
    written = make_figures(
        summaries=summaries,
        samples=samples,
        output_dir=cast(Path, args.output_dir),
        formats=formats,
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
