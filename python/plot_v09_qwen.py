#!/usr/bin/env python3
"""Plot the official v0.9 Qwen KV-cache benchmark summaries as SVG."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Iterable


CONTROLLED_CASES = ("perf_p8", "perf_p32", "perf_p128", "perf_p256")
COLORS = {"full_recompute": "#dc2626", "kv_cache": "#16a34a"}
LABELS = {"full_recompute": "Full recompute", "kv_cache": "KV cache"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"missing required summary: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"summary is empty: {path}")
    return rows


def require_fields(rows: list[dict[str, str]], fields: Iterable[str], path: Path) -> None:
    missing = set(fields).difference(rows[0])
    if missing:
        raise RuntimeError(f"{path} lacks columns: {sorted(missing)}")


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 12,
    fill: str = "#111827",
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}" font-weight="{weight}" '
        f'text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** (len(str(int(value))) - 1)
    normalized = value / magnitude
    step = 1.0 if normalized <= 1 else 2.0 if normalized <= 2 else 5.0 if normalized <= 5 else 10.0
    return step * magnitude


def line_path(points: list[tuple[float, float]]) -> str:
    return " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def add_axes(
    lines: list[str],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_max: float,
    y_label: str,
) -> None:
    for tick in range(6):
        value = y_max * tick / 5
        y = y0 + height - height * tick / 5
        lines.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + width:.1f}" y2="{y:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(svg_text(x0 - 8, y + 4, f"{value:.1f}", fill="#6b7280", anchor="end"))
    lines.append(
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0 + height:.1f}" '
        'stroke="#4b5563"/>'
    )
    lines.append(
        f'<line x1="{x0:.1f}" y1="{y0 + height:.1f}" x2="{x0 + width:.1f}" '
        f'y2="{y0 + height:.1f}" stroke="#4b5563"/>'
    )
    lines.append(
        f'<text x="{x0 - 50:.1f}" y="{y0 + height / 2:.1f}" '
        'font-family="Arial, sans-serif" font-size="12" fill="#374151" '
        f'text-anchor="middle" transform="rotate(-90 {x0 - 50:.1f} {y0 + height / 2:.1f})">'
        f'{html.escape(y_label)}</text>'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plots/v09_qwen_kv_cache.svg"),
    )
    args = parser.parse_args()

    performance_path = args.run_dir / "performance_summary.csv"
    decode_path = args.run_dir / "decode_step_summary.csv"
    correctness_path = args.run_dir / "correctness_summary.csv"
    performance = read_csv(performance_path)
    decode = read_csv(decode_path)
    correctness = read_csv(correctness_path)
    require_fields(
        performance,
        (
            "case_id",
            "prompt_len",
            "requested_output_len",
            "variant",
            "repeats",
            "median_mean_tpot_ms",
            "p95_mean_tpot_ms",
            "paired_median_change_mean_tpot_ms_pct",
        ),
        performance_path,
    )
    require_fields(
        decode,
        (
            "case_id",
            "prompt_len",
            "requested_output_len",
            "variant",
            "history_len",
            "repeats",
            "median_step_total_ms",
            "p95_step_total_ms",
        ),
        decode_path,
    )
    require_fields(correctness, ("exact_match",), correctness_path)
    mismatches = [row for row in correctness if row["exact_match"] != "1"]
    if mismatches:
        raise RuntimeError(f"refusing to plot a run with {len(mismatches)} correctness mismatches")

    controlled = [
        row
        for row in performance
        if row["case_id"] in CONTROLLED_CASES
        and int(row["requested_output_len"]) == 32
        and row["variant"] in COLORS
    ]
    if len(controlled) != len(CONTROLLED_CASES) * 2:
        raise RuntimeError(f"expected eight controlled summary rows, found {len(controlled)}")
    if any(int(row["repeats"]) < 10 for row in controlled):
        raise RuntimeError("official plot requires at least 10 measured repeats")

    trend = [
        row
        for row in decode
        if row["case_id"] == "perf_p256"
        and int(row["requested_output_len"]) == 32
        and row["variant"] in COLORS
    ]
    by_variant = {
        variant: sorted(
            (row for row in trend if row["variant"] == variant),
            key=lambda row: int(row["history_len"]),
        )
        for variant in COLORS
    }
    expected_history = {
        int(row["history_len"]) for row in by_variant["full_recompute"]
    }
    if len(expected_history) != 31 or {
        int(row["history_len"]) for row in by_variant["kv_cache"]
    } != expected_history:
        raise RuntimeError("p256 decode trend does not contain matching 31-step full/KV histories")

    width, height = 1160, 500
    panel_y, panel_h, panel_w = 112.0, 292.0, 455.0
    left_x, right_x = 78.0, 650.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(28, 34, "MiniLLM-RT v0.9 · Qwen2.5-0.5B KV-cache benchmark", size=21, weight=700),
        svg_text(28, 58, f"Official run: {args.run_dir.name}", fill="#4b5563"),
        svg_text(left_x, 91, "A. Token-ready TPOT by prompt length", size=15, weight=700),
        svg_text(right_x, 91, "B. Decode latency as history grows (prompt=256)", size=15, weight=700),
    ]

    left_y_max = nice_max(max(float(row["p95_mean_tpot_ms"]) for row in controlled) * 1.05)
    add_axes(
        lines,
        x0=left_x,
        y0=panel_y,
        width=panel_w,
        height=panel_h,
        y_max=left_y_max,
        y_label="TPOT (ms, lower is better)",
    )
    case_positions = {
        case: left_x + 34 + index * (panel_w - 68) / (len(CONTROLLED_CASES) - 1)
        for index, case in enumerate(CONTROLLED_CASES)
    }
    left_y = lambda value: panel_y + panel_h - value / left_y_max * panel_h
    for case in CONTROLLED_CASES:
        row = next(row for row in controlled if row["case_id"] == case)
        x = case_positions[case]
        lines.append(svg_text(x, panel_y + panel_h + 22, row["prompt_len"], fill="#4b5563", anchor="middle"))
    lines.append(svg_text(left_x + panel_w / 2, panel_y + panel_h + 45, "Prompt tokens", fill="#374151", anchor="middle"))
    for variant in COLORS:
        rows = sorted(
            (row for row in controlled if row["variant"] == variant),
            key=lambda row: int(row["prompt_len"]),
        )
        median_points = [
            (case_positions[row["case_id"]], left_y(float(row["median_mean_tpot_ms"])))
            for row in rows
        ]
        p95_points = [
            (case_positions[row["case_id"]], left_y(float(row["p95_mean_tpot_ms"])))
            for row in rows
        ]
        color = COLORS[variant]
        lines.append(
            f'<path d="{line_path(p95_points)}" fill="none" stroke="{color}" '
            'stroke-width="1.3" stroke-dasharray="4 4" opacity="0.42"/>'
        )
        lines.append(
            f'<path d="{line_path(median_points)}" fill="none" stroke="{color}" stroke-width="2.8"/>'
        )
        for x, y in median_points:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        if variant == "kv_cache":
            for row, (x, y) in zip(rows, median_points):
                change = float(row["paired_median_change_mean_tpot_ms_pct"])
                lines.append(svg_text(x, y - 10, f"{change:+.1f}%", size=10, fill=color, weight=700, anchor="middle"))

    right_y_max = nice_max(
        max(float(row["p95_step_total_ms"]) for rows in by_variant.values() for row in rows) * 1.05
    )
    add_axes(
        lines,
        x0=right_x,
        y0=panel_y,
        width=panel_w,
        height=panel_h,
        y_max=right_y_max,
        y_label="Per-token latency (ms)",
    )
    min_history, max_history = min(expected_history), max(expected_history)
    right_px = lambda value: right_x + (value - min_history) / (max_history - min_history) * panel_w
    right_py = lambda value: panel_y + panel_h - value / right_y_max * panel_h
    for history in (min_history, min_history + 8, min_history + 16, min_history + 24, max_history):
        x = right_px(history)
        lines.append(svg_text(x, panel_y + panel_h + 22, str(history), fill="#4b5563", anchor="middle"))
    lines.append(svg_text(right_x + panel_w / 2, panel_y + panel_h + 45, "History tokens", fill="#374151", anchor="middle"))
    for variant, rows in by_variant.items():
        color = COLORS[variant]
        median_points = [
            (right_px(int(row["history_len"])), right_py(float(row["median_step_total_ms"])))
            for row in rows
        ]
        p95_points = [
            (right_px(int(row["history_len"])), right_py(float(row["p95_step_total_ms"])))
            for row in rows
        ]
        lines.append(
            f'<path d="{line_path(p95_points)}" fill="none" stroke="{color}" '
            'stroke-width="1.2" stroke-dasharray="4 4" opacity="0.42"/>'
        )
        lines.append(
            f'<path d="{line_path(median_points)}" fill="none" stroke="{color}" stroke-width="2.6"/>'
        )

    legend_x = 774.0
    for index, variant in enumerate(COLORS):
        x = legend_x + index * 165
        lines.append(f'<line x1="{x:.1f}" y1="62" x2="{x + 28:.1f}" y2="62" stroke="{COLORS[variant]}" stroke-width="3"/>')
        lines.append(svg_text(x + 35, 66, LABELS[variant], fill="#374151"))
    lines.append(
        svg_text(
            28,
            478,
            "Median of 10 measured runs · alternating paired A/B order · dashed=p95 across runs · all token comparisons exact",
            size=11,
            fill="#6b7280",
        )
    )
    lines.append("</svg>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"plot: {args.output}")


if __name__ == "__main__":
    main()
