#!/usr/bin/env python3
"""Render the official v1.0 continuous-batching benchmark as pure SVG."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from typing import Iterable


EQUAL_CASES = ("equal_n1", "equal_n2", "equal_n4", "equal_n8", "equal_n16")
MIXED_CASES = ("mixed_a1", "mixed_a4", "mixed_a8", "staggered_a8")
VARIANTS = ("serial", "continuous")
COLORS = {"serial": "#dc2626", "continuous": "#16a34a"}
LABELS = {"serial": "Serial scheduler", "continuous": "Continuous batching"}
CASE_LABELS = {
    "mixed_a1": "Mixed A=1",
    "mixed_a4": "Mixed A=4",
    "mixed_a8": "Mixed A=8",
    "staggered_a8": "Staggered A=8",
}


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


def line_path(points: list[tuple[float, float]]) -> str:
    return " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def nice_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for step in (1.0, 2.0, 2.5, 4.0, 5.0, 8.0, 10.0):
        if normalized <= step:
            return step * magnitude
    raise AssertionError("unreachable")


def number(value: float) -> str:
    return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"


def add_positive_axes(
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
        lines.append(svg_text(x0 - 8, y + 4, number(value), fill="#6b7280", anchor="end"))
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


def validate_and_select(
    run_dir: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, dict[str, str]],
    int,
    int,
]:
    performance_path = run_dir / "performance_summary.csv"
    batch_path = run_dir / "batch_summary.csv"
    correctness_path = run_dir / "correctness_summary.csv"
    performance = read_csv(performance_path)
    batches = read_csv(batch_path)
    correctness = read_csv(correctness_path)

    require_fields(
        performance,
        (
            "config_id",
            "category",
            "variant",
            "request_count",
            "max_active_requests",
            "repeats",
            "median_aggregate_output_tokens_per_s",
            "p95_aggregate_output_tokens_per_s",
            "paired_median_change_aggregate_output_tokens_per_s_pct",
            "median_mean_batch_size",
            "median_padding_ratio",
        ),
        performance_path,
    )
    require_fields(
        batches,
        (
            "config_id",
            "category",
            "repeats",
            "median_mean_batch_size",
            "median_padding_ratio",
            "median_decode_tokens_per_enqueue",
        ),
        batch_path,
    )
    require_fields(
        correctness,
        (
            "exact_match",
            "finish_reason",
            "expected_finish_reason",
            "reference_tokens",
            "candidate_tokens",
        ),
        correctness_path,
    )

    bad_correctness = [
        row
        for row in correctness
        if row["exact_match"] != "1"
        or row["finish_reason"] != row["expected_finish_reason"]
        or row["reference_tokens"] != row["candidate_tokens"]
    ]
    if bad_correctness:
        raise RuntimeError(
            f"refusing to plot a run with {len(bad_correctness)} correctness failures"
        )

    expected_cases = set(EQUAL_CASES + MIXED_CASES)
    selected = [
        row
        for row in performance
        if row["category"] == "performance"
        and row["config_id"] in expected_cases
        and row["variant"] in VARIANTS
    ]
    if len(selected) != len(expected_cases) * len(VARIANTS):
        raise RuntimeError(
            f"expected {len(expected_cases) * len(VARIANTS)} performance rows, "
            f"found {len(selected)}"
        )
    row_map: dict[tuple[str, str], dict[str, str]] = {}
    for row in selected:
        key = (row["config_id"], row["variant"])
        if key in row_map:
            raise RuntimeError(f"duplicate performance summary row: {key}")
        if int(row["repeats"]) < 10:
            raise RuntimeError(f"official plot requires >=10 repeats: {key}")
        row_map[key] = row

    for case in expected_cases:
        serial = row_map[(case, "serial")]
        continuous = row_map[(case, "continuous")]
        if serial["repeats"] != continuous["repeats"]:
            raise RuntimeError(f"unpaired repeat counts for {case}")
        if serial["request_count"] != continuous["request_count"]:
            raise RuntimeError(f"unpaired request counts for {case}")
        if serial["max_active_requests"] != continuous["max_active_requests"]:
            raise RuntimeError(f"unpaired active-request limits for {case}")

    expected_equal = {
        "equal_n1": (1, 1),
        "equal_n2": (2, 2),
        "equal_n4": (4, 4),
        "equal_n8": (8, 8),
        "equal_n16": (16, 8),
    }
    for case, (requests, active) in expected_equal.items():
        row = row_map[(case, "continuous")]
        if (int(row["request_count"]), int(row["max_active_requests"])) != (
            requests,
            active,
        ):
            raise RuntimeError(f"unexpected equal-length configuration: {case}")

    batch_map: dict[str, dict[str, str]] = {}
    for row in batches:
        case = row["config_id"]
        if row["category"] != "performance" or case not in expected_cases:
            continue
        if case in batch_map:
            raise RuntimeError(f"duplicate continuous batch summary row: {case}")
        if int(row["repeats"]) < 10:
            raise RuntimeError(f"official batch summary requires >=10 repeats: {case}")
        batch_map[case] = row
    missing_batches = expected_cases.difference(batch_map)
    if missing_batches:
        raise RuntimeError(f"missing continuous batch summaries: {sorted(missing_batches)}")

    for case in expected_cases:
        perf = row_map[(case, "continuous")]
        batch = batch_map[case]
        for field in ("median_mean_batch_size", "median_padding_ratio"):
            if not math.isclose(float(perf[field]), float(batch[field]), abs_tol=1e-6):
                raise RuntimeError(f"performance/batch {field} mismatch for {case}")

    equal_rows = [row_map[(case, variant)] for case in EQUAL_CASES for variant in VARIANTS]
    mixed_rows = [row_map[(case, "continuous")] for case in MIXED_CASES]
    repeat_count = min(int(row["repeats"]) for row in selected)
    return equal_rows, mixed_rows, batch_map, len(correctness), repeat_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plots/v10_continuous_batching.svg"),
    )
    args = parser.parse_args()

    equal_rows, mixed_rows, batch_map, correctness_count, repeats = validate_and_select(
        args.run_dir
    )

    width, height = 1240, 540
    panel_y, panel_h = 122.0, 294.0
    left_x, left_w = 78.0, 500.0
    right_x, right_w = 680.0, 520.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(
            28,
            34,
            "MiniLLM-RT v1.0 · Continuous-batching benchmark",
            size=21,
            weight=700,
        ),
        svg_text(28, 58, f"Official run: {args.run_dir.name}", fill="#4b5563"),
        svg_text(left_x, 94, "A. Equal-length throughput scaling", size=15, weight=700),
        svg_text(
            right_x,
            94,
            "B. Mixed/staggered paired throughput change",
            size=15,
            weight=700,
        ),
        svg_text(
            right_x,
            112,
            "Continuous vs serial; context labels report continuous mean batch and padding",
            size=10,
            fill="#6b7280",
        ),
    ]

    left_y_max = nice_upper(
        max(float(row["p95_aggregate_output_tokens_per_s"]) for row in equal_rows) * 1.05
    )
    add_positive_axes(
        lines,
        x0=left_x,
        y0=panel_y,
        width=left_w,
        height=panel_h,
        y_max=left_y_max,
        y_label="Aggregate output tokens/s",
    )
    equal_positions = {
        case: left_x + 34 + index * (left_w - 68) / (len(EQUAL_CASES) - 1)
        for index, case in enumerate(EQUAL_CASES)
    }
    left_py = lambda value: panel_y + panel_h - value / left_y_max * panel_h
    equal_map = {(row["config_id"], row["variant"]): row for row in equal_rows}
    for case in EQUAL_CASES:
        row = equal_map[(case, "continuous")]
        x = equal_positions[case]
        lines.append(
            svg_text(
                x,
                panel_y + panel_h + 22,
                row["request_count"],
                fill="#4b5563",
                anchor="middle",
            )
        )
    lines.append(
        svg_text(
            left_x + left_w / 2,
            panel_y + panel_h + 46,
            "Requests N (max active = min(N, 8))",
            fill="#374151",
            anchor="middle",
        )
    )
    for variant in VARIANTS:
        rows = [equal_map[(case, variant)] for case in EQUAL_CASES]
        median_points = [
            (
                equal_positions[row["config_id"]],
                left_py(float(row["median_aggregate_output_tokens_per_s"])),
            )
            for row in rows
        ]
        p95_points = [
            (
                equal_positions[row["config_id"]],
                left_py(float(row["p95_aggregate_output_tokens_per_s"])),
            )
            for row in rows
        ]
        color = COLORS[variant]
        lines.append(
            f'<path d="{line_path(p95_points)}" fill="none" stroke="{color}" '
            'stroke-width="1.3" stroke-dasharray="4 4" opacity="0.42"/>'
        )
        lines.append(
            f'<path d="{line_path(median_points)}" fill="none" stroke="{color}" '
            'stroke-width="2.8"/>'
        )
        for x, y in median_points:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        if variant == "continuous":
            for row, (x, y) in zip(rows, median_points):
                change = float(
                    row["paired_median_change_aggregate_output_tokens_per_s_pct"]
                )
                lines.append(
                    svg_text(
                        x,
                        y - 11,
                        f"{change:+.1f}%",
                        size=10,
                        fill=color if change >= 0 else COLORS["serial"],
                        weight=700,
                        anchor="middle",
                    )
                )

    changes = [
        float(row["paired_median_change_aggregate_output_tokens_per_s_pct"])
        for row in mixed_rows
    ]
    y_step = 20.0
    change_min = min(-y_step, math.floor(min(0.0, min(changes) * 1.2) / y_step) * y_step)
    change_max = max(y_step, math.ceil(max(0.0, max(changes) * 1.18) / y_step) * y_step)
    right_py = lambda value: panel_y + (change_max - value) / (change_max - change_min) * panel_h
    for value in range(int(change_min), int(change_max) + 1, int(y_step)):
        y = right_py(float(value))
        stroke = "#9ca3af" if value == 0 else "#e5e7eb"
        width_value = "1.3" if value == 0 else "1"
        lines.append(
            f'<line x1="{right_x:.1f}" y1="{y:.1f}" x2="{right_x + right_w:.1f}" '
            f'y2="{y:.1f}" stroke="{stroke}" stroke-width="{width_value}"/>'
        )
        lines.append(
            svg_text(right_x - 8, y + 4, f"{value:+d}%", fill="#6b7280", anchor="end")
        )
    lines.append(
        f'<line x1="{right_x:.1f}" y1="{panel_y:.1f}" x2="{right_x:.1f}" '
        f'y2="{panel_y + panel_h:.1f}" stroke="#4b5563"/>'
    )
    lines.append(
        f'<text x="{right_x - 55:.1f}" y="{panel_y + panel_h / 2:.1f}" '
        'font-family="Arial, sans-serif" font-size="12" fill="#374151" '
        f'text-anchor="middle" transform="rotate(-90 {right_x - 55:.1f} {panel_y + panel_h / 2:.1f})">'
        'Paired median throughput change</text>'
    )

    mixed_positions = {
        case: right_x + 62 + index * (right_w - 124) / (len(MIXED_CASES) - 1)
        for index, case in enumerate(MIXED_CASES)
    }
    zero_y = right_py(0.0)
    bar_width = 62.0
    mixed_map = {row["config_id"]: row for row in mixed_rows}
    for case in MIXED_CASES:
        row = mixed_map[case]
        value = float(row["paired_median_change_aggregate_output_tokens_per_s_pct"])
        value_y = right_py(value)
        color = COLORS["continuous"] if value >= 0 else COLORS["serial"]
        x = mixed_positions[case]
        top = min(zero_y, value_y)
        bar_height = max(1.0, abs(zero_y - value_y))
        lines.append(
            f'<rect x="{x - bar_width / 2:.2f}" y="{top:.2f}" width="{bar_width:.2f}" '
            f'height="{bar_height:.2f}" rx="3" fill="{color}" opacity="0.88"/>'
        )
        label_y = value_y - 8 if value >= 0 else value_y + 17
        lines.append(
            svg_text(
                x,
                label_y,
                f"{value:+.1f}%",
                size=11,
                fill=color,
                weight=700,
                anchor="middle",
            )
        )
        batch = batch_map[case]
        mean_batch = float(batch["median_mean_batch_size"])
        padding = float(batch["median_padding_ratio"]) * 100.0
        lines.append(
            svg_text(
                x,
                panel_y + panel_h + 22,
                CASE_LABELS[case],
                fill="#374151",
                weight=600,
                anchor="middle",
            )
        )
        lines.append(
            svg_text(
                x,
                panel_y + panel_h + 41,
                f"mean B={mean_batch:.2f} · pad={padding:.1f}%",
                size=10,
                fill="#6b7280",
                anchor="middle",
            )
        )

    legend_x = 815.0
    for index, variant in enumerate(VARIANTS):
        x = legend_x + index * 184
        lines.append(
            f'<line x1="{x:.1f}" y1="62" x2="{x + 28:.1f}" y2="62" '
            f'stroke="{COLORS[variant]}" stroke-width="3"/>'
        )
        lines.append(svg_text(x + 35, 66, LABELS[variant], fill="#374151"))

    lines.append(
        svg_text(
            28,
            516,
            f"Median of {repeats} measured runs · alternating paired A/B order · "
            f"dashed=p95 across runs · {correctness_count:,} request-level token comparisons exact",
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
