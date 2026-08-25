#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render routing-overhead results as a dependency-free SVG heatmap."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TypeGuard


@dataclass(frozen=True)
class OverheadPlotRow:
    """One route and workload comparison against the direct backend."""

    scenario: str
    load: str
    route: str
    ttft_delta_ms: float | None
    ttft_delta_pct: float | None
    token_throughput_delta: float | None
    token_throughput_delta_pct: float | None


@dataclass(frozen=True)
class _Panel:
    metric: str
    title: str
    subtitle: str
    primary_suffix: str
    primary_values: tuple[float | None, ...]
    percent_values: tuple[float | None, ...]
    higher_is_better: bool


_BACKGROUND = (255, 255, 255)
_BETTER = (42, 111, 174)
_WORSE = (197, 52, 52)
_NEUTRAL = "#f3f5f7"


def _finite(value: float | None) -> TypeGuard[float]:
    return value is not None and math.isfinite(value)


def _fill(value: float | None, maximum: float, higher_is_better: bool) -> str:
    if not _finite(value) or value == 0:
        return _NEUTRAL
    worse = value < 0 if higher_is_better else value > 0
    target = _WORSE if worse else _BETTER
    intensity = 0.16 + 0.64 * min(abs(value) / maximum, 1.0)
    channels = (
        round(background + (foreground - background) * intensity)
        for background, foreground in zip(_BACKGROUND, target, strict=True)
    )
    red, green, blue = channels
    return f"#{red:02x}{green:02x}{blue:02x}"


def _format_value(value: float | None, suffix: str) -> str:
    if not _finite(value):
        return "n/a"
    return f"{value:+,.2f}{suffix}"


def _text(
    x: float,
    y: float,
    value: str,
    anchor: str = "start",
    size: int = 13,
    weight: int = 400,
    fill: str = "#1d2733",
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="system-ui, '
        f'-apple-system, sans-serif" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}">{escape(value)}</text>'
    )


def write_overhead_plot(path: Path, rows: Sequence[OverheadPlotRow]) -> None:
    """Write an annotated SVG overview for every supplied route and workload."""
    if not rows:
        raise ValueError("an overhead plot requires at least one row")

    workloads = list(dict.fromkeys((row.scenario, row.load) for row in rows))
    routes = list(dict.fromkeys(row.route for row in rows))
    by_key: dict[tuple[str, str, str], OverheadPlotRow] = {}
    for row in rows:
        key = (row.scenario, row.load, row.route)
        if key in by_key:
            raise RuntimeError(
                f"duplicate overhead plot row for {row.scenario}, {row.load}, and {row.route}"
            )
        by_key[key] = row

    ordered_rows = [
        by_key.get((scenario, load, route)) for scenario, load in workloads for route in routes
    ]
    panels = (
        _Panel(
            "ttft_p50_delta_ms",
            "TTFT p50 overhead",
            "Each cell shows the absolute change in ms and the percent change; lower is better",
            " ms",
            tuple(row.ttft_delta_ms if row is not None else None for row in ordered_rows),
            tuple(row.ttft_delta_pct if row is not None else None for row in ordered_rows),
            False,
        ),
        _Panel(
            "output_tokens_per_second_delta",
            "Output token throughput overhead",
            "Each cell shows the absolute change in tokens/s and the percent change; higher is better",
            " tokens/s",
            tuple(row.token_throughput_delta if row is not None else None for row in ordered_rows),
            tuple(
                row.token_throughput_delta_pct if row is not None else None for row in ordered_rows
            ),
            True,
        ),
    )

    label_width = 250
    cell_width = 170
    row_height = 50
    panel_header_height = 76
    panel_gap = 52
    top = 112
    right = 24
    panel_height = panel_header_height + len(workloads) * row_height
    width = label_width + len(routes) * cell_width + right
    height = top + len(panels) * panel_height + panel_gap + 28
    center = width / 2
    legend_x = center - 260

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="routing-overhead-title routing-overhead-description">'
        ),
        '<title id="routing-overhead-title">Routing overhead versus the direct backend</title>',
        (
            '<desc id="routing-overhead-description">Two annotated heatmaps compare median '
            "time to first token in milliseconds and output token throughput in tokens per "
            "second for each route and workload. Every cell also gives the percent change. Red "
            "cells are worse than the direct backend and blue cells are better. Darker cells "
            "show a larger percent change within each heatmap.</desc>"
        ),
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        _text(
            center,
            31,
            "Routing overhead versus the direct backend",
            anchor="middle",
            size=22,
            weight=700,
        ),
        _text(
            center,
            55,
            "Measures Switchyard overhead when both arms use the same deployment, model, and settings",
            anchor="middle",
            size=13,
            fill="#596777",
        ),
        f'<rect x="{legend_x}" y="72" width="18" height="14" rx="2" fill="#e6a9a9"/>',
        _text(legend_x + 24, 84, "worse", size=12, fill="#596777"),
        f'<rect x="{legend_x + 80}" y="72" width="18" height="14" rx="2" fill="#a8c7e2"/>',
        _text(legend_x + 104, 84, "better", size=12, fill="#596777"),
        f'<rect x="{legend_x + 163}" y="72" width="18" height="14" rx="2" fill="#f3f5f7"/>',
        _text(
            legend_x + 187,
            84,
            "no change / unavailable; darker = larger % change",
            size=12,
            fill="#596777",
        ),
    ]

    for panel_index, panel in enumerate(panels):
        panel_y = top + panel_index * (panel_height + panel_gap)
        svg.append(_text(center, panel_y + 18, panel.title, anchor="middle", size=16, weight=650))
        svg.append(
            _text(
                center,
                panel_y + 39,
                panel.subtitle,
                anchor="middle",
                size=12,
                fill="#596777",
            )
        )
        svg.append(
            _text(
                label_width / 2,
                panel_y + 66,
                "Workload / load",
                anchor="middle",
                size=12,
                weight=600,
            )
        )
        for route_index, route in enumerate(routes):
            cell_x = label_width + route_index * cell_width
            svg.append(
                _text(
                    cell_x + cell_width / 2,
                    panel_y + 66,
                    route.replace("-", " ").replace("_", " "),
                    anchor="middle",
                    size=12,
                    weight=600,
                )
            )

        finite_values = [abs(value) for value in panel.percent_values if _finite(value)]
        maximum = max(finite_values, default=1.0) or 1.0
        row_y = panel_y + panel_header_height
        for workload_index, (scenario, load) in enumerate(workloads):
            y = row_y + workload_index * row_height
            svg.append(
                _text(
                    label_width / 2,
                    y + 21,
                    scenario.replace("-", " "),
                    anchor="middle",
                    size=12,
                    fill="#344150",
                )
            )
            svg.append(
                _text(
                    label_width / 2,
                    y + 38,
                    load.replace("-", " "),
                    anchor="middle",
                    size=11,
                    fill="#697786",
                )
            )
            for route_index in range(len(routes)):
                value_index = workload_index * len(routes) + route_index
                primary_value = panel.primary_values[value_index]
                percent_value = panel.percent_values[value_index]
                x = label_width + route_index * cell_width
                cell_title = (
                    f"{scenario}, {load}, {routes[route_index]}: "
                    f"{_format_value(primary_value, panel.primary_suffix)} "
                    f"({_format_value(percent_value, '%')})"
                )
                svg.append(
                    f'<g data-scenario="{escape(scenario, quote=True)}" '
                    f'data-load="{escape(load, quote=True)}" '
                    f'data-route="{escape(routes[route_index], quote=True)}" '
                    f'data-metric="{panel.metric}">'
                )
                svg.append(f"<title>{escape(cell_title)}</title>")
                svg.append(
                    f'<rect x="{x + 2}" y="{y + 2}" width="{cell_width - 4}" '
                    f'height="{row_height - 4}" rx="3" '
                    f'fill="{_fill(percent_value, maximum, panel.higher_is_better)}" '
                    'stroke="#d9e0e7"/>'
                )
                svg.append(
                    _text(
                        x + cell_width / 2,
                        y + 21,
                        _format_value(primary_value, panel.primary_suffix),
                        anchor="middle",
                        size=12,
                        weight=600,
                    )
                )
                svg.append(
                    _text(
                        x + cell_width / 2,
                        y + 38,
                        f"({_format_value(percent_value, '%')})",
                        anchor="middle",
                        size=11,
                        fill="#44515f",
                    )
                )
                svg.append("</g>")

    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
