# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET

from scripts.routing_overhead_plot import OverheadPlotRow, write_overhead_plot


def test_overhead_plot_preserves_matrix_and_unavailable_values(tmp_path) -> None:
    path = tmp_path / "routing-overhead.svg"
    rows = (
        OverheadPlotRow("short-<interactive>", "fixed", "random", 3.0, 25.0, -80.0, -11.43),
        OverheadPlotRow("short-<interactive>", "fixed", "stage&router", None, None, -14.0, -2.0),
        OverheadPlotRow("long-context", "traffic-burst", "random", 1.25, 3.0, None, None),
        OverheadPlotRow("long-context", "traffic-burst", "stage&router", -0.5, -1.0, 7.0, 1.0),
    )

    write_overhead_plot(path, rows)

    root = ET.fromstring(path.read_text())
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    cells = root.findall(".//svg:g", namespace)
    text_nodes = root.findall(".//svg:text", namespace)
    values = {
        (
            cell.attrib["data-scenario"],
            cell.attrib["data-load"],
            cell.attrib["data-route"],
            cell.attrib["data-metric"],
        ): " ".join("".join(cell.itertext()).split())
        for cell in cells
    }
    assert len(values) == 8
    assert (
        values[
            (
                "short-<interactive>",
                "fixed",
                "random",
                "output_tokens_per_second_delta",
            )
        ]
        == "short-<interactive>, fixed, random: -80.00 tokens/s (-11.43%) "
        "-80.00 tokens/s (-11.43%)"
    )
    assert (
        "n/a"
        in values[("long-context", "traffic-burst", "random", "output_tokens_per_second_delta")]
    )
    assert "n/a" in values[("short-<interactive>", "fixed", "stage&router", "ttft_p50_delta_ms")]
    centered_labels = {
        node.text: node.attrib["text-anchor"]
        for node in text_nodes
        if node.text in {"Routing overhead versus the direct backend", "short <interactive>"}
    }
    assert centered_labels == {
        "Routing overhead versus the direct backend": "middle",
        "short <interactive>": "middle",
    }
    assert all(
        text.attrib["text-anchor"] == "middle"
        for cell in cells
        for text in cell.findall("svg:text", namespace)
    )
