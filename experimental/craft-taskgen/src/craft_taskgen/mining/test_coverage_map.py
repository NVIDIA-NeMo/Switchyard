# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map test functions to the application functions they exercise.

Strategy:
1. Identify test files (``test_*.py`` or ``*_test.py``).
2. Find test functions within those files (any function named ``test_*``).
3. Use the call graph to BFS-expand from each test function, collecting all
   transitively reachable callees that correspond to non-test source functions.
"""

from __future__ import annotations

import os
from collections import defaultdict, deque

from .schemas import CallGraph, TestCoverageMap, TestMapping
from .tree_sitter_extractor import TreeSitterExtractor, node_text, walk_nodes


def _is_test_file(file_path: str) -> bool:
    """Return True if *file_path* looks like a test file by naming convention."""
    basename = os.path.basename(file_path)
    return basename.startswith("test_") or basename.endswith("_test.py")


class TestCoverageMapper:
    """Build a TestCoverageMap using the call graph for reachability analysis."""

    def __init__(self, extractor: TreeSitterExtractor, call_graph: CallGraph) -> None:
        self._extractor = extractor
        self._call_graph = call_graph

    def build(self) -> TestCoverageMap:
        """Find all test functions and map them to the functions they cover."""
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in self._call_graph.edges:
            adjacency[edge.caller].add(edge.callee)

        app_functions: set[str] = set()
        test_files: list[str] = []

        for file_path in self._extractor.get_python_files():
            if _is_test_file(file_path):
                test_files.append(file_path)
            else:
                module = self._extractor.get_module_name(file_path)
                try:
                    for fn in self._extractor.extract_functions(file_path):
                        app_functions.add(f"{module}.{fn.name}")
                except OSError:
                    continue

        mappings: list[TestMapping] = []

        for test_file in test_files:
            try:
                tree, src = self._extractor.parse_file(test_file)
            except OSError:
                continue

            test_module = self._extractor.get_module_name(test_file)

            for node in walk_nodes(tree.root_node):
                if node.type != "function_definition":
                    continue
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    continue
                func_name = node_text(name_node, src)
                if not func_name.startswith("test_"):
                    continue

                test_qualified = f"{test_module}.{func_name}"
                tested = self._reachable_app_functions(test_qualified, adjacency, app_functions)

                mappings.append(
                    TestMapping(
                        test_file=test_file,
                        test_function=test_qualified,
                        tested_functions=sorted(tested),
                    )
                )

        return TestCoverageMap(mappings=mappings)

    def _reachable_app_functions(
        self,
        start: str,
        adjacency: dict[str, set[str]],
        app_functions: set[str],
    ) -> set[str]:
        """BFS from *start* through the call graph; return reachable app functions."""
        visited: set[str] = set()
        queue: deque[str] = deque([start])
        reachable: set[str] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for callee in adjacency.get(current, set()):
                if callee in app_functions:
                    reachable.add(callee)
                if callee not in visited:
                    queue.append(callee)

        return reachable
