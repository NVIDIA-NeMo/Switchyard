# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a call graph by walking function bodies and resolving call targets.

For each function definition found in the repository, we find all ``call``
nodes within its body and record a CallEdge from the enclosing function's
qualified name to a best-effort qualified callee name.
"""

from __future__ import annotations

from tree_sitter import Node

from .schemas import CallEdge, CallGraph
from .tree_sitter_extractor import TreeSitterExtractor, node_text, walk_nodes


def _resolve_callee(func_node: Node | None, src: bytes) -> str:
    """Return a best-effort string name for the callee of a call expression."""
    if func_node is None:
        return "<unknown>"
    if func_node.type == "identifier":
        return node_text(func_node, src)
    if func_node.type == "attribute":
        return node_text(func_node, src)
    return node_text(func_node, src)


def _qualified_name_for_function(func_node: Node, src: bytes, module: str) -> str:
    """Build a qualified caller name.

    Uses ``module.Class.method`` for methods and ``module.func`` for
    module-level functions.
    """
    name_node = func_node.child_by_field_name("name")
    func_name = node_text(name_node, src) if name_node else "<anonymous>"

    classes: list[str] = []
    current = func_node.parent
    while current is not None:
        if current.type == "class_definition":
            class_name_node = current.child_by_field_name("name")
            classes.append(node_text(class_name_node, src) if class_name_node else "<anonymous_class>")
        current = current.parent
    classes.reverse()

    if classes:
        return ".".join([module, *classes, func_name])
    return f"{module}.{func_name}"


def _enclosing_function(node: Node) -> Node | None:
    """Return the nearest ancestor function_definition node, or None."""
    current = node.parent
    while current is not None:
        if current.type in {"function_definition", "async_function_definition"}:
            return current
        current = current.parent
    return None


class CallGraphBuilder:
    """Build a CallGraph from all Python files in a repository."""

    def __init__(self, extractor: TreeSitterExtractor) -> None:
        self._extractor = extractor

    def build(self) -> CallGraph:
        """Walk all Python files and collect call edges."""
        edges: list[CallEdge] = []

        for file_path in self._extractor.get_python_files():
            try:
                tree, src = self._extractor.parse_file(file_path)
            except OSError:
                continue

            module = self._extractor.get_module_name(file_path)
            edges.extend(self._extract_edges(tree.root_node, src, file_path, module))

        return CallGraph(edges=edges)

    def _extract_edges(
        self,
        root: Node,
        src: bytes,
        file_path: str,
        module: str,
    ) -> list[CallEdge]:
        edges: list[CallEdge] = []

        for node in walk_nodes(root):
            if node.type != "call":
                continue

            func_node = node.child_by_field_name("function")
            callee = _resolve_callee(func_node, src)

            enclosing = _enclosing_function(node)
            if enclosing is None:
                caller = module
            else:
                caller = _qualified_name_for_function(enclosing, src, module)

            edges.append(
                CallEdge(
                    caller=caller,
                    callee=callee,
                    file_path=file_path,
                    line=node.start_point[0] + 1,
                )
            )

        return edges
