# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an import graph by walking import statements in Python source files.

Handles both ``import X`` and ``from X import Y, Z`` forms, including aliases
and relative imports.
"""

from __future__ import annotations

from tree_sitter import Node

from .schemas import ImportEdge, ImportGraph
from .tree_sitter_extractor import TreeSitterExtractor, node_text, walk_nodes


def _dotted_name_text(node: Node, src: bytes) -> str:
    """Extract the module name from a dotted_name or aliased_import node."""
    if node.type == "dotted_name":
        return node_text(node, src).strip()
    if node.type == "aliased_import":
        name_child = node.child_by_field_name("name")
        if name_child is not None:
            return node_text(name_child, src).strip()
    if node.type == "identifier":
        return node_text(node, src).strip()
    return node_text(node, src).strip()


def _imported_name_text(node: Node, src: bytes) -> str:
    """Extract the public-facing name (alias preferred) from an import name node."""
    if node.type == "aliased_import":
        alias_node = node.child_by_field_name("alias")
        if alias_node is not None:
            return node_text(alias_node, src).strip()
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return node_text(name_node, src).strip()
    if node.type == "wildcard_import":
        return "*"
    return node_text(node, src).strip()


class ImportChainBuilder:
    """Build an ImportGraph from all Python files in a repository."""

    def __init__(self, extractor: TreeSitterExtractor) -> None:
        self._extractor = extractor

    def build(self) -> ImportGraph:
        """Walk all Python files and collect import edges."""
        edges: list[ImportEdge] = []

        for file_path in self._extractor.get_python_files():
            try:
                tree, src = self._extractor.parse_file(file_path)
            except OSError:
                continue

            source_module = self._extractor.get_module_name(file_path)
            edges.extend(self._extract_edges(tree.root_node, src, file_path, source_module))

        return ImportGraph(edges=edges)

    def _extract_edges(
        self,
        root: Node,
        src: bytes,
        file_path: str,
        source_module: str,
    ) -> list[ImportEdge]:
        edges: list[ImportEdge] = []

        for node in walk_nodes(root):
            if node.type == "import_statement":
                for name_node in node.children_by_field_name("name"):
                    target = _dotted_name_text(name_node, src)
                    if not target:
                        continue
                    edges.append(
                        ImportEdge(
                            source_module=source_module,
                            target_module=target,
                            imported_names=[],
                            file_path=file_path,
                            line=node.start_point[0] + 1,
                        )
                    )

            elif node.type == "import_from_statement":
                module_name_node = node.child_by_field_name("module_name")
                if module_name_node is None:
                    target_module = source_module.rsplit(".", 1)[0] if "." in source_module else source_module
                else:
                    target_module = node_text(module_name_node, src).strip()

                imported_names: list[str] = []
                for name_node in node.children_by_field_name("name"):
                    imported_names.append(_imported_name_text(name_node, src))

                for child in node.children:
                    if child.type == "wildcard_import" and "*" not in imported_names:
                        imported_names.append("*")

                if target_module:
                    edges.append(
                        ImportEdge(
                            source_module=source_module,
                            target_module=target_module,
                            imported_names=imported_names,
                            file_path=file_path,
                            line=node.start_point[0] + 1,
                        )
                    )

        return edges
