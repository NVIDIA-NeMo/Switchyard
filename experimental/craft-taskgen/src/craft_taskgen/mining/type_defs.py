# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract type definitions from Python source files.

Handles:
- Class definitions (delegated to TreeSitterExtractor)
- PEP 695 type alias statements: ``type Vector = list[float]``
- Old-style TypeAlias annotations: ``Vector: TypeAlias = list[float]``
"""

from __future__ import annotations

from tree_sitter import Node

from .schemas import ClassDef, TypeAlias, TypeDefinitions
from .tree_sitter_extractor import TreeSitterExtractor, node_text, walk_nodes


def _looks_like_type_alias_annotation(node: Node, src: bytes) -> bool:
    """Return True if *node* is an annotated assignment whose annotation is TypeAlias."""
    if node.type != "annotated_assignment":
        return False
    for child in node.children:
        if child.type in {"type", "subscript", "identifier", "attribute"}:
            text = node_text(child, src)
            if "TypeAlias" in text:
                return True
    return False


class TypeDefExtractor:
    """Extract class definitions and type aliases from all Python files in a repo."""

    def __init__(self, extractor: TreeSitterExtractor) -> None:
        self._extractor = extractor

    def build(self) -> TypeDefinitions:
        """Walk all Python files and collect type definitions."""
        classes: list[ClassDef] = []
        aliases: list[TypeAlias] = []

        for file_path in self._extractor.get_python_files():
            try:
                classes.extend(self._extractor.extract_classes(file_path))
                aliases.extend(self._extract_aliases(file_path))
            except OSError:
                continue

        return TypeDefinitions(classes=classes, aliases=aliases)

    def _extract_aliases(self, file_path: str) -> list[TypeAlias]:
        """Find all type alias definitions in *file_path*."""
        tree, src = self._extractor.parse_file(file_path)
        aliases: list[TypeAlias] = []

        for node in walk_nodes(tree.root_node):
            if node.type == "type_alias_statement":
                left_node = node.child_by_field_name("left")
                right_node = node.child_by_field_name("right")
                if left_node is None:
                    continue
                name = node_text(left_node, src).strip()
                target = node_text(right_node, src).strip() if right_node else "<unknown>"
                aliases.append(TypeAlias(name=name, file_path=file_path, target_type=target))

            elif node.type == "annotated_assignment":
                if not _looks_like_type_alias_annotation(node, src):
                    continue
                name_node = None
                for child in node.named_children:
                    if child.type == "identifier":
                        name_node = child
                        break
                if name_node is None:
                    continue
                name = node_text(name_node, src).strip()
                value_node = node.child_by_field_name("value")
                target = node_text(value_node, src).strip() if value_node else "<unknown>"
                aliases.append(TypeAlias(name=name, file_path=file_path, target_type=target))

        return aliases
