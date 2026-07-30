# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core AST parsing using tree-sitter.

Provides TreeSitterExtractor which parses Python files and extracts
function/class definitions with full metadata (line numbers, docstrings,
decorators, base classes).
"""

from __future__ import annotations

import os
from typing import Generator

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Tree

from .schemas import ClassDef, FunctionDef

PY_LANGUAGE = Language(tspython.language())

PRUNE_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build", ".tox"}


def walk_nodes(node: Node) -> Generator[Node, None, None]:
    """Yield all nodes in the subtree rooted at *node* (depth-first pre-order)."""
    yield node
    for child in node.children:
        yield from walk_nodes(child)


def node_text(node: Node, src: bytes) -> str:
    """Return the source text for *node* as a UTF-8 string."""
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_docstring(body_node: Node, src: bytes) -> str | None:
    """Return the docstring from a function/class body block, or None."""
    if body_node is None:
        return None
    for child in body_node.named_children:
        if child.type == "expression_statement":
            for sub in child.named_children:
                if sub.type == "string":
                    raw = node_text(sub, src)
                    for delim in ('"""', "'''", '"', "'"):
                        if raw.startswith(delim) and raw.endswith(delim) and len(raw) >= 2 * len(delim):
                            return raw[len(delim) : -len(delim)].strip()
                    return raw.strip()
        break
    return None


def _extract_decorators(decorated_node: Node, src: bytes) -> list[str]:
    """Extract decorator text strings from a decorated_definition node."""
    decorators: list[str] = []
    for child in decorated_node.children:
        if child.type == "decorator":
            text = node_text(child, src).strip()
            if text.startswith("@"):
                text = text[1:]
            decorators.append(text.strip())
    return decorators


class TreeSitterExtractor:
    """Extract structural facts from Python source files using tree-sitter."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self._parser = Parser(PY_LANGUAGE)

    def get_python_files(self) -> list[str]:
        """Return absolute paths of all .py files under *repo_path*."""
        results: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.repo_path):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in PRUNE_DIRS]
            for filename in sorted(filenames):
                if filename.endswith(".py"):
                    results.append(os.path.join(dirpath, filename))
        return sorted(results)

    def parse_file(self, file_path: str) -> tuple[Tree, bytes]:
        """Parse *file_path* and return the (Tree, source_bytes) pair."""
        with open(file_path, "rb") as fh:
            src = fh.read()
        tree = self._parser.parse(src)
        return tree, src

    def get_module_name(self, file_path: str) -> str:
        """Convert an absolute file path to a dotted module name relative to repo root."""
        rel = os.path.relpath(file_path, self.repo_path)
        if rel.endswith(".py"):
            rel = rel[:-3]
        if rel.endswith("/__init__") or rel.endswith(os.sep + "__init__"):
            rel = rel[: -(len("/__init__"))]
        return rel.replace(os.sep, ".").replace("/", ".")

    def extract_functions(self, file_path: str) -> list[FunctionDef]:
        """Return all FunctionDef instances found in *file_path*."""
        tree, src = self.parse_file(file_path)
        results: list[FunctionDef] = []

        for node in walk_nodes(tree.root_node):
            if node.type == "decorated_definition":
                inner = node.child_by_field_name("definition")
                if inner is not None and inner.type == "function_definition":
                    decorators = _extract_decorators(node, src)
                    results.append(self._build_function_def(inner, src, file_path, decorators))
            elif node.type == "function_definition":
                parent = node.parent
                if parent is not None and parent.type == "decorated_definition":
                    continue
                results.append(self._build_function_def(node, src, file_path, []))

        return results

    def _build_function_def(
        self,
        node: Node,
        src: bytes,
        file_path: str,
        decorators: list[str],
    ) -> FunctionDef:
        name_node = node.child_by_field_name("name")
        name = node_text(name_node, src) if name_node else "<anonymous>"
        body_node = node.child_by_field_name("body")
        docstring = _extract_docstring(body_node, src) if body_node else None
        return FunctionDef(
            name=name,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            docstring=docstring,
            decorators=decorators,
        )

    def extract_classes(self, file_path: str) -> list[ClassDef]:
        """Return all ClassDef instances found in *file_path*."""
        tree, src = self.parse_file(file_path)
        results: list[ClassDef] = []

        for node in walk_nodes(tree.root_node):
            if node.type == "decorated_definition":
                inner = node.child_by_field_name("definition")
                if inner is not None and inner.type == "class_definition":
                    results.append(self._build_class_def(inner, src, file_path))
            elif node.type == "class_definition":
                parent = node.parent
                if parent is not None and parent.type == "decorated_definition":
                    continue
                results.append(self._build_class_def(node, src, file_path))

        return results

    def _build_class_def(self, node: Node, src: bytes, file_path: str) -> ClassDef:
        name_node = node.child_by_field_name("name")
        name = node_text(name_node, src) if name_node else "<anonymous>"

        bases: list[str] = []
        superclasses_node = node.child_by_field_name("superclasses")
        if superclasses_node is not None:
            for arg in superclasses_node.named_children:
                if arg.type not in {"comment", ","}:
                    text = node_text(arg, src).strip()
                    if text:
                        bases.append(text)

        methods: list[str] = []
        body_node = node.child_by_field_name("body")
        if body_node is not None:
            for child in body_node.named_children:
                if child.type == "function_definition":
                    method_name_node = child.child_by_field_name("name")
                    if method_name_node:
                        methods.append(node_text(method_name_node, src))
                elif child.type == "decorated_definition":
                    inner = child.child_by_field_name("definition")
                    if inner is not None and inner.type == "function_definition":
                        method_name_node = inner.child_by_field_name("name")
                        if method_name_node:
                            methods.append(node_text(method_name_node, src))

        return ClassDef(
            name=name,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            bases=bases,
            methods=methods,
        )
