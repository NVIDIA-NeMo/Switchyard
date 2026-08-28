# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic v2 ground truth models for the repo mining pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FunctionDef(BaseModel):
    """A function or method definition extracted from source."""

    name: str
    file_path: str
    line_start: int
    line_end: int
    docstring: str | None = None
    decorators: list[str] = Field(default_factory=list)


class ClassDef(BaseModel):
    """A class definition extracted from source."""

    name: str
    file_path: str
    line_start: int
    line_end: int
    bases: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)


class TypeAlias(BaseModel):
    """A type alias definition (TypeAlias annotation or PEP 695 type statement)."""

    name: str
    file_path: str
    target_type: str


class CallEdge(BaseModel):
    """A directed call edge from one qualified name to another."""

    caller: str
    callee: str
    file_path: str
    line: int


class ImportEdge(BaseModel):
    """A directed import edge between modules."""

    source_module: str
    target_module: str
    imported_names: list[str] = Field(default_factory=list)
    file_path: str
    line: int


class TestMapping(BaseModel):
    """Maps a test function to the set of application functions it exercises."""

    test_file: str
    test_function: str
    tested_functions: list[str] = Field(default_factory=list)


class CallGraph(BaseModel):
    """Complete call graph for a repository."""

    edges: list[CallEdge] = Field(default_factory=list)


class ImportGraph(BaseModel):
    """Complete import graph for a repository."""

    edges: list[ImportEdge] = Field(default_factory=list)


class TypeDefinitions(BaseModel):
    """All class and type alias definitions in a repository."""

    classes: list[ClassDef] = Field(default_factory=list)
    aliases: list[TypeAlias] = Field(default_factory=list)


class TestCoverageMap(BaseModel):
    """Maps test functions to the application functions they cover."""

    mappings: list[TestMapping] = Field(default_factory=list)


class FileStructure(BaseModel):
    """Top-level file and directory layout of a repository."""

    root: str
    files: list[str] = Field(default_factory=list)
    directories: list[str] = Field(default_factory=list)


class RepoGroundTruth(BaseModel):
    """Complete ground truth facts extracted from a single repository."""

    repo_name: str
    repo_path: str
    call_graph: CallGraph = Field(default_factory=CallGraph)
    import_graph: ImportGraph = Field(default_factory=ImportGraph)
    type_defs: TypeDefinitions = Field(default_factory=TypeDefinitions)
    test_coverage: TestCoverageMap = Field(default_factory=TestCoverageMap)
    file_structure: FileStructure
