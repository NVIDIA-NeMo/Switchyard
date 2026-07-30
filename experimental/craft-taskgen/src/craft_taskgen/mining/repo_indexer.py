# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate the full mining pipeline and persist results.

Usage::

    indexer = RepoIndexer("/path/to/repo")
    ground_truth = indexer.index()
    indexer.save("/path/to/output")
"""

from __future__ import annotations

import os

from .call_graph import CallGraphBuilder
from .import_chain import ImportChainBuilder
from .schemas import FileStructure, RepoGroundTruth
from .test_coverage_map import TestCoverageMapper
from .tree_sitter_extractor import PRUNE_DIRS, TreeSitterExtractor
from .type_defs import TypeDefExtractor


def load_ground_truth(path: str) -> RepoGroundTruth:
    """Load a RepoGroundTruth from a JSON file."""
    with open(path) as f:
        return RepoGroundTruth.model_validate_json(f.read())


class RepoIndexer:
    """Coordinate all extractors and produce a RepoGroundTruth for a repository."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.repo_name = os.path.basename(self.repo_path)
        self._extractor = TreeSitterExtractor(self.repo_path)

    def index(self) -> RepoGroundTruth:
        """Run all extractors and assemble the complete RepoGroundTruth."""
        print(f"[repo_indexer] Indexing {self.repo_path!r}")

        print("[repo_indexer] Building import graph ...")
        import_graph = ImportChainBuilder(self._extractor).build()
        print(f"[repo_indexer]   {len(import_graph.edges)} import edges")

        print("[repo_indexer] Building call graph ...")
        call_graph = CallGraphBuilder(self._extractor).build()
        print(f"[repo_indexer]   {len(call_graph.edges)} call edges")

        print("[repo_indexer] Extracting type definitions ...")
        type_defs = TypeDefExtractor(self._extractor).build()
        print(f"[repo_indexer]   {len(type_defs.classes)} classes, {len(type_defs.aliases)} type aliases")

        print("[repo_indexer] Building test coverage map ...")
        test_coverage = TestCoverageMapper(self._extractor, call_graph).build()
        print(f"[repo_indexer]   {len(test_coverage.mappings)} test mappings")

        print("[repo_indexer] Collecting file structure ...")
        file_structure = self._collect_file_structure()

        return RepoGroundTruth(
            repo_name=self.repo_name,
            repo_path=self.repo_path,
            call_graph=call_graph,
            import_graph=import_graph,
            type_defs=type_defs,
            test_coverage=test_coverage,
            file_structure=file_structure,
        )

    def save(self, output_dir: str) -> None:
        """Index this repo and write all ground truth data to *output_dir*."""
        ground_truth = self.index()
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, self.repo_name)

        full_path = f"{base}.ground_truth.json"
        with open(full_path, "w", encoding="utf-8") as fh:
            fh.write(ground_truth.model_dump_json(indent=2))
        print(f"[repo_indexer] Wrote {full_path}")

        components: dict[str, object] = {
            "call_graph": ground_truth.call_graph,
            "import_graph": ground_truth.import_graph,
            "type_defs": ground_truth.type_defs,
            "test_coverage": ground_truth.test_coverage,
            "file_structure": ground_truth.file_structure,
        }
        for name, model in components.items():
            path = f"{base}.{name}.json"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(model.model_dump_json(indent=2))  # type: ignore[union-attr]
            print(f"[repo_indexer]   {path}")

    def _collect_file_structure(self) -> FileStructure:
        """Walk the repository and collect relative paths of all files and dirs."""
        all_files: list[str] = []
        all_dirs: list[str] = []

        for dirpath, dirnames, filenames in os.walk(self.repo_path):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in PRUNE_DIRS)
            rel_dir = os.path.relpath(dirpath, self.repo_path)
            if rel_dir != ".":
                all_dirs.append(rel_dir)
            for filename in sorted(filenames):
                all_files.append(os.path.relpath(os.path.join(dirpath, filename), self.repo_path))

        return FileStructure(
            root=self.repo_path,
            files=sorted(all_files),
            directories=sorted(all_dirs),
        )
