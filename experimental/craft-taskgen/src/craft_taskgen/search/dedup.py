# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deduplicate search-from-T2 tasks by embedding-based instruction similarity.

Loads tasks from all approaches, embeds instructions, removes near-duplicates
(cosine >= threshold), and merges removed tasks' gold into kept tasks' alt fields.
"""

from __future__ import annotations

import json
import math
import os


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _gold_richness(task: dict) -> tuple[int, int]:
    gold = task.get("gold_answer", {})
    n_elements = len(gold.get("files", [])) + len(gold.get("functions", []))
    n_assertions = len(gold.get("assertions", []))
    return (n_elements, n_assertions)


def _merge_alt_gold(kept: dict, removed: dict) -> None:
    """Merge removed task's gold files/functions into kept's alt fields. Mutates kept."""
    kept_gold = kept.get("gold_answer", {})
    removed_gold = removed.get("gold_answer", {})

    existing_files = set(kept_gold.get("files", [])) | set(kept_gold.get("alt_files", []))
    novel_files = [f for f in removed_gold.get("files", []) if f not in existing_files]

    existing_funcs = set(kept_gold.get("functions", [])) | set(kept_gold.get("alt_functions", []))
    novel_funcs = [f for f in removed_gold.get("functions", []) if f not in existing_funcs]

    if novel_files:
        kept_gold.setdefault("alt_files", []).extend(novel_files)
    if novel_funcs:
        kept_gold.setdefault("alt_functions", []).extend(novel_funcs)


def embed_instructions(instructions: list[str], model: str) -> list[list[float]]:
    """Batch-embed instructions via litellm."""
    import litellm

    response = litellm.embedding(model=model, input=instructions)
    return [item["embedding"] for item in response.data]


def deduplicate(
    tasks: list[dict],
    embeddings: list[list[float]],
    threshold: float = 0.65,
) -> tuple[list[dict], list[dict]]:
    """Remove near-duplicate tasks by cosine similarity on instruction embeddings.

    Keeps the task with richer gold (more files+functions, then more assertions).
    Merges removed task's gold into kept task's alt_files/alt_functions.

    Returns (kept_tasks, dedup_pairs).
    """
    n = len(tasks)
    removed: set[int] = set()
    dedup_pairs: list[dict] = []
    richness = [_gold_richness(t) for t in tasks]

    for i in range(n):
        if i in removed:
            continue
        for j in range(i + 1, n):
            if j in removed:
                continue
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                if richness[j] > richness[i]:
                    kept_idx, removed_idx = j, i
                else:
                    kept_idx, removed_idx = i, j

                _merge_alt_gold(tasks[kept_idx], tasks[removed_idx])

                dedup_pairs.append(
                    {
                        "cosine": round(sim, 4),
                        "kept": tasks[kept_idx]["id"],
                        "removed": tasks[removed_idx]["id"],
                        "kept_richness": richness[kept_idx],
                        "removed_richness": richness[removed_idx],
                    }
                )
                removed.add(removed_idx)
                if removed_idx == i:
                    break

    kept = [t for idx, t in enumerate(tasks) if idx not in removed]
    return kept, sorted(dedup_pairs, key=lambda p: -p["cosine"])


def run_dedup(
    output_dir: str,
    *,
    threshold: float = 0.65,
    embedding_model: str = "openai/azure/openai/text-embedding-3-small",
) -> None:
    """Load tasks from all approaches, embed, deduplicate, and write back.

    Args:
        output_dir: Directory containing approach-{a,b,c}/search_tasks.json.
        threshold: Cosine similarity threshold for duplicate detection.
        embedding_model: LiteLLM model identifier for embedding.
    """
    # Load all tasks across approaches, tracking which approach each came from
    all_tasks: list[dict] = []
    approach_map: dict[str, str] = {}  # task_id -> approach letter
    for approach in ["a", "b", "c"]:
        path = os.path.join(output_dir, f"approach-{approach}", "search_tasks.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            tasks = json.load(f)
        for t in tasks:
            approach_map[t["id"]] = approach
        all_tasks.extend(tasks)
        print(f"  Approach {approach.upper()}: {len(tasks)} tasks")

    print(f"  Total: {len(all_tasks)} tasks")

    if not all_tasks:
        print("No tasks found.")
        return

    # Embed
    print(f"\nEmbedding {len(all_tasks)} instructions (model: {embedding_model})...")
    instructions = [t.get("instruction", "") for t in all_tasks]
    embeddings = embed_instructions(instructions, embedding_model)
    print(f"  Got {len(embeddings)} embeddings ({len(embeddings[0])} dims)")

    # Dedup
    print(f"\nDeduplicating (threshold: {threshold})...")
    kept, pairs = deduplicate(all_tasks, embeddings, threshold=threshold)
    n_removed = len(all_tasks) - len(kept)

    print(f"  Removed: {n_removed} duplicates ({len(pairs)} pairs)")
    print(f"  Kept: {len(kept)} tasks")

    # Show top pairs
    if pairs:
        print(f"\n  Top duplicate pairs (cosine >= {threshold}):")
        for p in pairs[:15]:
            print(f"    {p['cosine']:.3f}  kept={p['kept']}  removed={p['removed']}")
        if len(pairs) > 15:
            print(f"    ... and {len(pairs) - 15} more")

    # Per-approach breakdown
    for approach in ["a", "b", "c"]:
        before = sum(1 for _tid, a in approach_map.items() if a == approach)
        after = sum(1 for t in kept if approach_map.get(t["id"]) == approach)
        print(f"  Approach {approach.upper()}: {before} -> {after} (-{before - after})")

    # Write back to approach files
    for approach in ["a", "b", "c"]:
        path = os.path.join(output_dir, f"approach-{approach}", "search_tasks.json")
        if not os.path.exists(path):
            continue
        approach_kept = [t for t in kept if approach_map.get(t["id"]) == approach]
        with open(path, "w") as f:
            json.dump(approach_kept, f, indent=2)
        print(f"  Updated {path} ({len(approach_kept)} tasks)")

    # Write manifest
    manifest = {
        "threshold": threshold,
        "embedding_model": embedding_model,
        "total_before": len(all_tasks),
        "total_after": len(kept),
        "duplicates_removed": n_removed,
        "pairs": pairs,
    }
    manifest_path = os.path.join(output_dir, "dedup_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")
