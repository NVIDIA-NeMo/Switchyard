# Annotation Schema

Annotations are exported from the gold-reviewer UI as JSON. Each key is a task ID.

```json
{
  "craft-repo-uuid8": {
    "status": "keep" | "reject" | "uncertain" | null,
    "notes": "Free-text rationale",
    "removed_files": ["path/to/file.py"],
    "removed_functions": ["module.Class.method"],
    "removed_assertions": [0, 2],
    "demoted_files": ["path/to/file.py"],
    "demoted_functions": ["module.Class.method"],
    "promoted_alt_files": ["path/to/file.py"],
    "promoted_alt_functions": ["module.Class.method"],
    "edited_assertions": {"2": "New assertion text"},
    "edited_explanation": "New explanation text",
    "updated_at": "ISO8601"
  }
}
```

## Semantics

- **removed**: Delete from gold entirely (wrong, irrelevant)
- **demoted**: Move from primary gold to alt (correct but not essential to the answer)
- **promoted**: Move from alt to primary gold (agents consistently find it, should count for scoring)
- **edited**: Replace text (assertion was misleading or explanation was shallow)

## Applying annotations to task JSONs

For each annotated task:
1. Remove `removed_files` from `gold_answer.files`, add to `gold_answer.alt_files`
2. Remove `removed_functions` from `gold_answer.functions`, add to `gold_answer.alt_functions`
3. Move `demoted_files` from `gold_answer.files` to `gold_answer.alt_files`
4. Move `demoted_functions` from `gold_answer.functions` to `gold_answer.alt_functions`
5. Move `promoted_alt_files` from `gold_answer.alt_files` to `gold_answer.files`
6. Move `promoted_alt_functions` from `gold_answer.alt_functions` to `gold_answer.functions`
7. Delete assertions at `removed_assertions` indices (process highest index first)
8. Replace assertion text at `edited_assertions` indices
9. Replace `gold_answer.explanation` with `edited_explanation` if present

After applying, rescore all tiers:
```bash
uv run python scripts/search/rescore.py \
    --tier-mapping validation/tier_job_dirs.json \
    --harbor-tasks harbor-tasks/craft-search/
uv run python scripts/search/analyze_discrimination.py \
    --tier-mapping validation/tier_job_dirs.json
```
