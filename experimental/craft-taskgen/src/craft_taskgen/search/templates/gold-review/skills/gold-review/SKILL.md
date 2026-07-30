---
name: gold-review
description: Review CRAFT Search gold answer quality for a specific task. Use when the user asks to review, check, investigate, or look closely at a craft-* task ID. Evaluates gold files, functions, assertions, and explanation against actual source code and agent tier results. Produces actionable recommendations (keep/remove/demote/promote/edit) with evidence.
---

# CRAFT Search Gold Data Review

Systematic evaluation of gold answer quality for CRAFT Search benchmark tasks.

## Trigger

User mentions a task ID like `craft-httpx-9e2db24f` or asks to review/investigate a specific task's gold data.

## Critical Principle

**Do NOT trust agent consensus.** The agents are the subjects being measured — using their majority vote to validate the gold answer is circular reasoning. Agent output is a secondary signal at best. The primary evidence is YOUR OWN deep reading of the source code. If 7/7 agents report a function but the code shows it's irrelevant to the question, it stays out. If 0/7 agents report a function but the code shows it's the core mechanism, it goes in.

## Workflow

### 1. Load task data

Run the loader script, which prints instruction, gold answer, tiers, and consensus counts:

```bash
uv run python .claude/skills/gold-review/scripts/load_task.py <task-id>
```

Read the output. Understand the question being asked — this anchors everything.

### 1.5. Trajectory Integrity Check

Run trajectory-based integrity checks against agent runs (if job dirs are available):

```bash
uv run python .claude/skills/gold-review/scripts/trajectory_integrity_check.py <task-id> \
    --job-dirs <opus-job-dir> <codex-job-dir>
```

This checks for four issues (from the agentic-benchmark-eval checklists):

| Check | What it detects | Action if FAIL |
|-------|----------------|----------------|
| **Gold Contamination** | Agent read `/tests/gold_answer.json` or `/solution/solve.sh` | Task is invalid — recommend reject |
| **Memorization** | High reward (>0.5) with <3 tool calls | Gold may be too easy or leaked — investigate |
| **Exploration Coverage** | Agent claims files in answer.json that it never actually read | Agent may have hallucinated files — check gold scope |
| **Answer Leakage** | Instruction text contains gold file paths or private function names | Instruction needs adversarial rewrite |

If no job dirs are available, skip this step — the checks require agent trajectory data.

### 2. Deep source code investigation (PRIMARY EVIDENCE)

This is the most important step. Read [references/investigation-rubrics.md](references/investigation-rubrics.md) for evidence standards, hallucination checks, and common gold answer mistakes before proceeding.

Launch a subagent (Explore type, "very thorough") to read the actual source code in `repos/{repo}/`. The subagent must:

1. **Answer the question yourself first.** Read the code as if you were an expert developer answering the instruction. Trace the execution path. Find the entry points, the branching logic, the key mechanisms. Form your own opinion of what files, functions, and behavioral insights constitute the correct answer.

2. **Read every gold function's source.** Use the auto-classifier script as a starting point, then go deeper:
   ```bash
   uv run python .claude/skills/gold-review/scripts/read_gold_functions.py <task-id>
   ```
   For each function, determine:
   - Does it contain substantive logic relevant to the question?
   - Or is it an abstract stub, one-liner delegation, trivial helper, registration-time code, generic plumbing, or API consumer?

3. **Trace the actual execution path.** Follow the call chain from entry point to resolution. Identify:
   - Where does behavior actually branch? (the key function)
   - What's the architectural insight? (mock connection, indexed lookup, registry dispatch, etc.)
   - Are there functions the gold missed that are central to the mechanism?

4. **Check assertions against the code.** For each assertion, verify it's factually correct by reading the relevant source. Look for:
   - Assertions that describe a mechanism differently than the code implements it
   - Assertions that are technically true but don't answer the question asked
   - Missing insights that the code reveals but the gold doesn't assert

### Classification table for gold functions

| Pattern | Action |
|---------|--------|
| Abstract method stub (`@abstractmethod` + docstring, no body) | Remove from gold |
| One-liner delegation (`return self.x.method()`) | Demote to alt or remove |
| Trivial helper (single `str.replace`, simple equality check) | Demote to alt |
| Registration/dedup helper (used at setup time, not at query time) | Remove if question is about runtime behavior |
| Substantive branching logic (if/else on mode, state, type) | Keep |
| Core algorithm (the loop, the dispatch, the main entry point) | Keep |
| Consumer/caller of the API (not part of the mechanism) | Demote to alt |
| Generic plumbing (socket write, base class everyone inherits) | Remove |

### 3. Agent consensus (SECONDARY SIGNAL ONLY)

After forming your own assessment from the code, check agent results for calibration:

```bash
uv run python .claude/skills/gold-review/scripts/load_task.py <task-id>
```

Use agent data to:
- **Sanity-check your analysis**: If you concluded a function is critical but 0/7 agents found it, re-examine — maybe it's not reachable from the question's entry point, or maybe agents are systematically blind to it (both are valid).
- **Spot items you might have missed**: If 5+ agents report a function not in gold/alt, investigate it — but only add it if the code justifies it.
- **NOT to override your code-based assessment**: "6/7 agents found X" is not a reason to keep X if the code shows it's irrelevant.

### 4. Evaluate assertions

For each assertion, verify against the actual source code and check for anti-patterns:

| Anti-pattern | Example | Action |
|-------------|---------|--------|
| **File-location trivia** | "X is handled in foo.py rather than bar.py" | Remove — that's what the files list is for |
| **Proves a negative** | "bar.py is NOT the code path used" | Remove — assertions should describe what IS, not what isn't |
| **Describes call chain, not behavior** | "X delegates to Y" | Remove unless the delegation IS the insight |
| **Misleading mechanism** | "normalizes in constructor" when code shows it happens at lookup time | Edit to match actual code |
| **Biases toward indirect path** | "append calls get_payload" when direct `append_json` is more relevant | Edit to cover both paths, or replace |
| **References specific file** | "tables.py has a setup() function" | Edit to remove file reference — assertions should be behavioral |
| **Factually incorrect** | Describes a mechanism that doesn't match what the code does | Edit or remove |
| **Correct and verifiable** | Describes actual behavior confirmed by reading the code | Keep |

Also check: do the assertions actually answer the question asked?

### 5. Evaluate explanation

Check against the code you read:
- Does it describe the actual mechanism or just name the layers?
- Does it mention the key branching point (the function where behavior actually diverges)?
- Does it miss the architectural insight you found in the code?
- Is it mechanistically accurate or does it handwave?
- Does it answer the specific question asked, or a broader/different question?

If the explanation is shallow or wrong, write a corrected version based on your code analysis.

### 6. Produce recommendations

Output a structured summary:

**Files:**
| Action | Item | Reason (cite code evidence) |

**Functions:**
| Action | Item | Reason (cite code evidence) |

**Assertions:**
| # | Action | Reason (cite code evidence) |

**Explanation:** Assessment + suggested rewrite if needed.

Actions: Keep, Remove, Demote to alt, Promote from alt, Add (not currently in gold/alt), Edit.

Every recommendation must cite **code evidence** (what the function actually does, what the assertion gets wrong). Agent consensus may be mentioned as secondary corroboration but never as the primary justification.

## Scripts

```bash
# Load task + print consensus
uv run python .claude/skills/gold-review/scripts/load_task.py craft-httpx-9e2db24f

# Read all gold functions from source with auto-classification
uv run python .claude/skills/gold-review/scripts/read_gold_functions.py craft-httpx-9e2db24f

# Apply annotations from UI export to task JSONs + JSONL audit log
uv run python .claude/skills/gold-review/scripts/apply_annotations.py tools/search/annotations.json --dry-run
uv run python .claude/skills/gold-review/scripts/apply_annotations.py tools/search/annotations.json
```

## Key reference files

- Task JSONs: `tasks/accepted/search/{repo}/*.json`
- Review data bundle: `tools/search/review_data.json`
- Tier config: `config/tiers.json`
- Repo source: `repos/{repo}/` (pinned commits in `repos/manifest.json`)
- Discrimination report: `validation/discrimination_report.json`
- Annotation patterns: see [references/annotation-schema.md](references/annotation-schema.md)
