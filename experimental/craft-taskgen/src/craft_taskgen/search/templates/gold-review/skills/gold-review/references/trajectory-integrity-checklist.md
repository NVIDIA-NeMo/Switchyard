# Trajectory Integrity Checklist

Based on the agentic-benchmark-eval checklists (verifier_audit.md, task_design.md).

## Check 1: Gold Contamination

**What:** Agent reads verifier files during its run.

**Contamination paths:**
- `/tests/gold_answer.json` — the gold answer the verifier scores against
- `/solution/solve.sh` — the oracle solution script
- `/tests/test_runner.py` — the verifier implementation
- `/tests/test.sh` — the verifier entry point

**How it's detected:** Parse all tool_calls in the trajectory. Check if any Read/Glob/Bash operation accesses these paths.

**Threshold:** Any access = FAIL. Zero tolerance.

**Why it matters:** From verifier_audit.md: "the verifier trusts the environment the agent controls." If the agent can read the gold answer, it can trivially achieve a perfect score.

## Check 2: Memorization / No-Search

**What:** Agent scores high without actually exploring the codebase.

**Signals:**
- Reward > 0.5 with fewer than 3 tool calls
- Agent writes answer.json without calling Read, Grep, or Glob
- Very short trajectory (1-2 steps)

**Threshold:** reward > 0.5 AND tool_calls < 3 = FAIL.

**Why it matters:** From task_design.md "Anti-Guessing Design": "Do tasks contain details unique to the source data (not in training data)? Would a knowledgeable domain expert (who hasn't read the documents) score <15%?"

If an agent scores well from training data alone, the task doesn't measure code exploration ability.

## Check 3: Exploration Coverage

**What:** Agent claims files in its answer that it never actually read.

**How it's detected:** Compare `agent_files` from reward.json (what the agent wrote in answer.json) against files accessed via Read/Glob tool calls in the trajectory.

**Threshold:** More than 50% of claimed files unverified = WARN.

**Why it matters:** From verifier_audit.md: "The verifier should reward answers that come from the right source for the right reason, not just answers that happen to be correct."

An agent might guess file names from package structure without reading them. This isn't exploration — it's pattern matching.

## Check 4: Answer Leakage

**What:** The instruction text contains gold file paths or private function names, making the task trivially solvable.

**Signals:**
- Gold file path appears verbatim in instruction
- Private function name (`_internal_method`) appears in instruction
- Module path (e.g., `uvicorn.config`) appears in instruction and matches gold

**Threshold:** Any private function or full file path in instruction = FAIL.

**Why it matters:** From task_design.md: "No solution leakage in instructions." The adversarial synthesis prompt already tries to prevent this, but the check validates it.

## Correlation Monitoring

From verifier_audit.md "Reward Fidelity Monitoring":

> Track correlation between exploration depth and score across many runs; if uncorrelated, verifier may not be catching guessers.

Over time, monitor whether agents with more tool calls score higher. If there's no correlation, the scoring may be rewarding guessing over genuine exploration.
