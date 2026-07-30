# Investigation Rubrics & Hallucination Checks

Read this before starting step 2 (Deep source code investigation).

## Subagent Investigation Requirements

The subagent MUST provide **verifiable evidence** for every claim. Summaries without line numbers, paraphrases of code without quotes, and assertions about what code "probably" does are not acceptable.

### Evidence standards

For each gold function reviewed, the subagent must provide:

1. **Exact file path and line numbers** where the function is defined
2. **Quoted code** — the actual function body (or key excerpt if long)
3. **Classification with justification** — not just "SUBSTANTIVE" but why (e.g., "contains if/else branching on `self.as_sql` at line 220 that selects offline vs online path")
4. **Call context** — who calls this function and when? Is it on the hot path for the question being asked, or is it a setup/registration/teardown function?

### What counts as "answering the question"

A function belongs in gold if and only if:
- It is on the **execution path** triggered by the scenario in the instruction
- It contains **logic that directly addresses** what the question asks about
- Understanding it is **necessary** to answer the question correctly

A function does NOT belong in gold merely because:
- It exists in the same file
- It is called somewhere in the call chain (thin wrappers don't count)
- It defines an interface that the real implementation fulfills
- It's the "proper" entry point but just delegates immediately

## Hallucination Checks

### Before finalizing each function assessment

- [ ] Did you actually READ the function body, or are you inferring from the name?
- [ ] Can you quote the key line(s) that justify your classification?
- [ ] If you say a function "branches on X" — did you see the `if` statement?
- [ ] If you say a function "delegates to Y" — did you see the call?
- [ ] If you say a function is "not called during resolution" — did you grep for call sites?

### Before finalizing assertion assessments

- [ ] For each assertion you mark "correct" — can you point to the specific code that implements it?
- [ ] For each assertion you mark "incorrect" — can you quote the code that contradicts it?
- [ ] Did you check whether the assertion describes behavior at the RIGHT level? (e.g., "iterates in order" might be correct for a simple loop but wrong if there's an indexed lookup)

### Before recommending new functions to add

- [ ] Did you verify the function EXISTS in the pinned commit? (not added in a later version)
- [ ] Is it genuinely on the execution path, or did you infer it from naming patterns?
- [ ] Would an expert answering this question naturally need to read this function?

## Common Mistakes in Gold Answers

These patterns appear frequently in the existing gold data. Watch for them:

### 1. Full call chain enumeration
The gold lists every function in the call chain from entry point to leaf, including trivial wrappers. Fix: keep only functions with substantive logic.

Example: `EnvContext.execute → MigrationContext.execute → DefaultImpl.execute → DefaultImpl._exec`. Only `_exec` has the branching logic — the other three are one-liner delegations.

### 2. Abstract + concrete duplication
The gold lists both the abstract method in `abc.py` AND the concrete implementation. The abstract method is never the answer — it's a contract definition with no logic.

### 3. Registration-time vs runtime confusion
Functions used to register/configure things at startup are listed as gold for questions about runtime behavior. Check WHEN the function runs relative to the question's scenario.

Example: `raw_match` is used during route registration to dedup resources, but the question asks about route resolution. Different execution phase.

### 4. Overly broad file inclusion
The gold includes files that contain one tangentially relevant function alongside hundreds of irrelevant ones (e.g., `helpers.py`, `utils.py`, `abc.py`). If only one small function in a large file matters, and that function is trivial, the file probably doesn't belong.

### 5. Generated-by-model artifacts
Gold answers synthesized by LLMs sometimes include:
- Functions the model "expected" to exist based on naming conventions but that don't
- Assertions about behavior the model inferred from docstrings rather than reading the implementation
- Explanations that describe the "standard pattern" for this type of library rather than what this specific codebase does

Always verify against the actual code at the pinned commit.

### 6. Question-answer mismatch
The gold answers "how does X work in general" when the question asked something narrower like "why does X behave differently when Y." The gold should answer the ACTUAL question, not a broader restatement.

## Scoring Rubric for Recommendations

Use this to calibrate confidence in your recommendations:

**High confidence (assert in recommendation):**
- Function body is 1-2 lines with no branching → classify as trivial/delegation
- Function has `@abstractmethod` decorator → classify as abstract stub
- Assertion contradicts code you quoted → mark as incorrect
- Function is called only during setup, question asks about runtime → wrong phase

**Medium confidence (recommend but flag for human review):**
- Function has moderate logic but it's unclear if it's relevant to the specific question
- Assertion is technically correct but might be testing the wrong level of detail
- Explanation describes the right mechanism but at the wrong granularity

**Low confidence (note but don't recommend action):**
- You're unsure whether a function is "substantive enough" for gold
- The question is ambiguous about what level of detail constitutes an answer
- You found a function that MIGHT be relevant but you're not sure it's on the execution path
