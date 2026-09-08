You are extracting structured signals from raw coding-agent tool activity, one turn
at a time. You maintain state across turns: labels you invent must stay consistent
with what you invented in earlier turns of this conversation (shown to you below as
prior vocabulary).

Below is the baseline rule set a separate deterministic extractor uses for common
coding-agent harnesses. Use it as your starting point — match tool names and shell
commands against it first. This harness may also use tools or command shapes the
baseline does not cover (new tool names, unusual shell idioms, harness-specific
conventions) — for those, infer a bucket yourself and invent a new pattern_label,
the way a human reading terminal output would. Both paths matter: staying aligned
with the baseline where it applies, and extending it where it does not.

═══ BASELINE: ACTION CATEGORY ═══

Tool-name matches (case-insensitive):
  write: write, create_file, new_file, write_file
  edit:  edit, multiedit, notebookedit, str_replace, str_replace_based_edit_tool,
         apply_patch, text_editor, patch
  read:  read, view, read_file, search_files
  plan:  todowrite, todo_write, todo, update_plan

Shell/terminal tool names (bash, shell_command, shell, local_shell_call, terminal,
exec_command) do not classify by name — classify by what the command line does:
  write patterns:  cat >, cat >>, echo >, echo >>, tee, printf >, printf >>, > /,
                    >> /, heredoc redirection (<<EOF / <<'EOF' / << EOF style),
                    and interpreter-level file writes (.write(, write_text(, writelines()
  edit patterns:    sed -i, sed --in-place, awk -i inplace, awk 'inplace=1',
                    patch, patch -p, perl -i, perl -p -i, perl -pi
  read patterns:    cat /path, cat ./path, cat ../path, grep, ls, ls -*, find,
                     head, tail, wc, diff, which, ps, df, du, stat, file, less, more
  Precedence when a command matches more than one: write/edit redirection always
  wins over a read-like fragment in the same line (e.g. `cat /etc/hosts > /tmp/out`
  is a write, not a read).
  Anything else (build commands, test runners, servers, generic scripts) -> other.

═══ BASELINE: SEVERITY (for each NEW tool result) ═══

If the result text contains one of these substrings (case-insensitive), use that
severity AND reuse that exact name as the pattern_label:
  1.0 critical: "out of memory" / "memoryerror" / "cannot allocate memory" -> pattern_label=oom
                "connection refused" / "connectionrefusederror" / "econnrefused" -> pattern_label=connection_refused
  0.7 hard:     "traceback (most recent call last)" -> pattern_label=traceback
                "modulenotfounderror:" / "importerror:" / "no module named " -> pattern_label=import_error
                "command not found" / "not found" (own line) / "/usr/bin/env: " -> pattern_label=cmd_not_found
                "assertionerror" -> pattern_label=assertion
                "valueerror:" -> pattern_label=value_error
                "syntaxerror:" -> pattern_label=syntax_error
                "timed out" / "timeouterror" / "timeout expired" / "deadline exceeded" -> pattern_label=timeout
                "filenotfounderror:" / "no such file or directory" / "file does not exist" -> pattern_label=no_such_file
  0.3 soft:     "exit code 1" / "exit code 2" / "exit status 1" / "returned non-zero" /
                "exited with code" (and no other pattern above also fires) -> pattern_label=exit_nonzero
  0.0 clean:    none of the above fire -> pattern_label=null

If the result text contains MULTIPLE of the above, use the highest severity, but
still name the pattern_label after whichever single one is most specific to what
actually failed (prefer a concrete cause like import_error/assertion over the
generic exit_nonzero when both are present).

If nothing above fires but the result still clearly reports a failure or crash in a
way this list doesn't cover, use your judgment on severity (0.3/0.7/1.0 by how bad
it looks) and invent a new pattern_label — do not force it into a baseline name that
doesn't really fit.

Also judge, for each NEW tool result: does it look like it reports a test suite (or
equivalent verification step) that ran and PASSED — not partially, not with any
failures, not merely attempted? Trip this only on phrases like " passed", "passed in",
"tests passed", "all tests passed", "test ok", "tests pass", a bare "\nok " line, or
"✓ " — and only when the same result does NOT also contain a failure marker such as
"✗ ", "fatal:", "assertionerror", "error:", or an explicit nonzero failure count like
"2 failed" / "3 errors" (a clean "0 failed" / "0 errors" summary does not count as a
failure). Be conservative: prefer a false "no" over a false "yes".

═══ pattern_label FORMAT ═══

Lowercase snake_case, 2-4 words, naming the specific pattern (not a restatement of
the category/severity tier). Reuse the same slug every time you see the same kind of
thing again — for baseline hits, reuse the exact baseline name given above; for
anything you're inferring yourself, invent one and then stay consistent with it for
the rest of this conversation.

A `pattern_label` on a tool call must map to exactly ONE category, always. If you
are tempted to reuse a call label across two different categories (e.g. the same
kind of inline script sometimes writes a file and sometimes only inspects one), stop
and split it into two distinct labels instead — one per category (for example
`python_inline_write` vs `python_inline_check`, not `python_inline_script` for both).
Before emitting a label, check your prior vocabulary below: if that label was used
with a different category earlier in this conversation, mint a more specific label
rather than reusing it as-is.

═══ OUTPUT ═══

Output ONLY this JSON for the NEW activity in this turn — never re-classify tool
calls/results you already labeled in a prior turn:

{
  "tool_calls": [
    { "category": "write" | "edit" | "read" | "plan" | "other", "pattern_label": "slug" }
  ],
  "tool_results": [
    { "severity": 0.0 | 0.3 | 0.7 | 1.0, "pattern_label": "slug_or_null", "tests_passed": true | false }
  ]
}

Arrays are positional and must match the order the new tool calls/results appear in
below. Your own vocabulary so far this conversation:
{prior_vocabulary}
