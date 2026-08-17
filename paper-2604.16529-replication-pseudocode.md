# Replication pseudocode for *Scaling Test-Time Compute for Agentic Coding*

Source: [arXiv:2604.16529v1](https://arxiv.org/abs/2604.16529), submitted April 16,
2026. The manuscript date printed in the PDF is April 21, 2026.

This document is a standalone implementation specification for the paper's Recursive
Tournament Voting (RTV), agentic Parallel-Distill-Refine (PDR), combined PDR+RTV method,
ablations, and analyses. It includes the appendices' operational details and example-derived
summary shapes. It does not require the paper to understand or implement the method.

## 1. Replication contract and disclosure labels

Use these labels throughout the implementation:

- `PAPER`: stated in the paper or recoverable exactly from its equations, tables, figures, or
  supplied appendix examples.
- `EXAMPLE-DERIVED`: observed in an appendix output, but not published as a required prompt,
  schema, parser, or runtime policy.
- `RECONSTRUCTED`: an implementation choice needed to make the method executable, but not
  specified by the paper. Keep it configurable and record its value in every run manifest.
- `REPO`: a behavior confirmed later against public author code. A repository audit should
  replace a `RECONSTRUCTED` choice with `REPO` only when the code unambiguously establishes it.

Do not silently invent a value for a replication-critical field. The controller must either load
it from an experiment config or stop with an error naming the missing field.

The paper does not train a new model. It runs inference-time control flow around existing
coding agents. For each evaluated model, the same language model acts as:

1. the coding agent that creates each rollout;
2. the summarizer of its own rollout;
3. the judge that compares summaries of its own rollouts.

The method makes selection decisions without official ground-truth outcomes, hidden tests, test
samples, reward values, or grader traces. Those values must never enter summaries, judge prompts,
refinement prompts, tournament pairing, or tie-breaking. Agents may inspect and run tests that the
benchmark deliberately exposes inside their task environment; the ban applies to official hidden
evaluation data.

Public-code audit, August 17, 2026: no official or author-linked implementation was found. The two
located repositories explicitly identify themselves as independent reproductions, so they do not
resolve any `RECONSTRUCTED` setting and nothing in this document is labeled `REPO`.

## 2. Exact experiment constants stated by the paper

```text
MAIN_PAPER_CONSTANTS = {
    # Parallel rollout count per iteration.
    N: 16,                         # PAPER

    # Total rollout iterations: iteration 0 and iteration 1.
    T: 2,                          # PAPER

    # Number of iteration-0 survivors placed in the refinement context.
    K: 4,                          # PAPER

    # RTV comparison group size.
    G: 2,                          # PAPER

    # Independent judge votes per comparison group.
    V: 8,                          # PAPER

    # Each iteration-1 rollout starts from the task's original clean image/snapshot.
    fresh_environment_per_rollout: true,    # PAPER

    # Reuse summaries only; do not copy prior patches, files, shells, or containers.
    persistent_workspace_between_iterations: false,  # PAPER
}
```

Benchmarks and task sets:

```text
BENCHMARKS = [
    {
        name: "SWE-Bench Verified",
        task_count: 500,                 # full test set; PAPER
        harness: "mini-SWE-agent",
        harness_mode: "bash-only",       # PAPER
        outcome: binary pass/fail,
    },
    {
        name: "Terminal-Bench v2.0",
        task_count: 88,
        available_task_count: 89,
        harness: "Terminus 1",           # PAPER
        outcome: binary pass/fail,
        # The paper does not identify the excluded task.
        excluded_task_id: REQUIRED_CONFIG,
    },
]
```

Models:

```text
MODELS = [
    "Claude-4.5-Opus",
    "Gemini-3.1-Pro",
    "Claude-4.5-Sonnet",
    "Gemini-3-Flash",
    "GPT-5-0825",
]
```

The paper does not state exact provider model IDs, API revisions, sampling temperatures, top-p,
token limits, context limits, agent step/time limits, retry policies, seeds, container digests,
or harness commit hashes. Treat each as a required run-manifest field until public code or author
configuration supplies it.

## 3. State and record types

Use immutable persisted records for scientific bookkeeping. A local builder may change while a
rollout or tournament is running, but `PERSIST` freezes it; later facts use append-only link or
evaluation records. Store large text blobs by content hash, but preserve a lossless copy.

```text
record TaskSpec:
    benchmark_name: string
    task_id: string
    problem_text: string
    clean_environment_image: string       # immutable image/digest
    clean_repository_revision: string | null
    official_grader_config: map
    public_task_assets: list[Asset]

record InferenceRoleConfig:
    provider: string
    exact_model_id: string
    api_revision: string | null
    temperature: float
    top_p: float | null
    max_output_tokens: int
    context_window: int
    stop_sequences: list[string]
    seed_policy: string
    prompt_text: string
    prompt_sha256: string
    tool_protocol: string | null
    retry_policy: RetryPolicy

record RetryPolicy:
    max_attempts: int
    retryable_error_kinds: list[string]
    deterministic_backoff_policy: string

record SummaryPolicy:
    serializer_revision: string
    overflow_policy: string
    max_content_attempts: int
    malformed_output_policy: string
    exhausted_call_policy: enum {ABORT_RUN, RETRY_CONTENT_ATTEMPT}

record InvalidVotePolicy:
    mode: enum {ABORT_RUN, REPLACE_UNTIL_V}
    max_replacement_calls_per_group: int

record PaperConstants:
    N: int
    T: int
    K: int
    G: int
    V: int
    fresh_environment_per_rollout: bool
    persistent_workspace_between_iterations: bool

record ExecutionPolicies:
    summary_policy: SummaryPolicy
    summary_schema_or_null: map | null
    pairing_policy: string
    display_order_policy: string
    tie_policy: string
    invalid_vote_policy: InvalidVotePolicy

record ExperimentConfig:
    paper_constants: PaperConstants
    execution: ExecutionPolicies
    benchmark_manifest: map
    agent_limits_by_benchmark_and_model: map
    experiment_seed: int
    replication_label: enum {EXACT_REPLICATION, CONCEPTUAL_REPLICATION}
    unresolved_field_names: list[string]
    manifest_schema_revision: string

record ModelBundle:
    logical_name: string
    action_role: InferenceRoleConfig
    summary_role: InferenceRoleConfig
    comparison_role: InferenceRoleConfig
    refinement_prompt_sha256: string

record Asset:
    path: string
    sha256: string
    visibility: enum {PUBLIC_TO_AGENT, GRADER_ONLY}

record Artifact:
    path: string
    sha256: string
    mode: int
    size_bytes: int

record Usage:
    input_tokens: int | null
    output_tokens: int | null
    cached_input_tokens: int | null
    provider_cost: decimal | null

record AgentLimits:
    max_steps: int
    wall_time_seconds: int
    per_command_time_seconds: int
    max_observation_bytes_per_step: int
    max_total_trajectory_bytes: int

record Action:
    thought: string                        # T_i
    bash_commands: list[string]            # B_i = {b_1, ..., b_m}
    scaffold_action: map                   # lossless fields such as keystrokes/is_blocking/timeout
    raw_model_response: string
    model_usage: Usage

record Observation:
    command: string
    exit_code: int | null
    stdout: bytes
    stderr: bytes
    timed_out: bool
    started_at: timestamp
    finished_at: timestamp

record TrajectoryStep:
    step_index: int
    context_hash_before: string
    action: Action                         # A_i = (T_i, B_i)
    observations: list[Observation]        # O_i
    context_hash_after: string

record Rollout:
    run_id: string
    task_id: string
    model_id: string
    iteration: int
    rollout_index: int
    seed: int
    environment_id: string
    refinement_summary_ids: list[string]
    steps: list[TrajectoryStep]
    termination_reason: enum {
        AGENT_FINISHED,
        MAX_STEPS,
        WALL_TIME,
        COMMAND_TIMEOUT,
        MODEL_ERROR,
        PARSE_ERROR,
        INFRASTRUCTURE_ERROR
    }
    final_agent_message: string | null
    final_patch: bytes | null               # SWE-Bench-like output
    final_artifact_manifest: list[Artifact] # Terminal-Bench-like output
    final_environment_snapshot: string

record StructuredSummary:
    summary_id: string
    rollout_id: string
    benchmark_name: string
    schema_version: string | null
    json_value: map
    raw_model_response: string
    parse_attempts: int
    validation_warnings: list[string]
    model_usage: Usage

record RolloutSummaryLink:
    rollout_id: string
    summary_id: string

record RolloutEvaluation:
    rollout_id: string
    grader_revision: string
    binary_outcome: bool
    score_details: map

record Candidate:
    candidate_id: string
    rollout_id: string
    summary_id: string | null
    display_payload: string                 # summary JSON, or raw trace in one ablation

record Vote:
    tournament_id: string
    iteration: int
    round_index: int
    group_index: int
    vote_index: int
    ordered_candidate_ids: list[string]
    prompt_sha256: string
    raw_model_response: string | null
    selected_display_position: int | null
    selected_candidate_id: string | null
    parse_error: string | null
    model_usage: Usage

record GroupDecision:
    round_index: int
    group_index: int
    effective_group_size: int
    input_candidate_ids: list[string]
    votes: list[Vote]
    vote_counts: map[candidate_id, int]
    selected_candidate_id: string
    tie_break_record: map | null

record Tournament:
    tournament_id: string
    task_id: string
    model_id: string
    rollout_iteration: int
    target_survivor_count: int
    configured_G: int
    V: int
    root_seed: int
    pairing_policy_hash: string
    display_order_policy_hash: string
    tie_policy_hash: string
    rounds: list[list[GroupDecision]]
    population_ids_by_round: list[list[string]]
    survivor_candidate_ids: list[string]

record TournamentCheckpoint:
    checkpoint_id: string
    tournament_id: string
    survivor_count: int
    completed_round_count: int
    survivor_candidate_ids: list[string]
    population_sha256: string

record EvaluationRow:
    task_id: string
    model_id: string
    stage: enum {ITER_0, SELECT_K, ITER_1, FINAL}
    candidate_ids: list[string]
    binary_outcomes: list[bool]

record TaskRun:
    task_id: string
    model_logical_name: string
    iter0_rollout_ids: list[string]
    iter0_summary_ids: list[string]
    iter0_tournament_id: string
    select_k_checkpoint_id: string
    refinement_summary_ids: list[string]
    iter1_rollout_ids: list[string]
    iter1_summary_ids: list[string]
    final_tournament_id: string
    final_rollout_id: string
    output_patch: bytes | null
    output_artifacts: list[Artifact]
    output_environment_snapshot: string
```

Every record must also carry the experiment manifest hash, code revision, prompt revision, model
configuration, harness revision, container digest, and creation timestamp, either directly or by
foreign key.

## 4. Agent rollout dynamics

The paper defines a rollout as interleaved agent actions and environment observations. At step
`i`, the accumulated context is `C_(i-1)`:

```text
A_i = (T_i, B_i) = LM(P_action(P_in; C_(i-1)))
O_i = Environment(C_(i-1); B_i)
C_i = append(C_(i-1), (A_i, O_i))
```

Executable pseudocode:

```text
interface AgentScaffoldAdapter:
    # Returns user/tool context only; CALL_ROLE adds the versioned role prompt once.
    build_initial_context(task, refinement_summaries)
    parse_action(raw_response) -> ScaffoldAction
    execute_action(environment, ScaffoldAction, limits) -> list[Observation]
    is_complete(ScaffoldAction) -> bool
    final_message(ScaffoldAction) -> string | null
    serialize_step_for_next_context(context, action, observations)
    serialize_lossless_trajectory(rollout)
    extract_patch(environment) -> bytes | null
    list_output_artifacts(environment) -> list[Artifact]

SWE_ADAPTER = MiniSWEAgentBashAdapter(
    exact_revision = REQUIRED_CONFIG,
    prompt_and_protocol_revision = REQUIRED_CONFIG,
    observation_truncation_policy = REQUIRED_CONFIG,
)
TERMINAL_ADAPTER = Terminus1Adapter(
    exact_revision = REQUIRED_CONFIG,
    prompt_and_protocol_revision = REQUIRED_CONFIG,
    observation_truncation_policy = REQUIRED_CONFIG,
)

function RUN_ROLLOUT(task, model_bundle, scaffold, limits, iteration, rollout_index,
                     refinement_summaries, seed) -> Rollout:
    assert iteration == 0 implies refinement_summaries is empty
    assert iteration > 0 implies len(refinement_summaries) > 0

    env = CREATE_FRESH_ENVIRONMENT(
        image = task.clean_environment_image,
        repository_revision = task.clean_repository_revision,
        public_assets = task.public_task_assets,
    )

    # Critical paper invariant: a new container/snapshot is created for every rollout,
    # including every refined rollout. No filesystem state crosses rollout boundaries.
    assert env.has_no_parent_rollout_state()

    context = scaffold.build_initial_context(
        task = task,
        refinement_summaries = refinement_summaries,
    )

    rollout = new Rollout(
        run_id = UNIQUE_ID(),
        task_id = task.task_id,
        model_id = model_bundle.action_role.exact_model_id,
        iteration = iteration,
        rollout_index = rollout_index,
        seed = seed,
        environment_id = env.id,
        refinement_summary_ids = [s.summary_id for s in refinement_summaries],
        steps = [],
    )

    deadline = NOW() + limits.wall_time_seconds

    for step_index in 0 .. limits.max_steps - 1:
        if NOW() >= deadline:
            rollout.termination_reason = WALL_TIME
            break

        response = CALL_ROLE_WITH_RECORDED_RETRIES(
            role = model_bundle.action_role,
            messages = context,
            tools = model_bundle.action_role.tool_protocol,
            seed = DERIVE_SEED(seed, "action", step_index),
        )

        if response.failed:
            rollout.termination_reason = MODEL_ERROR
            RECORD_FAILURE(response)
            break

        parse = scaffold.parse_action(response.raw)
        if parse.failed:
            rollout.termination_reason = PARSE_ERROR
            RECORD_FAILURE(parse)
            break

        action = Action(
            thought = parse.action.thought,
            bash_commands = parse.action.commands,
            scaffold_action = parse.action.lossless_value,
            raw_model_response = response.raw,
            model_usage = response.usage,
        )

        observations = scaffold.execute_action(env, parse.action, limits)
        new_context = scaffold.serialize_step_for_next_context(
            context, parse.action, observations
        )
        rollout.steps.append(TrajectoryStep(
            step_index = step_index,
            context_hash_before = SHA256(context),
            action = action,
            observations = observations,
            context_hash_after = SHA256(new_context),
        ))
        context = new_context

        if scaffold.is_complete(parse.action):
            rollout.termination_reason = AGENT_FINISHED
            rollout.final_agent_message = scaffold.final_message(parse.action)
            break
    else:
        rollout.termination_reason = MAX_STEPS

    rollout.final_patch = scaffold.extract_patch(env)
    rollout.final_artifact_manifest = scaffold.list_output_artifacts(env)
    rollout.final_environment_snapshot = env.SNAPSHOT_IMMUTABLY()
    scaffold.serialize_lossless_trajectory(rollout)
    PERSIST_LOSSLESS(rollout)
    return rollout
```

The appendix shows two different scaffold protocols, so one global `BASH_TOOL_SCHEMA` or parser is
not sufficient. The mini-SWE-agent trace uses `THOUGHT`, `<bash>`, `<returncode>`, and `<output>`.
The Terminus 1 trace uses JSON fields including `state_analysis`, `explanation`,
`bash_commands[{keystrokes,is_blocking,timeout_sec}]`, and `is_task_complete`. Exact scaffold
prompts, revisions, and serialization remain required configuration.

The scaffold's `build_initial_context` is the only conceptual difference between initial and
refined rollouts:

```text
function BUILD_INITIAL_AGENT_CONTEXT(task, refinement_summaries):
    # User-context messages only. CALL_ROLE_WITH_RECORDED_RETRIES adds the action-role prompt.
    messages = []

    if refinement_summaries is not empty:
        messages.append({role: "user", content: REFINEMENT_PREAMBLE})
        for position, summary in enumerate(refinement_summaries, start = 1):
            messages.append({
                role: "user",
                content:
                    "PRIOR ATTEMPT SUMMARY " + position + "\n" +
                    CANONICAL_JSON(summary.json_value),
            })
        messages.append({role: "user", content: REFINEMENT_POSTAMBLE})

    messages.append({role: "user", content: FORMAT_ORIGINAL_TASK(task.problem_text)})
    return messages
```

The paper states that the first refined action is conditioned on the original task and the
distilled prior summaries; later actions retain that refinement context while following normal
agent dynamics. Do not present refinement summaries as verified facts. They can describe failed
or mistaken attempts.

## 5. Structured rollout summarization

For rollout `R_i`, the paper computes:

```text
S_i = LM(P_sum(R_i))
```

The paper does not disclose how `R_i` is serialized, reduced when it exceeds context, or combined
with the task, patch, and artifact list. It also does not disclose a formal output schema, parsing,
repair, retry, exclusion, or replacement policy. A faithful implementation therefore makes those
choices part of a versioned adapter instead of presenting them as paper behavior.

```text
interface SummaryInputAdapter:
    serialize(task, rollout, policy) -> string
    parse(raw_response, configured_schema_or_null) -> {
        json_value: map | null,
        validation_warnings: list[string],
    }

function SUMMARIZE_ROLLOUT(task, rollout, model_bundle, input_adapter,
                           summary_policy, configured_schema_or_null) -> StructuredSummary:
    summary_input = input_adapter.serialize(
        task = task,
        rollout = rollout,
        policy = summary_policy,
    )

    for attempt in 1 .. summary_policy.max_content_attempts:
        response = CALL_ROLE_WITH_RECORDED_RETRIES(
            role = model_bundle.summary_role,
            messages = [{role: "user", content: summary_input}],
            seed = DERIVE_SEED(rollout.seed, "summary", attempt),
        )
        PERSIST_RAW_RESPONSE_BEFORE_PARSING(response)
        if response.failed:
            disposition = HANDLE_EXHAUSTED_MODEL_CALL(
                call_kind = "summary",
                policy = summary_policy.exhausted_call_policy,
                attempt = attempt,
                max_content_attempts = summary_policy.max_content_attempts,
                failed_response = response,
            )
            if disposition == RETRY_CONTENT_ATTEMPT:
                continue
            raise SummaryModelCallFailed(rollout.run_id, response.attempt_records)

        parsed = input_adapter.parse(response.raw, configured_schema_or_null)

        if parsed.json_value is not null:
            summary = StructuredSummary(
                summary_id = UNIQUE_ID(),
                rollout_id = rollout.run_id,
                benchmark_name = task.benchmark_name,
                schema_version = VERSION(configured_schema_or_null),
                json_value = parsed.json_value,
                raw_model_response = response.raw,
                parse_attempts = attempt,
                validation_warnings = parsed.validation_warnings,
                model_usage = response.usage,
            )
            PERSIST(summary)
            PERSIST(RolloutSummaryLink(rollout.run_id, summary.summary_id))
            return summary

        summary_input = APPLY_CONFIGURED_MALFORMED_SUMMARY_POLICY(
            policy = summary_policy.malformed_output_policy,
            original_input = summary_input,
            invalid_output = response.raw,
            warnings = parsed.validation_warnings,
        )

    raise SummaryGenerationFailed(rollout.run_id)
```

### 5.1 SWE-Bench summary shape observed in the appendix

`EXAMPLE-DERIVED`: the paper's one SWE-Bench example is a JSON object with the following keys and
observed value shapes. It is not a published JSON Schema. A formal schema, if used, must tolerate
the example's string/list and evidence-value variation or convert it under a recorded policy.

```text
OBSERVED_SWE_SUMMARY_EXAMPLE_SHAPE = {
  issue_requirements: {
    primary_objective: string,
    expected_behavior: string,
    current_behavior: string,
    reproduction_steps: string,
    explicit_constraints: list[string],
    success_criteria: list[string]
  },
  agent_actions: {
    exploration: {
      files_examined: list[string],
      codebase_navigation: list[string],
      commands_used: list[string],
      key_discoveries: list[string]
    },
    solution_approach: {
      strategy: string,
      stated_reasoning: string,
      approach_changes: list[string]
    },
    implementation: {
      modification_scope: string,
      files_modified: list[string],
      edit_methods: list[string],
      command_dependencies: list[string]
    }
  },
  code_changes: list[{
    description: string,
    file_path: string,
    function_method: string,
    condition_added: string,
    construct_added: string,
    before_code: string,
    after_code: string
  }],
  command_execution: {
    commands_with_response: list[{
      step: string,
      command: string,
      return_code: int,
      stdout_summary: string,
      stderr_summary: string
    }],
    commands_without_response: list[{
      step: string,
      command: string,
      intended_purpose: string,
      depended_on_by: list[string]
    }],
    error_responses: list[{
      step: string,
      command: string,
      return_code: int,
      error_message: string,
      addressed_in_subsequent_steps: bool
    }]
  },
  error_analysis: {
    intermediate_errors: list[{
      error_type: string,
      classification: string,
      evidence: string,
      resolved_before_submission: bool,
      resolution_evidence: string
    }],
    final_state: {
      status: string,
      errors: list[{
        error_type: string,
        classification: string,
        evidence: string
      }]
    }
  },
  evidence_inventory: {
    confirmed_outcomes: list[{
      step: string,
      command: string,
      outcome: string,
      key_output: string
    }],
    error_outcomes: list[{
      step: string,
      command: string,
      return_code: int,
      outcome: string,
      error_message: string
    }],
    unconfirmed_actions: list[{
      step: string,
      command: string,
      intended_purpose: string,
      dependent_commands: string | list[string]
    }]
  },
  final_state: {
    agent_claims: {
      completion_claim: string,
      reasoning_provided: string,
      uncertainty_expressed: string | list[string]
    },
    expected_artifacts: list[{
      artifact_or_change: string,
      existence_checked: bool | string,
      content_examined: bool | string,
      observation: string
    }],
    unresolved_issues: list[string]
  },
  patch_status: {
    status: string,
    success_patterns_found: list[string],
    failure_patterns_found: list[string]
  },
  requirement_coverage: list[{
    requirement: string,
    evidence_locations: list[string],
    related_actions: list[string],
    related_code_changes: list[string]
  }],
  verification_record: {
    verification_commands: list[{
      step: string,
      command: string,
      return_code: int,
      what_it_checked: string,
      has_response_block: bool
    }],
    verification_coverage: {
      verified_with_response: list[string],
      attempted_but_unconfirmed: list[string],
      not_verified: list[string],
      test_commands_executed: list[string],
      test_results: string | list[string]
    }
  }
}
```

### 5.2 Terminal-Bench summary shape observed in the appendix

`EXAMPLE-DERIVED`: the paper's one Terminal-Bench summary has the following keys and observed
value shapes. It likewise does not establish a formal required schema.

```text
OBSERVED_TERMINAL_SUMMARY_EXAMPLE_SHAPE = {
  task_requirements: {
    primary_objective: string,
    output_specifications: list[{
      artifact: string,
      format: string,
      verbatim_evidence: string
    }],
    explicit_constraints: list[string],
    success_criteria: list[{
      criterion: string,
      verbatim_evidence: string
    }],
    task_assertions: {broken_or_needs_fixing: string}
  },
  agent_actions: {
    exploration: {
      files_directories_examined: list[string],
      diagnostic_commands: list[string],
      key_information_discovered: list[string]
    },
    solution_approach: {
      strategy: string,
      stated_reasoning: string,
      approach_changes: list[string]
    },
    implementation_structure: {
      structure_type: string,
      intermediate_files: list[string],
      command_dependencies: list[{command: string, depends_on: string}]
    }
  },
  command_execution_record: {
    commands_with_response: list[{
      step: string,
      command: string,
      return_code: int,
      stdout_summary: string,
      stderr_summary: string
    }],
    commands_without_response: list[{
      step: string,
      command: string,
      intended_purpose: string,
      depended_on_by: list[string]
    }],
    error_responses: list[map]
  },
  evidence_inventory: {
    confirmed_outcomes: list[{
      step: string,
      command: string,
      description: string,
      key_output: string
    }],
    error_outcomes: list[map],
    unconfirmed_actions: list[{
      step: string,
      command: string,
      intended_purpose: string,
      depends_on_this: list[string]
    }],
    unverified_aspects: list[{aspect: string, factual_relevance: string}]
  },
  final_state: {
    agent_assessment: {
      completion_claim: string,
      reasoning_provided: string,
      uncertainty_expressed: string | list[string],
      verbatim_quote: string
    },
    expected_artifacts: list[{
      artifact: string,
      existence_checked: bool,
      existence_check_command: string,
      existence_check_step: string,
      content_examined: bool,
      content_examine_command: string,
      content_examine_step: string,
      observation: string
    }],
    unresolved_issues: list[string]
  },
  resource_access: {
    resources_accessed: {
      input_files_read: list[string],
      intermediate_files: list[string],
      output_files_created: list[{file_path: string, command_used: string}],
      external_resources: list[string]
    },
    access_timing: list[{resource: string, timing: string, description: string}],
    runtime_dependencies: {
      files_required_at_runtime: list[string],
      tools_invoked_at_runtime: list[string],
      environment_conditions: list[string]
    }
  },
  solution_characteristics: {
    code_script_generation: {
      applicable: bool,
      execution_behavior: string,
      external_files_accessed_at_runtime: list[string],
      access_method: string,
      contains_embedded_data: bool
    },
    debugging_fix: {
      applicable: bool,
      problem_identification: string,
      fix_implemented: string,
      no_fix_conclusion: string,
      test_inputs_used: list[string]
    },
    data_processing: {
      applicable: bool,
      processing_method: string,
      input_files_used: list[string],
      intermediate_files_created: list[string],
      final_output_produced: string
    }
  },
  verification_record: {
    verification_commands: list[{
      step: string,
      command: string,
      return_code: int,
      what_it_checked: string,
      has_response_block: bool
    }],
    verification_coverage: {
      verified_with_response: list[string],
      attempted_but_unconfirmed: list[string],
      not_verified: list[string],
      inputs_scenarios_tested: list[string],
      inputs_scenarios_not_tested: list[string]
    }
  }
}
```

The examples show that summaries distinguish actions with captured responses from actions merely
claimed by the agent, record intermediate errors and whether they were resolved, identify final
artifacts and runtime dependencies, and map verification evidence to requirements. Preserve these
distinctions; they are important inputs to the judge.

## 6. Recursive Tournament Voting

RTV receives `N` candidates, partitions them into groups of `G`, collects `V` independent model
votes per group, retains one candidate per group, and repeats until a requested survivor count is
reached. Main experiments use pairwise groups and eight votes.

For group `j` in round `r`, the paper defines the selected display position as:

```text
g_j^(r) = argmax over g in {1, ..., G} of
          sum over v in {1, ..., V} of
          1[ LM(P_comp(P_in; S_(j,1)^(r), ..., S_(j,G)^(r))) == g ]
```

The selected rollouts become the next population. Summaries remain attached to their original
rollouts; survivors are not re-summarized between tournament rounds.

```text
function COLLECT_RTV_VOTE(tournament, rollout_iteration, round_index, group_index,
                          vote_record_index, displayed_group, prompt, model_bundle,
                          effective_G, call_seed) -> Vote:
    response = CALL_ROLE_WITH_RECORDED_RETRIES(
        role = model_bundle.comparison_role,
        messages = [{role: "user", content: prompt}],
        seed = call_seed,
    )
    PERSIST_RAW_RESPONSE_BEFORE_PARSING(response)
    PERSIST_PROVIDER_ATTEMPTS(response.attempt_records)

    common = {
        tournament_id: tournament.tournament_id,
        iteration: rollout_iteration,
        round_index: round_index,
        group_index: group_index,
        vote_index: vote_record_index,
        ordered_candidate_ids: IDS(displayed_group),
        prompt_sha256: SHA256(prompt),
        model_usage: response.usage,
    }

    if response.failed:
        return Vote(
            **common,
            raw_model_response = null,
            selected_display_position = null,
            selected_candidate_id = null,
            parse_error = "MODEL_CALL_FAILED:" + response.error_kind,
        )

    position = PARSE_FINAL_VERDICT(response.raw, allowed = 1 .. effective_G)
    if position is invalid:
        return Vote(
            **common,
            raw_model_response = response.raw,
            selected_display_position = null,
            selected_candidate_id = null,
            parse_error = position.error,
        )

    return Vote(
        **common,
        raw_model_response = response.raw,
        selected_display_position = position,
        selected_candidate_id = displayed_group[position - 1].candidate_id,
        parse_error = null,
    )

function RUN_RTV(task, model_bundle, candidates, configured_G, V, target_survivors,
                 policies, seed, freeze_checkpoint_at = null)
                 -> (Tournament, TournamentCheckpoint | null):
    assert 1 <= target_survivors <= len(candidates)
    assert configured_G >= 2

    # The paper's main N=16, G=2, K=4 cases divide exactly in every used round.
    # Fail instead of inventing bye behavior for a non-divisible population unless an explicit
    # policy is configured.
    population = COPY(candidates)
    rollout_iteration = UNIQUE_VALUE([
        RESOLVE_ROLLOUT(candidate.rollout_id).iteration for candidate in candidates
    ])
    tournament = new Tournament(
        tournament_id = UNIQUE_ID(),
        task_id = task.task_id,
        model_id = model_bundle.logical_name,
        rollout_iteration = rollout_iteration,
        target_survivor_count = target_survivors,
        configured_G = configured_G,
        V = V,
        root_seed = seed,
        pairing_policy_hash = HASH(policies.pairing_policy),
        display_order_policy_hash = HASH(policies.display_order_policy),
        tie_policy_hash = HASH(policies.tie_policy),
        rounds = [],
        population_ids_by_round = [],
        survivor_candidate_ids = [],
    )
    frozen_checkpoint = null
    round_index = 0
    tournament.population_ids_by_round.append(IDS(population))

    while len(population) > target_survivors:
        # The G=8 ablation is 16 -> 2 -> 1, so the last round must shrink the configured group
        # size from eight to two. G=4 similarly uses [4, 4].
        effective_G = MIN(configured_G, len(population))
        if len(population) mod effective_G != 0:
            raise UndefinedByePolicy(len(population), effective_G)
        if len(population) / effective_G < target_survivors:
            raise TournamentWouldOvershootTarget(
                population_size = len(population),
                effective_group_size = effective_G,
                requested_survivors = target_survivors,
            )

        ordered_population = APPLY_PAIRING_POLICY(
            population,
            policies.pairing_policy,
            DERIVE_SEED(seed, "pairing", round_index),
        )
        groups = CHUNK(ordered_population, size = effective_G)
        results = ARRAY(size = len(groups))

        PARALLEL_FOR group_index, group in enumerate(groups):
            votes = []
            vote_contexts = ARRAY(size = V)

            for vote_index in 0 .. V - 1:
                displayed_group = APPLY_DISPLAY_ORDER_POLICY(
                    group,
                    policies.display_order_policy,
                    DERIVE_SEED(seed, "display", round_index, group_index, vote_index),
                )
                prompt = RENDER_COMPARISON_PROMPT(
                    task_text = task.problem_text,
                    candidates = displayed_group,
                    benchmark_name = task.benchmark_name,
                )
                vote_contexts[vote_index] = {
                    original_vote_index: vote_index,
                    displayed_group: displayed_group,
                    prompt: prompt,
                }
                votes.append(COLLECT_RTV_VOTE(
                    tournament = tournament,
                    rollout_iteration = rollout_iteration,
                    round_index = round_index,
                    group_index = group_index,
                    vote_record_index = vote_index,
                    displayed_group = displayed_group,
                    prompt = prompt,
                    model_bundle = model_bundle,
                    effective_G = effective_G,
                    call_seed = DERIVE_SEED(
                        seed, "vote", round_index, group_index, vote_index
                    ),
                ))

            valid_votes = [v for v in votes if v.selected_candidate_id is not null]
            if len(valid_votes) != V:
                invalid_contexts = [
                    vote_contexts[index]
                    for index in 0 .. V - 1
                    if votes[index].selected_candidate_id is null
                ]
                policy = policies.invalid_vote_policy
                if policy.mode == ABORT_RUN:
                    raise IncompleteVoteSet(
                        expected = V,
                        actual = len(valid_votes),
                        vote_records = votes,
                    )

                assert policy.mode == REPLACE_UNTIL_V
                replacement_index = 0
                while (
                    len(valid_votes) < V
                    and replacement_index < policy.max_replacement_calls_per_group
                ):
                    source = invalid_contexts[replacement_index mod len(invalid_contexts)]
                    replacement = COLLECT_RTV_VOTE(
                        tournament = tournament,
                        rollout_iteration = rollout_iteration,
                        round_index = round_index,
                        group_index = group_index,
                        vote_record_index = V + replacement_index,
                        displayed_group = source.displayed_group,
                        prompt = source.prompt,
                        model_bundle = model_bundle,
                        effective_G = effective_G,
                        call_seed = DERIVE_SEED(
                            seed,
                            "replacement-vote",
                            round_index,
                            group_index,
                            source.original_vote_index,
                            replacement_index,
                        ),
                    )
                    votes.append(replacement)
                    if replacement.selected_candidate_id is not null:
                        valid_votes.append(replacement)
                    replacement_index += 1

            if len(valid_votes) != V:
                # The paper's equation sums exactly V valid votes. A recorded replacement may
                # supply an invalid slot, but the run aborts when the configured bound is exhausted.
                raise IncompleteVoteSet(
                    expected = V,
                    actual = len(valid_votes),
                    vote_records = votes,
                )

            counts = COUNT_BY(valid_votes, key = selected_candidate_id)
            assert counts is not empty
            winners = ARGMAX_KEYS(counts)
            if len(winners) == 1:
                selected_id = winners[0]
                tie_record = null
            else:
                selected_id, tie_record = BREAK_TIE_WITH_RECORDED_POLICY(
                    winners = winners,
                    group = group,
                    policy = policies.tie_policy,
                    seed = DERIVE_SEED(seed, "tie", round_index, group_index),
                )

            decision = GroupDecision(
                round_index = round_index,
                group_index = group_index,
                effective_group_size = effective_G,
                input_candidate_ids = IDS(group),
                votes = votes,
                vote_counts = counts,
                selected_candidate_id = selected_id,
                tie_break_record = tie_record,
            )
            results[group_index] = {
                decision: decision,
                survivor: FIND_BY_ID(group, selected_id),
            }

        round_decisions = [results[j].decision for j in 0 .. len(groups) - 1]
        next_population = [results[j].survivor for j in 0 .. len(groups) - 1]
        tournament.rounds.append(round_decisions)
        population = next_population
        tournament.population_ids_by_round.append(IDS(population))
        round_index += 1

        if freeze_checkpoint_at is not null and len(population) == freeze_checkpoint_at:
            assert frozen_checkpoint is null
            frozen_checkpoint = TournamentCheckpoint(
                checkpoint_id = UNIQUE_ID(),
                tournament_id = tournament.tournament_id,
                survivor_count = len(population),
                completed_round_count = round_index,
                survivor_candidate_ids = IDS(population),
                population_sha256 = HASH(IDS(population)),
            )
            PERSIST_IMMUTABLE(frozen_checkpoint)

    tournament.survivor_candidate_ids = IDS(population)
    if freeze_checkpoint_at is not null:
        assert frozen_checkpoint is not null
    PERSIST(tournament)
    return tournament, frozen_checkpoint
```

For `N=16, G=2`:

```text
16 -> 8 -> 4                         # Select-K; stop with K=4
16 -> 8 -> 4 -> 2 -> 1              # Final RTV; stop with one
```

The group-size ablation uses these per-round effective group sizes:

```text
configured G=16: [16]          # 16 -> 1
configured G=8:  [8, 2]        # 16 -> 2 -> 1
configured G=4:  [4, 4]        # 16 -> 4 -> 1
configured G=2:  [2, 2, 2, 2]  # 16 -> 8 -> 4 -> 2 -> 1
```

The paper's analysis continues the iteration-0 tournament from four to one even though the method
uses the four-candidate population as the refinement context. Preserve the `K=4` checkpoint before
continuing the diagnostic tournament.

### 6.1 Comparison output shape and reconstructed prompt contract

`EXAMPLE-DERIVED`: appendix outputs show long, evidence-based comparisons followed by an exact
line such as:

```text
Final verdict: Solution 1
```

The paper defines `P_comp` but does not publish its text. The following rubric is reconstructed
from two example outputs. The comparison input must include the original task and each candidate's
structured summary, and must not include official pass/fail outcomes:

1. restate the task requirements and explicit constraints;
2. detect disqualifying evidence, including a missing/unapplied patch, missing artifacts, dirty or
   inconsistent final state, and unresolved fatal errors;
3. assess code/output completeness and correctness;
4. assess scope and whether the solution treats a root cause or only masks a symptom;
5. map verification commands to requirements and ensure verification happened after final edits;
6. distinguish confirmed command output from unconfirmed agent claims;
7. compare alternative interpretations of an ambiguous task;
8. rank candidates with the priority order shown by the SWE appendix example:
   `disqualification > code correctness > code completeness > verification validity > test
   results > fix scope > execution evidence > interpretation`;
9. end with exactly one parseable verdict naming a displayed candidate number.

For Terminal-Bench, additionally assess constraint compliance, self-contained reproducibility,
required output files, runtime dependencies, input coverage, and whether available tests were run.

```text
function PARSE_FINAL_VERDICT(text, allowed):
    matches = REGEX_FIND_ALL(
        pattern = case_insensitive("^\\s*Final verdict:\\s*Solution\\s+(\\d+)\\s*$"),
        text = text,
        multiline = true,
    )
    if len(matches) != 1:
        return INVALID
    position = INT(matches[0].group(1))
    return position if position in allowed else INVALID
```

## 7. PDR refinement variants

Let iteration-0 rollouts be `R_1 ... R_N` and summaries be `S_1^(0) ... S_N^(0)`.

### 7.1 Single-rollout refinement ablation

```text
function SINGLE_ROLLOUT_REFINEMENT(task, model_bundle, scaffold, limits, iter0_rollouts):
    assert len(iter0_rollouts) == N
    iter1 = ARRAY(size = N)
    PARALLEL_FOR i in 0 .. N - 1:
        iter1[i] = RUN_ROLLOUT(
            task = task,
            model_bundle = model_bundle,
            scaffold = scaffold,
            limits = limits,
            iteration = 1,
            rollout_index = i,
            refinement_summaries = [SUMMARY_FOR_ROLLOUT(iter0_rollouts[i].run_id)],
            seed = DERIVE_SEED(
                EXPERIMENT_SEED, task.task_id, model_bundle.logical_name, 1, i
            ),
        )
    return iter1
```

### 7.2 Random-K PDR ablation

The paper describes sampling a separate size-`K` subset for each next-iteration rollout:

```text
function RANDOM_K_REFINEMENT(task, model_bundle, scaffold, limits, iter0_rollouts, K):
    summaries = [SUMMARY_FOR_ROLLOUT(r.run_id) for r in iter0_rollouts]
    iter1 = ARRAY(size = N)
    PARALLEL_FOR i in 0 .. N - 1:
        J_i = SAMPLE_WITHOUT_REPLACEMENT(
            population = 0 .. N - 1,
            sample_size = K,
            seed = DERIVE_SEED(
                EXPERIMENT_SEED, task.task_id, model_bundle.logical_name, "random-k", i
            ),
        )
        iter1[i] = RUN_ROLLOUT(
            task = task,
            model_bundle = model_bundle,
            scaffold = scaffold,
            limits = limits,
            iteration = 1,
            rollout_index = i,
            refinement_summaries = [summaries[j] for j in J_i],
            seed = DERIVE_SEED(
                EXPERIMENT_SEED, task.task_id, model_bundle.logical_name, 1, i
            ),
        )
    return iter1
```

The original PDR procedure selects one rollout as its final answer. For this paper's pure-PDR
ablation, do not select a single final rollout: report iteration-1 performance as average pass@1
over all N refined rollouts for every task.

```text
function PURE_PDR_ABLATION_SCORE(task_runs):
    return AVERAGE_PASS_AT_1([
        RESOLVE_ROLLOUTS(run.iter1_rollout_ids) for run in task_runs
    ])
```

### 7.3 Select-K refinement used by the full method

Run RTV on the 16 iteration-0 candidates and stop after two pairwise rounds, leaving the same four
selected summaries as the refinement context for every fresh iteration-1 rollout.

```text
function SELECT_K_REFINEMENT(task, model_bundle, scaffold, limits, iter0_candidates, K,
                             execution_policies, experiment_seed):
    selection, unused_checkpoint = RUN_RTV(
        task = task,
        model_bundle = model_bundle,
        candidates = iter0_candidates,
        configured_G = 2,
        V = 8,
        target_survivors = K,
        policies = execution_policies,
        seed = DERIVE_SEED(
            experiment_seed, task.task_id, model_bundle.logical_name, "select-k-helper"
        ),
    )
    assert unused_checkpoint is null
    selected = RESOLVE_CANDIDATES(selection.survivor_candidate_ids)
    selected_summaries = [SUMMARY_FOR_ROLLOUT(c.rollout_id) for c in selected]

    iter1 = ARRAY(size = N)
    PARALLEL_FOR i in 0 .. N - 1:
        iter1[i] = RUN_ROLLOUT(
            task = task,
            model_bundle = model_bundle,
            scaffold = scaffold,
            limits = limits,
            iteration = 1,
            rollout_index = i,
            refinement_summaries = selected_summaries,
            seed = DERIVE_SEED(
                experiment_seed, task.task_id, model_bundle.logical_name, 1, i
            ),
        )
    return selection, iter1
```

## 8. Complete PDR+RTV controller

```text
function VALIDATE_EXPERIMENT_CONFIG(config, model_bundle, task, scaffold, summary_adapter):
    c = config.paper_constants
    assert c.N > 0 and c.T == 2 and 0 < c.K <= c.N
    assert c.G >= 2 and c.V >= 1
    assert c.fresh_environment_per_rollout
    assert not c.persistent_workspace_between_iterations
    assert config.execution.summary_policy.max_content_attempts >= 1
    vote_policy = config.execution.invalid_vote_policy
    if vote_policy.mode == ABORT_RUN:
        assert vote_policy.max_replacement_calls_per_group == 0
    else:
        assert vote_policy.mode == REPLACE_UNTIL_V
        assert vote_policy.max_replacement_calls_per_group >= 1
    for role in [
        model_bundle.action_role,
        model_bundle.summary_role,
        model_bundle.comparison_role,
    ]:
        assert role.retry_policy.max_attempts >= 1
        assert role.prompt_text is not empty
        assert SHA256(role.prompt_text) == role.prompt_sha256
    assert LOOKUP_REQUIRED(
        config.agent_limits_by_benchmark_and_model,
        task.benchmark_name,
        model_bundle.logical_name,
    ) is AgentLimits

    complete_manifest = {
        experiment_config: config,
        model_bundle: model_bundle,
        benchmark_manifest: config.benchmark_manifest,
        task_spec: task,
        scaffold_adapter: scaffold,
        summary_input_adapter: summary_adapter,
    }
    # The schema marks which nullable values are legal. It reports missing keys, null required
    # values, and any REQUIRED_CONFIG sentinel recursively; it does not trust a hand-written list.
    derived_unresolved = VALIDATE_AGAINST_REQUIRED_MANIFEST_SCHEMA(
        complete_manifest,
        schema_revision = config.manifest_schema_revision,
    ).unresolved_field_paths
    assert SORT(config.unresolved_field_names) == SORT(derived_unresolved)
    if config.replication_label == EXACT_REPLICATION:
        assert derived_unresolved is empty

function RUN_PDR_RTV_FOR_TASK(task, model_bundle, config) -> TaskRun:
    scaffold = ADAPTER_FOR(task.benchmark_name)
    summary_adapter = SUMMARY_INPUT_ADAPTER_FOR(task.benchmark_name)
    VALIDATE_EXPERIMENT_CONFIG(config, model_bundle, task, scaffold, summary_adapter)
    constants = config.paper_constants
    policies = config.execution
    if constants != MAIN_PAPER_CONSTANTS:
        RECORD_DEVIATIONS(constants)
    ASSERT_SAME_PROVIDER_MODEL_REVISION(
        model_bundle.action_role,
        model_bundle.summary_role,
        model_bundle.comparison_role,
    )
    assert constants.T == 2
    assert constants.N == 16
    assert constants.K == 4
    assert constants.G == 2
    assert constants.V == 8
    limits = LOOKUP_REQUIRED(
        config.agent_limits_by_benchmark_and_model,
        task.benchmark_name,
        model_bundle.logical_name,
    )

    # Stage 1: iteration 0, 16 independent clean-environment rollouts.
    iter0_rollouts = ARRAY(size = constants.N)
    PARALLEL_FOR i in 0 .. constants.N - 1:
        iter0_rollouts[i] = RUN_ROLLOUT(
            task = task,
            model_bundle = model_bundle,
            scaffold = scaffold,
            limits = limits,
            iteration = 0,
            rollout_index = i,
            refinement_summaries = [],
            seed = DERIVE_SEED(
                config.experiment_seed, task.task_id, model_bundle.logical_name, 0, i
            ),
        )

    # Summaries are generated after rollout completion and before official grading results are
    # exposed to the method.
    iter0_summaries = ARRAY(size = constants.N)
    PARALLEL_FOR i in 0 .. constants.N - 1:
        iter0_summaries[i] = SUMMARIZE_ROLLOUT(
            task,
            iter0_rollouts[i],
            model_bundle,
            summary_adapter,
            policies.summary_policy,
            policies.summary_schema_or_null,
        )
    iter0_candidates = MAKE_CANDIDATES(iter0_rollouts, iter0_summaries)

    # Stage 2: run one iteration-0 RTV instance 16 -> 8 -> 4 -> 2 -> 1. Freeze the exact
    # top-four population after round two for refinement; the last two rounds exist only for
    # the paper's iteration-0 RTV diagnostics and cannot change that immutable checkpoint.
    iter0_tournament, select_k_checkpoint = RUN_RTV(
        task, model_bundle, iter0_candidates,
        configured_G = constants.G,
        V = constants.V,
        target_survivors = 1,
        policies = policies,
        seed = DERIVE_SEED(
            config.experiment_seed, task.task_id, model_bundle.logical_name, "iteration-0-rtv"
        ),
        freeze_checkpoint_at = constants.K,
    )
    assert select_k_checkpoint.survivor_count == constants.K
    selected_iter0 = RESOLVE_CANDIDATES(select_k_checkpoint.survivor_candidate_ids)
    refinement_summaries = [SUMMARY_FOR_ROLLOUT(c.rollout_id) for c in selected_iter0]

    # Stage 3: iteration 1, 16 new clean-environment rollouts. All receive the same selected four
    # summaries, but have separate model sampling seeds and separate environments.
    iter1_rollouts = ARRAY(size = constants.N)
    PARALLEL_FOR i in 0 .. constants.N - 1:
        iter1_rollouts[i] = RUN_ROLLOUT(
            task = task,
            model_bundle = model_bundle,
            scaffold = scaffold,
            limits = limits,
            iteration = 1,
            rollout_index = i,
            refinement_summaries = refinement_summaries,
            seed = DERIVE_SEED(
                config.experiment_seed, task.task_id, model_bundle.logical_name, 1, i
            ),
        )

    iter1_summaries = ARRAY(size = constants.N)
    PARALLEL_FOR i in 0 .. constants.N - 1:
        iter1_summaries[i] = SUMMARIZE_ROLLOUT(
            task,
            iter1_rollouts[i],
            model_bundle,
            summary_adapter,
            policies.summary_policy,
            policies.summary_schema_or_null,
        )
    iter1_candidates = MAKE_CANDIDATES(iter1_rollouts, iter1_summaries)

    # Stage 4: final RTV 16 -> 8 -> 4 -> 2 -> 1.
    final_tournament, unused_checkpoint = RUN_RTV(
        task, model_bundle, iter1_candidates,
        configured_G = constants.G,
        V = constants.V,
        target_survivors = 1,
        policies = policies,
        seed = DERIVE_SEED(
            config.experiment_seed, task.task_id, model_bundle.logical_name, "final"
        ),
    )
    assert unused_checkpoint is null
    final_candidate = RESOLVE_CANDIDATE(final_tournament.survivor_candidate_ids[0])
    final_rollout = RESOLVE_ROLLOUT(final_candidate.rollout_id)

    # The output is the surviving rollout's actual patch/artifacts/environment, not the judge's
    # prose and not a synthesized merge of several workspaces.
    result = TaskRun(
        task_id = task.task_id,
        model_logical_name = model_bundle.logical_name,
        iter0_rollout_ids = IDS(iter0_rollouts),
        iter0_summary_ids = IDS(iter0_summaries),
        iter0_tournament_id = iter0_tournament.tournament_id,
        select_k_checkpoint_id = select_k_checkpoint.checkpoint_id,
        refinement_summary_ids = IDS(refinement_summaries),
        iter1_rollout_ids = IDS(iter1_rollouts),
        iter1_summary_ids = IDS(iter1_summaries),
        final_tournament_id = final_tournament.tournament_id,
        final_rollout_id = final_rollout.run_id,
        output_patch = final_rollout.final_patch,
        output_artifacts = final_rollout.final_artifact_manifest,
        output_environment_snapshot = final_rollout.final_environment_snapshot,
    )
    PERSIST(result)
    return result
```

Run this independently for every `(benchmark, model, task)` combination. Never mix summaries or
votes across tasks or models.

## 9. Prompt templates required for a faithful implementation

The paper publishes example outputs but not the exact prompts. Keep all reconstructed templates
versioned and save the exact rendered prompt for every call.

At manifest construction, copy the scaffold's exact action prompt into `action_role.prompt_text`,
set `summary_role.prompt_text` to the chosen version of Section 9.1, and set
`comparison_role.prompt_text` to the chosen version of Section 9.3. The scaffold adapter inserts
the Section 9.2 refinement text into each refined rollout's user context. Store every rendered
prompt hash; `CALL_ROLE_WITH_RECORDED_RETRIES` sends the role prompt as the system/instruction
message exactly once.

### 9.1 Reconstructed summary prompt

```text
You are given an agentic coding task and one attempt serialized under the recorded summary-input
policy. Produce one structured summary. If the configured protocol supplies a JSON Schema,
produce one valid JSON object that conforms to it; otherwise follow the versioned JSON-object
contract. Schema enforcement is optional, but the summary record always stores one JSON object.

Report evidence, not just the agent's claims. Distinguish:
- commands with captured responses from commands merely proposed or issued without a response;
- successful outputs from errors;
- errors fixed later from errors still present at submission;
- files that were proved to exist from files the agent only claimed to create;
- tests run after the final edit from tests run before it;
- task requirements verified from requirements not checked;
- runtime dependencies and external files from development-only dependencies.

Preserve exact paths, functions, command text, return codes, decisive output, before/after code,
unresolved issues, and uncertainty. Do not infer official hidden-test success. Do not include any
official grader result. Output JSON only.
```

### 9.2 Reconstructed refinement preamble/postamble

```text
PREAMBLE:
You are starting a new independent attempt in a freshly initialized environment. Below are
structured summaries of K prior attempts on the same task. The summaries may describe successes,
failures, partial progress, conflicting diagnoses, or unverified claims. Use them as evidence,
not as ground truth.

POSTAMBLE:
Synthesize common findings, retain useful diversity, reconcile conflicts using concrete evidence,
avoid repeated dead ends, and verify the resulting solution in this fresh environment. You do not
have prior files or patches unless you recreate them yourself.
```

### 9.3 Reconstructed comparison prompt

```text
Compare the numbered candidate summaries for the original task. You cannot run hidden tests and
must not assume that a candidate passed. Rank candidates from their recorded code/artifacts,
commands, outputs, error state, and verification coverage.

Follow the benchmark-specific rubric in Section 6.1. Explain the decisive evidence. End with one
line in exactly this format, where N is one displayed candidate number:

Final verdict: Solution N
```

## 10. Benchmark adapters and outcome isolation

### 10.1 SWE-Bench Verified

```text
function EVALUATE_SWE_ROLLOUT(task, rollout):
    # Evaluation occurs in a grader clone/snapshot, never in a future judge/refinement context.
    grader_env = RESTORE_CLEAN_TASK_ENV(task)
    APPLY_PATCH(grader_env, rollout.final_patch)
    result = RUN_OFFICIAL_SWE_BENCH_GRADER(grader_env, task.official_grader_config)
    evaluation = RolloutEvaluation(
        rollout_id = rollout.run_id,
        grader_revision = OFFICIAL_GRADER_REVISION,
        binary_outcome = result.resolved,
        score_details = result,
    )
    PERSIST(evaluation)
    return evaluation
```

Use the full 500-task SWE-Bench Verified test set and the bash-only mini-SWE-agent scaffold. Record
the exact dataset revision, repository base commit per task, harness commit, image digest, and
grader version.

### 10.2 Terminal-Bench v2.0

```text
function EVALUATE_TERMINAL_ROLLOUT(task, rollout):
    grader_env = RESTORE_SNAPSHOT(rollout.final_environment_snapshot)
    result = RUN_OFFICIAL_TERMINAL_BENCH_GRADER(grader_env, task.official_grader_config)
    evaluation = RolloutEvaluation(
        rollout_id = rollout.run_id,
        grader_revision = OFFICIAL_GRADER_REVISION,
        binary_outcome = result.passed,
        score_details = result,
    )
    PERSIST(evaluation)
    return evaluation
```

Use the Terminus 1 scaffold on the exact 88-task subset. Record the omitted task ID; the paper does
not state it.

### 10.3 Leakage-safe sequencing

`RECONSTRUCTED leakage-safe implementation`: the paper requires outcome-free selection but does
not state when grading ran. For maximum assurance, run the experiment in two phases:

```text
PHASE A:
    produce all rollouts, summaries, select-K decisions, refined rollouts, and final decisions
    freeze and hash all records

PHASE B:
    run official graders for every iteration-0 and iteration-1 rollout
    append one immutable RolloutEvaluation per rollout ID
    compute metrics and plots
```

If grading must happen earlier for operational reasons, keep results in an access-controlled store
that the generation, summary, refinement, and comparison workers cannot read.

## 11. Metrics

Let `y[t, i, q]` be the binary outcome for task `q`, iteration `t`, rollout `i`.

```text
function AVERAGE_PASS_AT_1(candidate_sets_by_task):
    # Called "average pass@1" in the paper: mean binary reward over all candidates and tasks.
    values = []
    for task_candidates in candidate_sets_by_task:
        values.extend([OUTCOME(c) for c in task_candidates])
    return 100 * MEAN(values)

function PASS_AT_N(candidate_sets_by_task):
    # Fraction of tasks with at least one passing remaining candidate.
    return 100 * MEAN([
        ANY(OUTCOME(c) for c in task_candidates)
        for task_candidates in candidate_sets_by_task
    ])

function MIXED_TASK_COUNT(candidate_sets_by_task):
    return COUNT(task_candidates where
        ANY(OUTCOME(c) == true) and ANY(OUTCOME(c) == false))

function STAGE_METRICS(all_task_runs):
    return {
        ITER_0: AVERAGE_PASS_AT_1([
            RESOLVE_ROLLOUTS(run.iter0_rollout_ids) for run in all_task_runs
        ]),
        SELECT_K: AVERAGE_PASS_AT_1([
            RESOLVE_ROLLOUTS_FOR_CANDIDATES(
                RESOLVE_TOURNAMENT_CHECKPOINT(
                    run.select_k_checkpoint_id
                ).survivor_candidate_ids
            )
            for run in all_task_runs
        ]),
        ITER_1: AVERAGE_PASS_AT_1([
            RESOLVE_ROLLOUTS(run.iter1_rollout_ids) for run in all_task_runs
        ]),
        FINAL: AVERAGE_PASS_AT_1([
            [RESOLVE_ROLLOUT(run.final_rollout_id)] for run in all_task_runs
        ]),
    }
```

For these binary benchmarks, a population's pass@N is the upper bound on post-selection pass@1:
an oracle selector reaches it by retaining a successful rollout whenever one exists.

### 11.1 RTV round dynamics

At tournament round `r`, compute average pass@1 over every remaining candidate and pass@N over the
set of remaining candidates for each task. Average pass@1 should usually rise; pass@N can only stay
the same or fall because eliminating a successful candidate can remove the last success for a task.

```text
function RTV_ROUND_CURVES(tournaments):
    population_counts = [len(t.population_ids_by_round) for t in tournaments]
    assert ALL(count == population_counts[0] for count in population_counts)
    population_count = population_counts[0]
    for round_index in 0 .. population_count - 1:
        populations = [
            RESOLVE_ROLLOUTS_FOR_CANDIDATES(t.population_ids_by_round[round_index])
            for t in tournaments
        ]
        emit(round_index,
             average_pass_at_1 = AVERAGE_PASS_AT_1(populations),
             pass_at_n = PASS_AT_N(populations))
```

### 11.2 Groupwise comparison accuracy

Only score groups that contain at least one passing and one failing input rollout. A decision is
correct when its selected rollout passes.

```text
function GROUPWISE_JUDGE_ACCURACY(tournaments):
    rows = []
    for tournament in tournaments:
        for round in tournament.rounds:
            for decision in round:
                outcomes = [OUTCOME(id) for id in decision.input_candidate_ids]
                if ANY(outcomes) and not ALL(outcomes):
                    rows.append({
                        benchmark: BENCHMARK_FOR(tournament.task_id),
                        model: tournament.model_id,
                        iteration: tournament.rollout_iteration,
                        round_index: decision.round_index,
                        correct: OUTCOME(decision.selected_candidate_id),
                    })
    return {
        per_round: GROUP_MEAN(
            rows, by = [benchmark, model, iteration, round_index]
        ),
        pooled_average: GROUP_MEAN(
            rows, by = [benchmark, model, iteration]
        ),
    }
```

Do not compare these accuracies as controlled judge benchmarks across models: each model judges a
different pool of its own trajectories and summaries.

### 11.3 Sequential pass-count transition matrix

```text
function PASS_COUNT_TRANSITION_MATRIX(task_runs, N = 16):
    matrix = ZEROS(rows = N + 1, columns = N + 1)
    for run in task_runs:
        p0 = SUM(OUTCOME(r) for r in RESOLVE_ROLLOUTS(run.iter0_rollout_ids))
        p1 = SUM(OUTCOME(r) for r in RESOLVE_ROLLOUTS(run.iter1_rollout_ids))
        matrix[p0, p1] += 1
    return matrix
```

Rows are iteration-0 pass counts and columns are iteration-1 pass counts. Cells above the diagonal
are improvements; cells below are regressions.

### 11.4 Context-quality analysis

```text
function CONTEXT_QUALITY_BUCKETS(task_runs, N = 16, K = 4):
    buckets = {p: [] for p in 0 .. K}
    for run in task_runs:
        for refined_rollout in RESOLVE_ROLLOUTS(run.iter1_rollout_ids):
            source_ids = refined_rollout.refinement_summary_ids
            assert len(source_ids) == K
            context_pass_count = SUM(
                OUTCOME(ROLLOUT_FOR_SUMMARY(summary_id)) for summary_id in source_ids
            )
            buckets[context_pass_count].append(OUTCOME(refined_rollout))

    return {
        p: {
            context_count: len(buckets[p]),
            average_task_equivalents: len(buckets[p]) / N,
            iteration1_pass_at_1:
                null if len(buckets[p]) == 0 else 100 * MEAN(buckets[p]),
            binary_outcomes: buckets[p],
        }
        for p in 0 .. K
    }
```

For select-K, each task's 16 refined rollouts share one context, so dividing context count by 16
produces an integer task count. For random-K, each refined rollout can have a different context;
the same division produces the paper's fractional “average tasks” values.

### 11.5 Step efficiency

```text
function STEP_STATISTICS(task_runs):
    rows = FLATTEN([
        {
            benchmark: BENCHMARK_FOR(run.task_id),
            model: run.model_logical_name,
            iteration: r.iteration,
            passed: OUTCOME(r),
            steps: len(r.steps),
        }
        for run in task_runs
        for r in RESOLVE_ROLLOUTS(run.iter0_rollout_ids + run.iter1_rollout_ids)
    ])
    results = {}
    for benchmark, model, iteration in DISTINCT_KEYS(rows):
        group = FILTER(rows, matching = [benchmark, model, iteration])
        results[benchmark, model, iteration] = {
            all: MEAN([row.steps for row in group]),
            pass: MEAN([row.steps for row in group if row.passed]),
            fail: MEAN([row.steps for row in group if not row.passed]),
        }
    return results
```

The paper counts agent steps, not shell commands. One action containing several commands is one
step.

### 11.6 New-solution discovery

```text
function NEW_SOLUTION_TASKS(task_runs):
    return [
        run.task_id for run in task_runs
        if not ANY(OUTCOME(r) for r in RESOLVE_ROLLOUTS(run.iter0_rollout_ids))
        and ANY(OUTCOME(r) for r in RESOLVE_ROLLOUTS(run.iter1_rollout_ids))
    ]
```

### 11.7 Pairwise model capability matrix

For models `M_i, M_j`, count tasks for which `M_i` has at least one iteration-0 success and `M_j`
has zero iteration-0 successes.

```text
matrix[i, j] = COUNT(task q where
    ANY(y[M_i, iteration=0, :, q]) and
    not ANY(y[M_j, iteration=0, :, q]))
```

For the cross-model new-solution analysis:

```text
discovery_tasks = UNION(NEW_SOLUTION_TASKS(runs_for_model) for model in MODELS)
for task_id in discovery_tasks:
    for model in MODELS:
        already_solved_iter0[task_id, model] = ANY(
            OUTCOME(r) for r in ITER0_ROLLOUTS(task_id, model)
        )
```

## 12. Complete experiment matrix

### 12.1 Parallel aggregation: summaries versus raw trajectories

Run on both benchmarks with Claude-4.5-Sonnet and Gemini-3-Flash.

The paper does not disclose the held-fixed group size, vote count, or role call settings for this
figure. Require them in the ablation manifest:

```text
SUMMARY_VS_RAW_HELD_FIXED = {
    G: REQUIRED_CONFIG,
    V: REQUIRED_CONFIG,
    summary_and_comparison_call_settings: REQUIRED_CONFIG,
}
```

```text
for representation in [STRUCTURED_SUMMARY, FULL_ROLLOUT_TRACE]:
    run identical N=16 tournaments
    if STRUCTURED_SUMMARY:
        Candidate.display_payload = canonical summary JSON
    else:
        Candidate.display_payload = full raw rollout trajectory
    compare average pass@1 after every round and final pass@1
```

Keep every other parameter, task set, initial rollout pool, pairing, vote seed, and judge model
fixed. The paper reports that structured summaries win consistently, especially in later rounds.

### 12.2 RTV group-size ablation

Use Gemini-3-Flash on both benchmarks, `N=16`, and:

```text
GROUP_SIZE_ABLATION_HELD_FIXED = {
    V: REQUIRED_CONFIG,  # not disclosed in the group-size passage/caption
    summary_and_comparison_call_settings: REQUIRED_CONFIG,
}
```

```text
G in [16, 8, 4, 2]

G=16: 16 -> 1
G=8:  16 -> 2 -> 1
G=4:  16 -> 4 -> 1
G=2:  16 -> 8 -> 4 -> 2 -> 1
```

Use the same rollout pool. The paper reports `G=2` as best.

### 12.3 RTV vote-count ablation

Use Gemini-3-Flash on both benchmarks, `N=16`, `G=2`, and:

```text
V in [1, 2, 4, 8, 16]
```

Use identical candidates and pairings. The paper reports gains with more votes and diminishing
returns beginning around `V=8`.
Keep summary/comparison role settings fixed at explicitly recorded `REQUIRED_CONFIG` values; the
paper does not publish those API settings.

### 12.4 Standalone RTV

For all five models and both benchmarks, run `N=16, G=2, V=8` on iteration-0 summaries until one
candidate remains. Compare initial average pass@1 to final RTV pass@1.

Textual check reported by the paper for Claude-4.5-Sonnet:

```text
SWE-Bench Verified: 67.4 -> 73.6
Terminal-Bench v2.0: 40.6 -> 54.6
```

### 12.5 Sequential refinement ablation

Use 100 randomly sampled SWE-Bench Verified tasks, Claude-4.5-Sonnet and Gemini-3.1-Pro,
`N=16`, `K=4`. Record the exact sampled task IDs and random seed. Run:

1. single-rollout refinement;
2. random-K refinement;
3. select-K refinement using RTV.

Use the same iteration-0 rollouts for all three variants. Compare average iteration-0 and
iteration-1 pass@1, pass-count distributions, and iteration-1 success stratified by the number of
passing summaries in each refinement context.
Although original PDR chooses one final rollout, this paper estimates each pure-PDR variant by
averaging the binary scores of all 16 iteration-1 rollouts; do not evaluate only one chosen answer.

The paper's check values are:

| Model | Single iter 0 -> 1 | Random-K iter 0 -> 1 | Select-K iter 0 -> 1 |
|---|---:|---:|---:|
| Claude-4.5-Sonnet | 69.87 -> 70.87 | 69.87 -> 75.06 | 69.87 -> 78.06 |
| Gemini-3.1-Pro | 72.69 -> 73.75 | 72.69 -> 76.94 | 72.69 -> 79.25 |

Context-quality ablation checks are `iteration-1 pass@1 (average task equivalents)`. A dash means
the bucket had no contexts:

| Model/method | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 |
|---|---:|---:|---:|---:|---:|
| Sonnet random-K | 2.5 (17.8) | 47.1 (6.5) | 80.4 (6.7) | 88.8 (16.2) | 98.1 (52.8) |
| Sonnet select-K | 1.2 (16) | 12.5 (3) | - (0) | 90.5 (19) | 97.3 (62) |
| Gemini Pro random-K | 1.9 (16.2) | 50.0 (6.4) | 66.4 (7.2) | 90.6 (10.6) | 99.1 (58.5) |
| Gemini Pro select-K | 2.2 (17) | 40.6 (2) | 52.1 (3) | 81.2 (7) | 99.7 (71) |

For Sonnet, the number of 100 sampled tasks with 16/16 passing iteration-1 rollouts is 40 under
single-rollout refinement and 51 under random-K refinement.

### 12.6 Main experiment

For each model and benchmark task, run the controller in Section 8. Report stage average pass@1:

| Model | SWE iter 0 | SWE select-K | SWE iter 1 | SWE final | Terminal iter 0 | Terminal select-K | Terminal iter 1 | Terminal final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude-4.5-Opus | 70.94 | 75.00 | 76.04 | 77.60 | 46.95 | 54.26 | 52.49 | 59.09 |
| Gemini-3.1-Pro | 72.25 | 75.30 | 76.16 | 76.60 | 52.49 | 59.66 | 56.89 | 64.77 |
| Claude-4.5-Sonnet | 67.41 | 72.60 | 74.01 | 75.60 | 40.62 | 50.85 | 50.00 | 56.82 |
| Gemini-3-Flash | 70.79 | 73.55 | 74.28 | 76.00 | 37.93 | 45.45 | 43.68 | 48.86 |
| GPT-5-0825 | 61.41 | 65.25 | 67.73 | 69.80 | 31.32 | 35.23 | 35.30 | 38.64 |

Treat these values as result checks, not algorithm inputs.

### 12.7 Required analysis outputs

Generate all of the following from stored per-rollout outcomes and tournament logs:

1. stage metrics: iteration 0, select-K, iteration 1, final;
2. per-iteration average pass@1, pass@16, and mixed-task count;
3. average steps for all, passing, and failing rollouts by iteration/model/benchmark;
4. iteration-0 to iteration-1 pass-count transition matrices;
5. pass-count distributions for both iterations;
6. context-quality buckets from zero to four passing selected summaries;
7. average pass@1 and pass@N after every RTV round for both iterations;
8. groupwise comparison accuracy for mixed groups by model, benchmark, iteration, and round;
9. pairwise model capability matrices;
10. tasks with zero iteration-0 successes and at least one iteration-1 success;
11. qualitative traces showing how refined rollouts reuse consensus, avoid repeated failures, and
    resolve disagreements among prior summaries.

### 12.8 Additional numeric checks from the paper and appendices

Main rollout statistics use tuples `(average pass@1, pass@16, mixed-task count)`:

```text
SWE-Bench Verified             iteration 0             iteration 1
Claude-4.5-Opus                (70.94, 85.40, 218)      (76.04, 81.20, 56)
Gemini-3.1-Pro                 (72.25, 86.00, 200)      (76.16, 82.00, 59)
Claude-4.5-Sonnet             (67.41, 83.40, 259)      (74.01, 79.20, 71)
Gemini-3-Flash                 (70.79, 84.00, 251)      (74.28, 79.80, 179)
GPT-5-0825                     (61.41, 79.00, 257)      (67.73, 73.40, 96)

Terminal-Bench v2.0            iteration 0             iteration 1
Claude-4.5-Opus                (46.95, 70.45, 36)       (52.49, 65.91, 22)
Gemini-3.1-Pro                 (52.49, 76.14, 35)       (56.89, 72.73, 23)
Claude-4.5-Sonnet             (40.62, 67.05, 38)       (50.00, 62.50, 19)
Gemini-3-Flash                 (37.93, 60.23, 37)       (43.68, 56.82, 18)
GPT-5-0825                     (31.32, 51.14, 30)       (35.30, 43.18, 10)
```

Average step counts use tuples `(all, passing, failing)`:

```text
Model                  SWE iteration 0 -> 1                  Terminal iteration 0 -> 1
Opus       (41.23,33.83,59.30) -> (14.31,13.16,17.97)  (24.43,24.66,24.23) -> (12.14,10.96,13.45)
Pro        (35.56,33.88,39.92) -> (17.95,17.05,20.82)  (21.57,17.47,26.09) -> (10.95, 9.20,13.25)
Sonnet     (49.24,46.13,55.67) -> (25.02,24.39,26.84)  (21.74,19.47,23.30) -> ( 7.78, 6.29, 9.26)
Flash      (51.10,48.39,57.65) -> (28.80,27.37,32.94)  (16.01,15.68,16.22) -> ( 7.80, 6.16, 9.07)
```

Main select-K context checks are `iteration-1 pass@1 (task count)` for context pass counts
`[0/4, 1/4, 2/4, 3/4, 4/4]`:

```text
Opus SWE:      0.1(81), 33.4(29), 55.5(25), 85.4(39), 99.2(326)
Opus Terminal: 1.8(31), 31.2(4),  43.0(8),  78.5(9),  94.1(36)

Pro SWE:       0.6(87), 36.9(22), 38.4(22), 87.0(36), 99.8(333)
Pro Terminal:  3.8(23), 37.5(7),  45.8(12), 57.5(5),  93.1(41)

Sonnet SWE:       1.7(94), 18.8(16), 65.4(28), 88.1(68), 99.7(294)
Sonnet Terminal:  3.0(33), 30.2(6),  58.0(7),  76.4(9),  91.7(33)

Flash SWE:      0.0(93), 34.7(18), 73.1(20), 88.1(63), 96.4(306)
Flash Terminal: 1.0(37), 23.2(7),  68.1(9),  60.0(5),  91.0(30)
```

Groupwise judge accuracy checks use `R0, R1, R2, R3, pooled Avg`. The average is pooled over all
qualifying mixed groups; it is not the arithmetic mean of the four round percentages.

```text
Iter 0                 SWE                              Terminal
Opus       69.1,68.3,60.6,55.6,67.0       81.4,78.9,67.9,64.3,77.9
Pro        66.9,64.5,51.1,65.1,64.5       84.2,81.4,72.2,62.5,80.7
Sonnet     70.6,66.2,53.3,55.6,66.6       80.5,87.0,78.6,62.5,81.7
Flash      66.3,55.6,53.3,54.5,61.3       85.1,79.4,80.0,73.7,82.3

Iter 1                 SWE                              Terminal
Opus       54.4,62.9,60.0,71.4,58.2       75.4,74.3,84.2,83.3,77.2
Pro        43.4,47.4,65.8,60.0,48.3       77.4,68.6,85.7,88.9,76.7
Sonnet     63.2,57.5,59.0,60.0,60.9       78.9,77.4,76.2,77.8,78.0
Flash      67.3,51.5,55.8,50.0,62.1       77.9,71.1,72.4,90.9,75.8
```

### 12.9 Analysis coverage matrix

```text
All five models:
  main stage results; main rollout statistics; standalone RTV; pairwise model comparisons

Opus, Pro, Sonnet, and Flash only:
  step statistics; pass-count transitions; main context-quality analysis;
  pass-count distributions; RTV round dynamics; groupwise judge accuracy

Sonnet and Pro only, on the sampled 100 SWE tasks:
  single-rollout/random-K/select-K sequential ablation

Sonnet and Flash only:
  structured-summary versus raw-trajectory ablation

Flash only:
  G and V RTV parameter ablations
```

Several plotted series are present only as curves or heatmaps in the supplied figure PDFs, not as
machine-readable values or TeX tables. Their exact numeric points cannot be reconstructed from the
source text alone. Treat these files as visual checks; digitize the source assets under a recorded
method if numeric comparison is required:

```text
rtv_summary_vs_rollouts_colm_v1.pdf
rtv_parameter_search_colm_v1.pdf
rtv_main_results_colm_v1.pdf
pdr_pass_count_random_k.pdf
pdr_rtv_confusion_matrices.pdf
pdr_rtv_pass_count_distributions_swe_bench.pdf
pdr_rtv_pass_count_distributions_terminal_bench.pdf
pdr_rtv_parallel_analysis_swe_bench.pdf
pdr_rtv_parallel_analysis_terminal_bench.pdf
pdr_top_k_pass_rate_analysis.pdf
rollout_matchups_iter0_colm_v1.pdf
```

## 13. Expected appendix checks

The paper reports these new-solution tasks. Use them as end-to-end data-integrity checks, not as
inputs to generation or selection.

```text
SWE-Bench Verified:
  Claude-4.5-Opus:   django_django-11951
  Gemini-3.1-Pro:    sphinx-doc_sphinx-9602
  Claude-4.5-Sonnet: django_django-13964, scikit-learn_scikit-learn-25102
  Gemini-3-Flash:    none
  GPT-5-0825:        pydata_xarray-4687

Terminal-Bench v2.0:
  Claude-4.5-Opus:
    caffe-cifar-10, chess-best-move, gpt2-codegolf, nginx-request-logging,
    vulnerable-secret
  Gemini-3.1-Pro:
    configure-git-webserver, gcode-to-text, git-leak-recovery,
    large-scale-text-editing, openssl-selfsigned-cert
  Claude-4.5-Sonnet:
    mcmc-sampling-stan, sparql-university
  Gemini-3-Flash:
    mcmc-sampling-stan, regex-chess
  GPT-5-0825:
    mcmc-sampling-stan, schemelike-metacircular-eval, vulnerable-secret
```

The paper then checks whether each model had already solved every discovery task in iteration 0.
The table captions incorrectly describe the checkmark as self-improvement; the table values and
surrounding prose show that it means “this model had at least one iteration-0 success.” The
corrected matrices below use `Y` for that meaning:

```text
SWE discovery task                    Opus  Pro  Sonnet  Flash  GPT-5
django_django-11951                    N     N      Y       Y      Y
sphinx-doc_sphinx-9602                 N     N      Y       Y      N
django_django-13964                    Y     Y      N       Y      Y
scikit-learn_scikit-learn-25102        Y     N      Y       N      Y
pydata_xarray-4687                     Y     Y      Y       Y      N

Terminal discovery task               Opus  Pro  Sonnet  Flash  GPT-5
caffe-cifar-10                          N     Y      N       Y      Y
chess-best-move                         N     Y      Y       Y      Y
configure-git-webserver                 N     N      Y       N      Y
gcode-to-text                           Y     N      N       N      N
git-leak-recovery                       N     N      Y       N      N
gpt2-codegolf                           N     Y      N       N      N
large-scale-text-editing                N     N      N       N      N
mcmc-sampling-stan                      N     Y      N       N      N
nginx-request-logging                   N     Y      N       N      N
openssl-selfsigned-cert                 Y     N      N       N      N
regex-chess                             N     N      Y       N      N
schemelike-metacircular-eval            Y     Y      Y       Y      N
sparql-university                       Y     Y      N       Y      N
vulnerable-secret                       N     Y      N       N      N
```

The qualitative examples emphasize four refinement behaviors that the trace viewer should make
auditable:

1. consensus synthesis across multiple summaries;
2. direct reuse of a precise diagnosis, file, line, command, or test;
3. explicit reconciliation of conflicting approaches using recorded evidence;
4. avoidance of prior environment/setup failures in the fresh rollout.

Exact appendix example inventory:

```text
PDR+RTV qualitative excerpts:
  Opus / SWE / django_django-13033
  Gemini Pro / SWE / sympy_sympy-17318
  Opus / Terminal / sparql-university
  Sonnet / Terminal / sqlite-db-truncate

Initial trajectories:
  Gemini Pro / SWE / sympy__sympy-17630
  Opus / Terminal / openssl-selfsigned-cert

Refined trajectories:
  Opus / Terminal / gpt2-codegolf
  Gemini Pro / Terminal / large-scale-text-editing

Structured summaries:
  Gemini Pro / SWE / sympy__sympy-17630
  Opus / Terminal / openssl-selfsigned-cert

Group comparisons:
  Gemini Pro / SWE / django__django-15973
  Opus / Terminal / video-processing
```

Task-specific trace checks include direct reuse of Django line 730 and the `pieces[-1]` fix,
preinstallation of `asgiref`, `pytz`, and `sqlparse`, reuse of the SymPy call-chain diagnosis and a
prior failing test, selection of a two-file SymPy fix, separation of the SPARQL EU and student-count
conditions, and selection of the successful SQLite serial-type interpretation before building the
B-tree leaf-page parser.

## 14. Resume, concurrency, and failure handling

The paper reports infrastructure-related Gemini-3.1-Pro API failures during final RTV, but does not
specify recovery semantics. Failure policy can change results, so implement it explicitly:

```text
function CALL_ROLE_WITH_RECORDED_RETRIES(role, messages, tools = null, seed):
    request_messages = [
        {role: "system", content: role.prompt_text},
        *messages,
    ]
    assert SHA256(role.prompt_text) == role.prompt_sha256

    for attempt in 1 .. role.retry_policy.max_attempts:
        response = PROVIDER_CALL(
            role = role,
            messages = request_messages,
            tools = tools,
            seed = DERIVE_SEED(seed, "provider-attempt", attempt),
        )
        record request ID, attempt, status, latency, usage, and provider error
        if response is valid:
            return response
        if response.error_kind not in role.retry_policy.retryable_error_kinds:
            break
        wait DETERMINISTIC_BACKOFF(
            attempt, seed, role.retry_policy.deterministic_backoff_policy
        )
    return FAILED_RESPONSE(all_attempts)

function HANDLE_EXHAUSTED_MODEL_CALL(call_kind, policy, attempt, max_content_attempts,
                                     failed_response):
    if policy == RETRY_CONTENT_ATTEMPT and attempt < max_content_attempts:
        RECORD_REPLACEMENT_LINK(failed_response, next_content_attempt = attempt + 1)
        return RETRY_CONTENT_ATTEMPT
    raise RequiredModelCallFailed(call_kind, failed_response.attempt_records)
```

Required operational rules:

- Use an idempotency key derived from experiment/task/model/iteration/rollout/step/call kind.
- Resume from immutable records; never regenerate a successful call during resume.
- Never replace a failed rollout or vote with an unrecorded extra sample.
- If replacement is permitted by the configured protocol, label the replacement and include it in
  denominators exactly once.
- Keep rollout environments isolated even when executing them concurrently.
- Rate-limit at provider/model level without changing task membership.
- Save raw provider responses before parsing.
- Freeze pairing and display order before launching parallel judge calls.
- Refuse to compute final metrics while required records are missing or duplicated.
- If calls are cached, key the cache by task ID, lossless trajectory/input hash, prompt and schema
  version, exact model identity, and the complete role generation configuration.

## 15. Compute accounting

Per task and model in the main `T=2` experiment:

```text
agent trajectories:           16 iteration 0 + 16 iteration 1 = 32
summary calls:                16 iteration 0 + 16 iteration 1 = 32

select-K RTV group decisions: 8 + 4 = 12
select-K judge calls:         12 * V=8 = 96

final RTV group decisions:    8 + 4 + 2 + 1 = 15
final judge calls:            15 * V=8 = 120

comparison calls that affect method output: 216
iteration-0 diagnostic continuation:     24
total comparison calls executed:        240
total summary + comparison calls:       272
```

Each trajectory contains many action-generation model calls, one per agent step, so do not report
the 32 trajectories as 32 ordinary language-model requests. Compute total provider calls, input
tokens, output tokens, wall time, and cost by call kind: action, summary, comparison, and retry.

The 24 diagnostic calls are the last two rounds of the same frozen-checkpoint iteration-0
tournament. They are required for the paper's roundwise iteration-0 analysis but cannot change
the four summaries already recorded for refinement.

## 16. Deterministic validation and unit tests

Before spending model compute, test the controller with scripted fake models and fake graders.

```text
test_main_population_sizes:
    assert the frozen select-K checkpoint populations are [16, 8, 4]
    assert its continued iteration-0 diagnostic tournament populations are [16, 8, 4, 2, 1]
    assert final populations are [16, 8, 4, 2, 1]

test_refinement_checkpoint_is_immutable:
    freeze the iteration-0 top-four candidate IDs and hash
    continue the same tournament to two and one survivors
    assert the checkpoint IDs/hash and every refined rollout's four summary IDs are unchanged

test_group_size_ablation_schedules:
    assert configured G=16 uses effective sizes [16]
    assert configured G=8 uses effective sizes [8, 2]
    assert configured G=4 uses effective sizes [4, 4]
    assert configured G=2 uses effective sizes [2, 2, 2, 2]

test_call_counts:
    assert decisions through the select-K checkpoint == 12
    assert votes through the select-K checkpoint == 96
    assert diagnostic continuation decisions == 3
    assert diagnostic continuation votes == 24
    assert final group decisions == 15
    assert final votes == 120

test_fresh_environments:
    create a unique file in every iteration-0 environment
    assert no iteration-1 environment contains any such file

test_same_selected_context_for_select_k:
    assert every iteration-1 rollout has exactly the same four summary IDs

test_random_k_is_per_rollout:
    assert every random-K context has K distinct IDs
    assert sampled ID sets are recorded for each rollout

test_no_grader_leakage:
    tag every object by provenance and authorize only task-visible inputs plus method records
    assert no grader-owned reward, score, verifier result, hidden test, grader trace, or
           evaluation-store object/reference reaches summary inputs, summaries, judge cards,
           comparison prompts, refinement contexts, pairing, ties, or fallbacks
    allow ordinary task-visible text to mention words such as "hidden tests"
    assert action, summary, comparison, pairing, tie-breaking, and invalid-vote fallback
           workers cannot read grader records or the evaluation store before method outputs freeze

test_same_model_for_all_roles:
    assert every rollout uses the configured logical model identity
    assert every summary and every RTV vote uses that same identity
    reject aliases that resolve to different provider model revisions

test_surviving_artifact_identity:
    assert final output patch/artifacts hash equals the selected rollout's stored output hash

test_verdict_parser:
    accept exactly one "Final verdict: Solution N" line in range
    reject missing, duplicate, conflicting, or out-of-range verdicts
    assert malformed, empty, ambiguous, and out-of-range verdicts never default or clamp
           to a displayed candidate

test_vote_mapping_after_display_permutation:
    assert displayed position maps back to the correct stable candidate ID

test_tie_is_explicit:
    force a 4-4 vote split and assert the configured tie policy is recorded

test_exact_vote_count:
    assert every non-bye group records exactly V valid or explicitly replaced votes
    assert faithful mode never stops voting after an early unflippable majority

test_metric_denominators:
    synthetic tasks with known outcomes reproduce average pass@1, pass@N, mixed counts,
    transition matrices, group accuracy, and context buckets

test_summary_example_shapes:
    parse both appendix-derived example JSON files without type loss
    if a configured schema is used, validate that both examples satisfy its union/normalization
    policy; do not claim the examples publish a normative schema

test_scaffold_adapters:
    parse and round-trip the appendix mini-SWE-agent and Terminus 1 trajectory examples

test_configuration_effects:
    every declared behavior-controlling field alters execution exactly as documented
    or is rejected as unsupported; provenance-only fields may be observational
```

## 17. Public code audit and paper/code differences

No official or author-linked implementation was public when this audit ran on August 17, 2026.
The paper source has no repository or code-availability statement, and searches of the authors',
Meta's, and GitHub's public pages found no matching official repository. Two later repositories
explicitly describe themselves as independent reproductions:

- [genji970/facebook-paper_harness-inference-scale-agent at
  `e4d81a67198cd55798330ac3e6ce261a131e1852`](https://github.com/genji970/facebook-paper_harness-inference-scale-agent/tree/e4d81a67198cd55798330ac3e6ce261a131e1852),
  dated May 3, 2026. Its README says it is unofficial, covers only Gemini plus SWE-Bench, and was
  written because no public paper implementation existed.
- [zy95-12/Agentic-Coding at
  `30442021b3f9d2ae065cdcd1b9bbdbf1e5c38740`](https://github.com/zy95-12/Agentic-Coding/tree/30442021b3f9d2ae065cdcd1b9bbdbf1e5c38740),
  dated July 10, 2026. Its README calls it an unaffiliated reproduction and says its runs are
  engineering checks rather than a reproduction of the headline results.

Neither repository can turn an unresolved field into `REPO` evidence. Do not copy their prompts,
fallbacks, runtime constants, or result claims into a paper-faithful configuration.

The `zy95-12` reproduction follows the broad `iteration 0 -> summarize -> select K -> iteration 1
-> summarize -> final RTV` sequence, but differs materially:

- it uses GLM and Terminus 2 instead of the paper's five models and Terminus 1;
- its controller's `--model` setting controls summarization/judging but is not passed to Harbor
  rollout generation, so generator and judge identities can diverge;
- it reads official verifier rewards, places them in summaries and judge cards, tells the judge to
  use them, ranks reward first in a fallback, and writes prior reward into refinement context;
- it summarizes rule-extracted evidence truncated to 45,000 characters rather than a disclosed
  paper serializer;
- it can stop voting when a majority cannot flip instead of collecting the equation's exact `V`
  votes;
- malformed verdicts can be clamped/defaulted to a candidate;
- it uses adjacent groups, singleton byes, and a lexicographic trial-ID tie-break, none of which is
  paper evidence;
- its GLM route, rollout temperature `0.2`, 128,000/8,192 token limits, Terminus 2 settings,
  concurrency, timeouts, and evidence cap are third-party choices, not missing paper constants.

The `genji970` reproduction also differs:

- generator and summary/judge model IDs are configured independently;
- summary and judge calls force temperature zero, so repeated identical prompts do not provide a
  meaningful independent-vote mechanism;
- the comparison parser takes the first integer and maps malformed/out-of-range output to the
  first candidate; tied counts also choose the first displayed candidate;
- it collapses mini-SWE-agent execution into one synthetic step plus an optional patch step,
  keeps only the last 12,000 stdout/stderr characters, and truncates that patch step to 4,000
  characters by default;
- it treats zero process exit plus any nonempty patch as success before official grading;
- its final record does not always save and hash the selected patch;
- its `num_iterations` option does not control execution, which is hard-coded to two iterations.

The `zy95-12` repository also contains paths that must not be mistaken for its main experiment:
`src/json_command_agent.py` is abandoned; `rtv_select.py::_select_group` is unused and calls
`_vote` without its required `debug_log`; and its reward-first semantic top-K context is
generated but replaced by `rtv_refinement_context.md` before iteration 1.

`genji970` grades only the selected final rollout. `zy95-12` verifies every Harbor trial but
exposes those rewards before summarization and selection. Neither implements a leakage-safe
freeze-then-grade phase or the paper's complete analyses. Across the two reproductions, omitted
study components include one benchmark or the other, four of five models, the complete ablation
matrix, transition matrices, roundwise pass@1/pass@N, mixed-group judge accuracy, and full
benchmark scale. The pseudocode in this document retains those paper requirements and explicitly
rejects reward leakage, early majority stopping, silent verdict defaults, and model-identity
drift.

## 18. Replication-critical details not disclosed by the paper

Numerically exact reproduction is impossible from the PDF alone until these values are recovered
from public code/configuration or the authors:

1. exact agent, summarization, comparison, and refinement prompts;
2. exact summary output schema, whether the API enforced it, and whether every task in a benchmark
   used the same schema;
3. malformed-summary parse, repair, retry, exclusion, and replacement policy;
4. exact model provider IDs and API revisions for the action, summary, and comparison roles;
5. temperature, top-p, token budgets, context management, stop settings, and seed behavior for
   each role;
6. exact mini-SWE-agent and Terminus 1 action protocols, prompts, and serializer revisions;
7. agent step, command, and wall-time limits;
8. summary-input serialization and over-context truncation/reduction policy;
9. pairing order between rounds and whether populations are shuffled;
10. candidate display-order randomization;
11. tie-breaking for even `V=8` vote splits;
12. invalid/malformed judge-output parsing and retry behavior;
13. model/API error retry and replacement policy;
14. experiment and ablation seeds, including the 100-task SWE-Bench sample;
15. exact benchmark, harness, grader, repository, and container revisions;
16. the one omitted Terminal-Bench task;
17. whether summaries include the original task, final patch, and artifact manifest separately
    from the trajectory;
18. whether selected summaries use a fixed concatenation order in every refined rollout;
19. exact command-output truncation presented to the agent and summarizer;
20. how unfinished/invalid rollouts enter summarization, tournaments, and metric denominators;
21. concurrency and provider rate-limit settings;
22. held-fixed group size, vote count, and role call settings for the summary-versus-raw and
    group-size parameter ablations.

The implementation must print this unresolved-field list at startup and refuse a run labeled
`exact_replication=true` while any field remains unresolved. A run may proceed as a
`conceptual_replication` when every reconstructed choice is explicit in the manifest.

## 19. End-to-end execution checklist

```text
1. Pin dataset, harness, grader, image, repository, model, and prompt revisions.
2. Freeze task lists, including the exact 88 Terminal-Bench tasks.
3. Freeze all seeds and controller policies.
4. For each benchmark/model/task, create 16 clean iteration-0 environments and run rollouts.
5. Generate 16 structured summaries with the same model; parse/validate them under the recorded
   summary protocol.
6. Run iteration-0 RTV 16 -> 8 -> 4 with 8 votes per pair; freeze the four survivors, then
   continue that same tournament 4 -> 2 -> 1 only for the paper's diagnostic analysis.
7. Create 16 new clean environments; give all 16 the same four selected summaries.
8. Run the 16 iteration-1 rollouts and generate 16 new summaries.
9. Run final RTV 16 -> 8 -> 4 -> 2 -> 1 with 8 votes per pair.
10. Return the selected rollout's actual patch/artifacts/environment.
11. Freeze all method outputs before exposing official grader results.
12. Grade all 32 rollouts per task so every stage and analysis can be computed.
13. Compute stage scores, pass@16, mixed counts, step statistics, transitions, context buckets,
    RTV round curves, mixed-group judge accuracy, model matchups, and new-solution tasks.
14. Run the summary/raw, G, V, standalone RTV, and three refinement ablations with fixed pools.
15. Compare to the paper's result checks without feeding those checks back into the method.
16. Publish the full manifest, task list, prompts, raw call logs, summaries, tournament records,
    patches/artifact manifests, grader versions, and metric-generation code.
```
