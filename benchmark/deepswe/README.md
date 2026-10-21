# DeepSWE v1.1

Use Linux x86_64, cgroup v2, Git, uv, and Docker Compose v2. Start Docker on the same host.

From the repository root, run Oracle:

```bash
uv run --script benchmark/deepswe/run.py --agent oracle
```

To run Oracle, then Codex:

```bash
export INFERENCE_HUB_KEY="..."
uv run --script benchmark/deepswe/run.py --agent all
```

Add `--task scc`, `--task wazero`, or `--task optique` to select a task.
Use `--agent codex` to run Codex only.
Codex uses GPT-6-Luna with high reasoning effort through NVIDIA Inference Hub. These runs call Hub directly.
The scripts download the DeepSWE revision and tool versions specified in `run.py`.

## Check memory

Each task and verifier has 2 CPUs, an 8 GiB RAM limit, and no swap.
The proxy has a separate 512 MiB RAM limit. Allow extra host RAM for Docker, the proxy, and host processes.

Set `RUN_DIR` to the directory that the script prints. Use a second terminal:

```bash
RUN_DIR=/absolute/path/to/run
watch -n 1 "uv run --script benchmark/deepswe/memory.py '$RUN_DIR'"
```

The monitor reads kernel peak RAM and out-of-memory (OOM) counters each second and before container removal.
The combined RAM estimate is `max(agent peak + proxy peak, verifier peak)`.
The agent and proxy stop before the verifier starts. Individual peaks can occur at different times.
Missing counters, swap use, OOM events, or a combined estimate of 8 GiB or more fail the memory check.

Read `summary.json` for scores and memory results. Incomplete runs and failed memory checks return exit code 1.
Task scores are separate from memory results. Image builds and host processes are outside the memory measurement.
Run records stay in `benchmark/tb_runs/deepswe/`. Git excludes these records. Keep them local; they can contain credentials.

## Recorded results

| Task | Oracle tests passed | Codex tests passed | Codex combined RAM estimate |
| --- | ---: | ---: | ---: |
| SCC | 317/317 | 317/317 | 4.270 GiB |
| Wazero | 80/80 | 79/80 | 0.432 GiB |
| Optique | 2070/2070 | 2069/2070 | 0.682 GiB |

These results come from one attempt per task and agent. All tests must pass for reward 1.
The original runs allowed swap and had no proxy RAM limit. The records show no swap use or OOM kills.
Oracle passed all three tasks with the new limits. Codex passed a startup check with these limits.
Repeat the Codex tasks to check memory during a full run.
A separate 64 MiB container test caused an OOM kill. The memory check correctly failed.
