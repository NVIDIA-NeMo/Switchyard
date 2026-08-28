# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker helpers for the task generation pipeline.

Builds task Docker images, runs F2P/P2P 2-run classification, and executes
score.py checks (NOP and oracle gates).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import tempfile
from pathlib import Path


def image_tag_for_task(task_dir: str) -> str:
    return f"craft-{Path(task_dir).name}:latest".lower()


async def cleanup_task_images(task_dir: str) -> None:
    """Remove docker images produced for this task.

    Matches two families of tags keyed off the task_dir basename:
    - `craft-<basename>` — our own build via image_tag_for_task
    - `<basename[:32]>*` — Harbor compose-project-prefixed images
      (trial_name = basename[:32] + '-' + uuid; compose appends
      `-<service>` per image — normally just `-main`).

    Call only when task.stage is terminal (ACCEPTED/REJECTED/NEEDS_FIX).
    Best-effort: never raises. Does NOT touch Claude's ad-hoc verify-builds
    from the build_dockerfile step (those use arbitrary tag names like
    `test-*` and are cleaned via the operator-prune runbook step).

    Base layers shared with other live task images stay referenced —
    removing tags only frees layers when the last reference is gone. This
    is correct behaviour.
    """
    if not task_dir:
        return

    basename = Path(task_dir).name.lower()
    patterns = [f"craft-{basename}", f"{basename[:32]}*"]

    for pattern in patterns:
        try:
            lst = await asyncio.create_subprocess_exec(
                "docker",
                "images",
                "--format",
                "{{.Repository}}:{{.Tag}}",
                "--filter",
                f"reference={pattern}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(lst.communicate(), timeout=30)
        except (OSError, asyncio.TimeoutError) as e:
            print(f"[cleanup] WARN: listing {pattern!r} failed: {e}", flush=True)
            continue

        tags = [t.strip() for t in stdout.decode().splitlines() if t.strip()]
        if not tags:
            continue

        try:
            rm = await asyncio.create_subprocess_exec(
                "docker",
                "image",
                "rm",
                "-f",
                *tags,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(rm.wait(), timeout=30)
            print(f"[cleanup] removed {len(tags)} image(s) for {basename} (pattern={pattern!r})", flush=True)
        except (OSError, asyncio.TimeoutError) as e:
            print(f"[cleanup] WARN: rm {pattern!r} failed: {e}", flush=True)


async def run_docker_build_async(task_dir: str, timeout: int = 900) -> tuple[bool, str]:
    """Build Docker image for a task asynchronously. Returns (success, output)."""
    dockerfile = os.path.join(task_dir, "environment", "Dockerfile")
    if not os.path.isfile(dockerfile):
        return False, f"No Dockerfile at {dockerfile}"

    image_tag = image_tag_for_task(task_dir)
    abs_dockerfile = os.path.abspath(dockerfile)
    abs_env_dir = os.path.abspath(os.path.join(task_dir, "environment"))
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "build",
            "-t",
            image_tag,
            "-f",
            abs_dockerfile,
            abs_env_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = (stdout.decode() + stderr.decode())[-3000:]
        return proc.returncode == 0, output
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return False, "Docker build timed out"
    except OSError as e:
        return False, f"Docker build failed: {e}"


async def run_f2p_p2p_classify_async(
    task_dir: str,
    test_paths: list[str],
    timeout: int = 600,
) -> tuple[list[str] | None, list[str] | None, str]:
    """Run 2-run F2P/P2P classification inside a single Docker container.

    Two sequential runs determine which tests are F2P vs P2P:
      Run 1 (overlay):  pre-merge code + postmerge test files overlaid onto /code
      Run 2 (oracle):   solve.sh applied + postmerge test files

    Classification per test (overlay_collected = PASSED union FAILED):
      - Passes overlay + passes oracle -> P2P (was already working before the fix)
      - Fails overlay  + passes oracle -> F2P (newly passing after the fix)
      - Passes overlay + fails oracle  -> regression; task is dropped (invalid)
      - Not collected in overlay       -> task is dropped (cannot classify)

    Returns (f2p_tests, p2p_tests, output).
    Returns (None, None, output) on: oracle 0 tests, solve.sh failure, timeout,
    regression (overlay-passed test fails oracle), or test not collected in overlay.
    """
    image_tag = image_tag_for_task(task_dir)
    solution_mount = os.path.abspath(os.path.join(task_dir, "solution"))
    postmerge_dir = os.path.abspath(os.path.join(task_dir, "tests", "postmerge_tests"))

    paths_str = " ".join(shlex.quote(p) for p in test_paths)

    classify_script = (
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        "cd /code\n\n"
        "if [ -d /postmerge ]; then\n"
        "    (cd /postmerge && find . -type f -name '*.py' | while IFS= read -r f; do\n"
        '        dst="/code/${f#./}"\n'
        '        mkdir -p "$(dirname "$dst")"\n'
        '        cp "$f" "$dst"\n'
        "    done)\n"
        "fi\n\n"
        "echo '===OVERLAY_START==='\n"
        f"python3 -m pytest {paths_str} -v --tb=no --continue-on-collection-errors 2>&1 || true\n"
        "echo '===OVERLAY_END==='\n\n"
        'bash /solution/solve.sh 2>&1; echo "===SOLVE_EXIT=$?==="\n\n'
        "echo '===ORACLE_START==='\n"
        f"python3 -m pytest {paths_str} -v --tb=no --continue-on-collection-errors 2>&1 || true\n"
        "echo '===ORACLE_END==='\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as sf:
        script_path = sf.name
        sf.write(classify_script)

    try:
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{os.path.abspath(script_path)}:/classify.sh:ro",
            "-v",
            f"{solution_mount}:/solution:ro",
        ]
        if os.path.isdir(postmerge_dir):
            cmd.extend(["-v", f"{postmerge_dir}:/postmerge:ro"])
        cmd.extend([image_tag, "bash", "/classify.sh"])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode() + stderr.decode()
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return None, None, "TIMEOUT: F2P/P2P classification timed out"
        except OSError as e:
            return None, None, f"F2P/P2P classification failed: {e}"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    lines = output.split("\n")

    def _extract_tests(section: str, statuses: tuple[str, ...] = ("PASSED",)) -> set[str]:
        start_marker = f"==={section}_START==="
        end_marker = f"==={section}_END==="
        in_section = False
        tests: set[str] = set()
        for line in lines:
            if start_marker in line:
                in_section = True
                continue
            if end_marker in line:
                in_section = False
                continue
            if in_section:
                for status in statuses:
                    m = re.match(rf"^(\S+::\S+)\s+{status}", line)
                    if m:
                        tests.add(m.group(1))
        return tests

    solve_exit = None
    for line in lines:
        m = re.match(r"===SOLVE_EXIT=(\d+)===", line)
        if m:
            solve_exit = int(m.group(1))
            break
    if solve_exit is None:
        return None, None, f"SOLVE_FAIL: solve.sh sentinel not found (container crash?)\n{output[-3000:]}"
    if solve_exit != 0:
        return None, None, f"SOLVE_FAIL: patch exited {solve_exit}\n{output[-3000:]}"

    overlay_passed = _extract_tests("OVERLAY")
    overlay_collected = _extract_tests("OVERLAY", statuses=("PASSED", "FAILED"))
    oracle_passed = _extract_tests("ORACLE")

    if not oracle_passed:
        return None, None, f"ORACLE_ZERO: {output[-3000:]}"

    # Regression: overlay-passed test fails oracle → invalid task
    regressions = overlay_passed - oracle_passed
    if regressions:
        reg_list = sorted(regressions)
        return None, None, f"OVERLAY_REGRESSION: {reg_list}\n{output[-3000:]}"

    f2p_set: set[str] = set()
    p2p_set: set[str] = set()
    for test in oracle_passed:
        if test in overlay_passed:
            p2p_set.add(test)
        elif test in overlay_collected:
            f2p_set.add(test)
        else:
            # Not collected in overlay at all → cannot classify → drop task
            return None, None, f"OVERLAY_UNCOLLECTED: {test}\n{output[-3000:]}"

    return sorted(f2p_set), sorted(p2p_set), output[-3000:]


async def run_score_check_async(
    task_dir: str,
    apply_solution: bool = False,
    timeout: int = 600,
) -> dict:
    """Run score.py in Docker to compute F2P/P2P scores.

    apply_solution=False: NOP check (f2p_score should be 0, p2p_score should be 1)
    apply_solution=True:  oracle check (resolved should be True)

    Returns dict with reward, resolved, f2p_score, p2p_score, and diagnostic counts.
    On error returns {"error": reason, "output": text}.
    """
    image_tag = image_tag_for_task(task_dir)
    tests_mount = os.path.abspath(os.path.join(task_dir, "tests"))
    solution_mount = os.path.abspath(os.path.join(task_dir, "solution"))

    docker_cmd = "bash /solution/solve.sh && bash /tests/test.sh" if apply_solution else "bash /tests/test.sh"

    logs_dir = tempfile.mkdtemp()
    try:
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{tests_mount}:/tests:ro",
            "-v",
            f"{solution_mount}:/solution:ro",
            "-v",
            f"{logs_dir}:/logs",
            image_tag,
            "bash",
            "-c",
            docker_cmd,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = (stdout.decode() + stderr.decode())[-3000:]
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return {"error": "timeout", "output": "Score check timed out"}
        except OSError as e:
            return {"error": "docker_unavailable", "output": str(e)}

        if proc.returncode != 0:
            output = f"[container exited {proc.returncode}]\n" + output

        reward_path = os.path.join(logs_dir, "verifier", "reward.json")
        if not os.path.isfile(reward_path):
            return {"error": "no_reward_json", "output": output}

        try:
            with open(reward_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return {"error": "json_parse", "parse_error": str(e), "output": output}

        return {
            "reward": data.get("reward", 0.0),
            "resolved": data.get("resolved", False),
            "f2p_score": data.get("f2p_score", 0.0),
            "p2p_score": data.get("p2p_score", 0.0),
            "f2p_passed": data.get("f2p_passed", 0),
            "f2p_total": data.get("f2p_total", 0),
            "p2p_passed": data.get("p2p_passed", 0),
            "p2p_total": data.get("p2p_total", 0),
            "output": output,
        }
    finally:
        try:
            shutil.rmtree(logs_dir)
        except Exception:
            pass  # root-owned files from container; leave for OS to clean up
