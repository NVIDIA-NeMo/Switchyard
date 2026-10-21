# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = ["datacurve-pier==0.3.1", "harbor==0.24.0"]
# ///
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Set Docker memory limits and read Linux cgroup v2 counters."""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from pier.environments.docker.docker import DockerEnvironment

GIB = 2**30
LIMIT = 8 * GIB
PROXY_LIMIT = GIB // 2


class MemoryDocker(DockerEnvironment):
    @property
    def _docker_compose_paths(self):
        services = {"main": {"mem_limit": LIMIT, "memswap_limit": LIMIT}}
        if self._egress_proxy_compose_path:
            services["pier-egress-proxy"] = {"mem_limit": PROXY_LIMIT, "memswap_limit": PROXY_LIMIT}
        path = self.trial_paths.trial_dir / "docker-compose-memory.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"services": services}))
        return [*super()._docker_compose_paths, path]

    async def start(self, force_build):
        self._monitor = None
        self._stop = asyncio.Event()
        self._records, self._groups, self._error = [], [], None
        self._sampled_peak = 0
        phase = "verifier" if "__verifier" in self.session_id else "agent"
        self._report = self.trial_paths.trial_dir / f"memory-{phase}.json"
        await super().start(force_build)
        ids = (await self._run_docker_compose_command(["ps", "--all", "--quiet"])).stdout.split()
        containers = json.loads(await asyncio.to_thread(subprocess.check_output, ["docker", "inspect", *ids]))
        services = {c["Config"]["Labels"]["com.docker.compose.service"] for c in containers}
        expected = {"main", "pier-egress-proxy"} if self._egress_proxy_compose_path else {"main"}
        if services != expected:
            raise RuntimeError(f"Expected containers {expected}; found {services}")
        for container in containers:
            cid, pid = container["Id"], container["State"]["Pid"]
            group = next(line[3:] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines() if line.startswith("0::"))
            if cid not in group:
                raise RuntimeError("Use a local Linux Docker daemon with cgroup v2")
            self._groups.append(Path("/sys/fs/cgroup") / group.lstrip("/"))
            service = container["Config"]["Labels"]["com.docker.compose.service"]
            host = container["HostConfig"]
            limit = LIMIT if service == "main" else PROXY_LIMIT
            if host["Memory"] != limit or host["MemorySwap"] != limit or (service == "main" and host["NanoCpus"] != 2_000_000_000):
                raise RuntimeError("Incorrect Docker RAM, swap, or CPU limit")
            self._records.append({"id": cid, "service": service, "limit_bytes": limit,
                                  "current_bytes": 0, "peak_bytes": 0, "swap_bytes": 0, "events": {}})
        await asyncio.to_thread(self._sample)
        if self._error:
            raise RuntimeError(self._error)
        self._monitor = asyncio.create_task(self._watch())

    def _sample(self, complete=False):
        try:
            for group, record in zip(self._groups, self._records, strict=True):
                values = {name: int((group / name).read_text()) for name in (
                    "memory.current", "memory.peak", "memory.swap.current", "memory.max", "memory.swap.max")}
                if values["memory.max"] != record["limit_bytes"] or values["memory.swap.max"] != 0:
                    raise RuntimeError("Incorrect cgroup RAM or swap limit")
                record["current_bytes"] = values["memory.current"]
                record["peak_bytes"] = max(record["peak_bytes"], values["memory.peak"])
                record["swap_bytes"] = max(record["swap_bytes"], values["memory.swap.current"])
                record["events"] = {key: int(value) for key, value in (
                    line.split() for line in (group / "memory.events").read_text().splitlines())}
            if complete and self._records:
                states = json.loads(subprocess.check_output(["docker", "inspect", *(r["id"] for r in self._records)]))
                if any(c["State"]["OOMKilled"] for c in states):
                    raise RuntimeError("Docker recorded an OOM kill")
        except Exception as error:
            self._error = str(error)
        current = sum(r["current_bytes"] for r in self._records)
        peak = sum(r["peak_bytes"] for r in self._records)
        self._sampled_peak = max(self._sampled_peak, current)
        healthy = bool(self._records) and not self._error and peak < LIMIT and all(
            r["peak_bytes"] > 0 and r["swap_bytes"] == 0
            and all(r["events"].get(key) == 0 for key in ("max", "oom", "oom_kill"))
            for r in self._records)
        report = {"complete": complete, "passed": complete and healthy,
                  "status": ("PASS" if complete else "RUN") if healthy else "FAIL",
                  "error": self._error, "current_bytes": current, "peak_bytes": peak,
                  "sampled_peak_bytes": self._sampled_peak, "containers": self._records}
        temporary = self._report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(self._report)
        with self._report.with_suffix(".jsonl").open("a") as output:
            output.write(json.dumps({"time": time.time(), **report}) + "\n")

    async def _watch(self):
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1)
                return
            except TimeoutError:
                await asyncio.to_thread(self._sample)

    async def stop(self, delete):
        try:
            if getattr(self, "_monitor", None):
                self._stop.set()
                await self._monitor
            if getattr(self, "_records", None):
                await asyncio.to_thread(self._sample, True)
        finally:
            await super().stop(delete)


if __name__ == "__main__":
    for path in sorted(Path(sys.argv[1]).glob("*/*/memory-*.json")):
        report = json.loads(path.read_text())
        oom = sum(r["events"].get("oom_kill", 0) for r in report["containers"])
        print(f"{path.parent.name}/{path.stem}: {report['status']}; "
              f"RAM {report['current_bytes'] / GIB:.3f} GiB; "
              f"peak estimate {report['peak_bytes'] / GIB:.3f}/8 GiB; OOM kills {oom}")
