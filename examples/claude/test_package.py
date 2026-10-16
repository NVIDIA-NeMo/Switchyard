# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the packaged native helper: python test_package.py <plugin directory>."""

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def tool(call_id, name, text, failed=False):
    return [
        {"role": "assistant", "content": [
            {"type": "tool_call", "id": call_id, "name": name, "arguments": {}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_call_id": call_id,
             "content": [{"type": "text", "text": text}], "is_error": failed}
        ]},
    ]


PROMPT = [{"role": "user", "content": [{"type": "text", "text": "Fix the tests"}]}]


class Helper:
    """Speaks the mod's protocol: request files in, JSON lines out."""

    def __init__(self, plugin, tmp, config=None, session="session"):
        self.proc = subprocess.Popen(
            [plugin / "bin/switchyard-claude", config or plugin / "switchyard.toml"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, cwd="/tmp",
            env={"PATH": "/usr/bin:/bin", "TMPDIR": tmp},
        )
        self.session = session
        self.buffer = b""
        self.count = 0
        self.dir = Path(self.line()["dir"])

    def line(self, timeout=5):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            left = deadline - time.monotonic()
            assert left > 0 and select.select([self.proc.stdout], [], [], left)[0], "helper silent"
            chunk = os.read(self.proc.stdout.fileno(), 4096)
            assert chunk, f"helper exited: {self.proc.stderr.read().decode()}"
            self.buffer += chunk
        text, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(text)

    def send(self, body):
        self.count += 1
        (self.dir / f"req-{self.count}.json").write_text(body)
        return self.count

    def ask(self, messages, session=None):
        number = self.send(json.dumps({"session_id": session or self.session, "messages": messages}) + "\n")
        answer = self.line()
        assert answer["id"] == number, answer
        return answer

    def model(self, messages, session=None):
        return self.ask(messages, session)["model"]

    def exits(self, timeout=3):
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise AssertionError("helper did not exit") from None
        assert not self.dir.exists(), "helper left its directory"

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
        self.exits()


class HungClassifier:
    """A local HTTP server that accepts a classifier call and never answers it."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", "0")))
                outer.entered.set()
                outer.release.wait(10)
                self.send_response(500)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def config(self, path):
        path.write_text(f'''schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:{self.server.server_port}/v1"
[targets.fast]
id = "claude-sonnet-5-5"
llm_client = "local"
[targets.strong]
id = "claude-opus-5-5"
llm_client = "local"
[routes.test]
id = "switchyard"
type = "llm_classifier"
classifier_target = "strong"
strong_target = "strong"
weak_target = "fast"
base_threshold = 0.5
''')
        return path

    def close(self):
        self.release.set()
        self.server.shutdown()


def stops_during_decision(plugin, tmp, how):
    """A helper waiting on the classifier must still stop when told to."""
    stub = HungClassifier()
    try:
        config = stub.config(Path(tmp) / f"classifier-{how}.toml")
        request = json.dumps({"session_id": "session", "messages": PROMPT}) + "\n"
        if how == "parent":
            # The intermediate parent exits while the decision is in flight.
            parent = subprocess.Popen(
                [sys.executable, "-I", "-c",
                 "import subprocess,sys\n"
                 "p=subprocess.Popen(sys.argv[1:],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)\n"
                 "sys.stdout.buffer.write(p.stdout.readline());sys.stdout.flush()\n"
                 "sys.stdin.readline()\n"
                 "import os;os._exit(0)",
                 str(plugin / "bin/switchyard-claude"), str(config), how],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                env={"PATH": "/usr/bin:/bin", "TMPDIR": tmp},
            )
            directory = Path(json.loads(parent.stdout.readline())["dir"])
        else:
            helper = Helper(plugin, tmp, config=config, session=how)
            directory = helper.dir
        (directory / "req-1.json").write_text(request)
        assert stub.entered.wait(3), "decision never reached the classifier"
        if how == "parent":
            parent.stdin.write(b"\n")
            parent.stdin.close()
            parent.wait()
        elif how == "directory":
            shutil.rmtree(directory)
        else:
            helper.proc.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and (directory.exists() if how != "directory" else helper.proc.poll() is None):
            time.sleep(0.05)
        if how == "parent":
            assert not directory.exists(), "helper kept running after its parent died"
        else:
            assert helper.proc.poll() is not None, f"helper kept running after {how}"
            if how != "directory":
                assert not directory.exists()
    finally:
        stub.close()
        if how != "parent" and helper.proc.poll() is None:
            helper.proc.kill()


def main():
    plugin = Path(sys.argv[1]).resolve()
    config = plugin / "switchyard.toml"
    with tempfile.TemporaryDirectory() as tmp:
        def helpers(**kw):
            return Helper(plugin, tmp, **kw)

        # Selection, escalation, and recovery over one growing transcript.
        h = helpers()
        assert oct(h.dir.stat().st_mode & 0o777) == "0o700"
        pid = h.proc.pid
        messages = list(PROMPT)
        assert h.model(messages) == "claude-sonnet-5-5"
        messages += tool("failure", "Bash", "out of memory", True)
        assert h.model(messages) == "claude-opus-5-5"
        answers = []
        for i in range(3):
            messages += tool(f"write-{i}", "Write", "wrote file")
            answers.append(h.model(messages))
        assert answers == ["claude-opus-5-5"] * 2 + ["claude-sonnet-5-5"], answers
        assert h.proc.pid == pid and h.proc.poll() is None
        h.close()

        # One helper keeps the capable-tier hold across decisions; a fresh one starts clean.
        # Each tail omits the failure, so only retained state can still pick Opus.
        def tail(i):
            return PROMPT + tool(f"write-{i}", "Write", "wrote file")
        h = helpers(session="a")
        assert h.model(PROMPT + tool("failure", "Bash", "out of memory", True)) == "claude-opus-5-5"
        assert h.model(tail(0), session="b") == "claude-sonnet-5-5"
        other = helpers(session="b")  # concurrent helpers share nothing
        assert other.model(tail(0)) == "claude-sonnet-5-5"
        assert [h.model(tail(i)) for i in range(3)] == ["claude-opus-5-5"] * 2 + ["claude-sonnet-5-5"]
        other.close()
        h.close()
        h = helpers(session="a")
        assert h.model(PROMPT + tool("failure", "Bash", "out of memory", True)) == "claude-opus-5-5"
        h.close()
        restarted = helpers(session="a")
        assert restarted.model(tail(0)) == "claude-sonnet-5-5"
        restarted.close()

        # The same Runner repeatedly escalates and recovers, without a helper restart.
        h = helpers()
        pid = h.proc.pid
        for cycle in range(3):
            assert h.model(PROMPT + tool(f"failure-{cycle}", "Bash", "out of memory", True)) == "claude-opus-5-5"
            # Switching sessions must not reset the original session's recovery hold.
            assert h.model(tail(0), session=f"other-{cycle}") == "claude-sonnet-5-5"
            assert [h.model(tail(i)) for i in range(3)] == ["claude-opus-5-5"] * 2 + ["claude-sonnet-5-5"]
            assert h.proc.pid == pid and h.proc.poll() is None
        h.close()

        # A slow first write must not let a later decision advance the session first.
        h = helpers()
        try:
            first = h.send("{")
            time.sleep(0.1)
            second = h.send(json.dumps({"session_id": h.session, "messages": tail(0)}))
            time.sleep(0.1)
            (h.dir / f"req-{first}.json").write_text(json.dumps({
                "session_id": h.session,
                "messages": PROMPT + tool("ordered-failure", "Bash", "out of memory", True),
            }))
            answers = [h.line(), h.line()]
            assert [a["id"] for a in answers] == [first, second], answers
            assert [a["model"] for a in answers] == ["claude-opus-5-5"] * 2, answers
        finally:
            h.close()

        # A bad request gets an error answer and the helper keeps serving.
        h = helpers()
        assert h.ask([{"role": "nobody"}])["error"] == "Invalid routing messages"
        number = h.send("{")
        answer = h.line(timeout=5)
        assert answer == {"id": number, "error": "Invalid routing request"}, answer
        assert h.model(PROMPT) == "claude-sonnet-5-5"
        # A request still being written is read once it is complete.
        number = h.send("")
        time.sleep(0.2)
        (h.dir / f"req-{number}.json").write_text(json.dumps({"session_id": "session", "messages": PROMPT}) + "\n")
        assert h.line()["id"] == number
        h.close()

        # The helper ends when its directory is removed, on SIGTERM, and when its parent dies.
        h = helpers()
        for entry in h.dir.iterdir():
            entry.unlink()
        subprocess.run(["rmdir", h.dir], check=True)
        h.proc.wait(3)
        h = helpers()
        h.proc.send_signal(signal.SIGTERM)
        h.exits()
        parent = subprocess.Popen(
            [sys.executable, "-I", "-c",
             "import os,subprocess,sys\n"
             "p=subprocess.Popen(sys.argv[1:],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)\n"
             "sys.stdout.buffer.write(p.stdout.readline());sys.stdout.flush()\n"
             "os._exit(0)",
             str(plugin / "bin/switchyard-claude"), str(config), "orphan"],
            stdout=subprocess.PIPE, env={"PATH": "/usr/bin:/bin", "TMPDIR": tmp},
        )
        orphan_dir = Path(json.loads(parent.stdout.readline())["dir"])
        parent.wait()
        deadline = time.monotonic() + 3
        while orphan_dir.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not orphan_dir.exists(), "orphaned helper kept running"
        assert not list(Path(tmp).iterdir()), "helpers left directories behind"

        # A decision waiting on a hung classifier is cancelled by every way to stop the helper.
        for how in ("directory", "sigterm", "parent"):
            stops_during_decision(plugin, tmp, how)
        assert not list(Path(tmp).glob("switchyard-claude-*")), "helpers left directories behind"

        # Startup failures exit nonzero with a message.
        def startup(*args):
            return subprocess.run(
                [plugin / "bin/switchyard-claude", *args], capture_output=True, text=True,
                timeout=3, cwd="/tmp", env={"PATH": "/usr/bin:/bin", "TMPDIR": tmp},
            )

        assert startup().returncode != 0
        assert startup(plugin / "missing.toml", "s").returncode != 0
        invalid = Path(tmp) / "invalid.toml"
        invalid.write_text(config.read_text().replace('id = "switchyard"', 'id = "other"'))
        h = helpers(config=invalid)
        assert h.ask(PROMPT)["error"] == "Missing switchyard route"
        h.close()
    print("Packaged helper: selection, hold, restart, isolation, bad input, and shutdown checks passed")


if __name__ == "__main__":
    main()
