# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared proxy/runtime helpers for one-command launchers."""

import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

_debug_file_handler: logging.FileHandler | None = None
log = logging.getLogger(__name__)

#: Platform flags resolved once at import; ``os.name`` never changes at
#: runtime. Read via these module constants so launcher helpers stay testable
#: without mutating the real ``os`` module (which would also flip pathlib's
#: ``WindowsPath``/``PosixPath`` selection).
_IS_WINDOWS = os.name == "nt"
_IS_POSIX = os.name == "posix"

#: Opener that ignores env proxies — loopback probes to the in-process proxy
#: must never be routed through a configured HTTP_PROXY.
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

#: File suffixes that mark a Windows batch shim rather than a native executable.
_WINDOWS_BATCH_SUFFIXES = (".bat", ".cmd")


def is_windows_batch_shim(program: str) -> bool:
    """Return whether *program* is a Windows ``.bat``/``.cmd`` shim.

    Agent CLIs installed through npm on Windows are batch shims that
    ``CreateProcess`` cannot exec directly; they must run through the shell.
    """
    return _IS_WINDOWS and program.lower().endswith(_WINDOWS_BATCH_SUFFIXES)


def is_executable_file(path: Path) -> bool:
    """Return whether *path* is a runnable file on this platform.

    Windows has no executable permission bit, so any existing file under a
    candidate directory may be a shim; POSIX additionally requires the
    executable bit via :func:`os.access`.
    """
    if not path.is_file():
        return False
    if _IS_WINDOWS:
        return True
    return os.access(path, os.X_OK)


def _default_state_dir() -> Path:
    """Return the platform-appropriate state directory.

    On Windows, launcher diagnostics live under ``%LOCALAPPDATA%`` rather than
    a POSIX dot-directory; ``Path.home()`` is the fallback when that variable
    is unset.
    """
    if _IS_WINDOWS:
        local_app_data = os.environ.get("LOCALAPPDATA")
        return Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return Path.home() / ".local" / "state"


def wait_for_proxy_ready(port: int, *, timeout_s: float) -> bool:
    """Probe ``GET /health`` until HTTP 200 or timeout."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            # Loopback: bypass env proxies so a configured HTTP_PROXY can't
            # intercept the 127.0.0.1 health probe.
            with _LOCAL_OPENER.open(url, timeout=0.5):
                return True
        except Exception:
            time.sleep(0.05)
    return False


def configure_debug_file_logging(*, display_model: str) -> Path:
    """Move launcher diagnostics to a per-run debug log and return its path."""
    global _debug_file_handler

    state_home = os.environ.get("XDG_STATE_HOME")
    state_dir = (
        Path(state_home).expanduser()
        if state_home
        else _default_state_dir()
    )
    log_dir = state_dir / "switchyard" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"switchyard-{os.getpid()}.log"

    if _debug_file_handler is not None:
        _debug_file_handler.close()

    # Truncate any prior file under this pid, then let the delayed handler open
    # it only when the first diagnostic is emitted.
    log_path.write_text("", encoding="utf-8")
    file_handler = logging.FileHandler(
        log_path, mode="a", encoding="utf-8", delay=True,
    )
    _debug_file_handler = file_handler
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.WARNING)

    switchyard_logger = logging.getLogger("switchyard")
    for handler in switchyard_logger.handlers[:]:
        switchyard_logger.removeHandler(handler)
        handler.close()
    switchyard_logger.addHandler(file_handler)
    switchyard_logger.setLevel(logging.DEBUG)
    switchyard_logger.propagate = False

    logging.getLogger("switchyard").info(
        "=== switchyard debug log: model=%s pid=%d ===",
        display_model,
        os.getpid(),
    )
    return log_path


def silence_launch_loggers(*, local_logger: logging.Logger) -> None:
    """Keep dependency chatter out of a child process terminal UI."""
    logging.getLogger("switchyard").setLevel(logging.WARNING)
    local_logger.setLevel(logging.INFO)


def stdin_is_tty() -> bool:
    """Return whether stdin is a usable TTY.

    The interactive footer (ShellTUI) and the raw-mode key-press pause both
    require a POSIX pseudo-terminal, so on Windows this always returns False
    and launchers fall back to a plain ``subprocess`` child that inherits the
    console directly.
    """
    if not _IS_POSIX:
        return False
    try:
        return os.isatty(sys.stdin.fileno())
    except Exception:
        return False


# Keys that are abbreviated in the banner display.
_KEY_ABBREV: dict[str, str] = {
    "confidence_threshold": "conf",
    "llm-classifier": "classifier",
}


def _c(code: str, text: str, enabled: bool) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if enabled else text


def _format_strategy_lines(summary: str) -> list[str]:
    """Expand a strategy summary string into one or more banner lines.

    ``type: k1=v1, k2=v2`` → type on first line, each k=v pair indented.
    ``noop`` → single line (no key=value pairs to split).
    """
    col = "  routing   "  # label column: 2 indent + 7 label + 3 gap = 12 chars
    pad = " " * len(col)
    if ": " not in summary:
        return [f"{col}{summary}"]
    route_type, rest = summary.split(": ", 1)
    pairs = [p.strip() for p in rest.split(", ") if p.strip()]
    return [f"{col}{route_type}", *(f"{pad}{p}" for p in pairs)]


def _format_strategy_indented(
    summary: str, indent: str, *, color: bool = False,
) -> list[str]:
    """Expand a strategy summary as aligned key→value rows under *indent*."""
    def dim(t: str) -> str: return _c("2", t, color)
    def bold(t: str) -> str: return _c("1", t, color)

    if ": " not in summary:
        return [f"{indent}{summary}"]

    route_type, rest = summary.split(": ", 1)
    raw_pairs = [p.strip() for p in rest.split(", ") if p.strip()]

    kvs: list[tuple[str, str]] = []
    for p in raw_pairs:
        k, _, v = p.partition("=")
        kvs.append((_KEY_ABBREV.get(k, k), v if _ else p))

    max_key = max((len(k) for k, _ in kvs if _), default=0)

    out = [f"{indent}{dim(route_type)}"]
    for k, v in kvs:
        pad = " " * (max_key - len(k) + 2)
        out.append(f"{indent}{dim(k)}{pad}{bold(v)}")
    return out


def print_ready_banner(
    *,
    port: int,
    display_model: str,
    log_path: Path | None = None,
    strategy_summary: str | None = None,
    routes: list[str] | None = None,
    default_route: str | None = None,
) -> None:
    """Write proxy/stats/routing details to stderr before the child takes over."""
    clr = sys.stderr.isatty()

    def dim(t: str) -> str:   return _c("2", t, clr)
    def bold(t: str) -> str:  return _c("1", t, clr)
    def green(t: str) -> str: return _c("32", t, clr)
    def cyan(t: str) -> str:  return _c("36", t, clr)
    def amber(t: str) -> str: return _c("33", t, clr)

    base = f"http://127.0.0.1:{port}"
    sep = "  " + dim("─" * 58)

    lines: list[str] = [
        "",
        sep,
        f"  {bold('switchyard')}  {green('ready')}  →  {bold(display_model)}",
    ]

    if routes:
        col = "  routes    "
        pad = " " * len(col)
        lines.append("")
        shown, rest = routes[:5], routes[5:]
        for i, route in enumerate(shown):
            is_default = route == default_route
            marker = amber("▶") + " " if is_default else dim("○") + " "
            default_tag = dim("  (default)") if is_default else ""
            prefix = col if i == 0 else pad
            name = bold(route) if is_default else dim(route)
            lines.append(f"{prefix}{marker}{name}{default_tag}")
            if is_default and strategy_summary:
                lines.extend(
                    _format_strategy_indented(strategy_summary, pad + "    ", color=clr)
                )
        if rest:
            lines.append(f"{pad}  {dim(f'… +{len(rest)} more')}")
    elif strategy_summary:
        lines.append("")
        lines.extend(_format_strategy_lines(strategy_summary))

    lines += [
        "",
        f"  {dim('proxy')}     {cyan(base)}",
        f"  {dim('models')}    {dim(f'curl -s {base}/v1/models')}",
        f"  {dim('stats')}     {dim(f'curl -s {base}/v1/stats | python3 -m json.tool')}",
    ]
    if log_path is not None:
        lines.append(f"  {dim('debug')}     {dim(str(log_path))}")
    lines += [sep, ""]

    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


def banner_pause(timeout: float = 10.0) -> None:
    """Hold the banner on screen for up to *timeout* seconds.

    Returns early if the user presses any key. Only call when stdin is a TTY.
    The raw-mode key wait is POSIX-only; on Windows this is a no-op (the
    console passes keystrokes straight to the child process anyway).
    """
    if not _IS_POSIX:
        return
    import select
    import termios
    import tty
    if not sys.stdin.isatty():
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sys.stderr.write(f"  [press any key to start, or waiting {int(timeout)}s…]\n")
    sys.stderr.flush()
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            os.read(fd, 1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    sys.stderr.write("\r\x1b[2K")
    sys.stderr.flush()


def print_startup_failure(*, port: int, timeout_s: float, log_path: Path) -> None:
    """Write proxy startup failure details to stderr."""
    sys.stderr.write(
        f"switchyard: proxy failed to become ready within {timeout_s:.1f}s — "
        f"GET http://127.0.0.1:{port}/health never returned 200\n"
        f"Check {log_path} for details.\n"
    )
    sys.stderr.flush()
