# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared proxy/runtime helpers for one-command launchers."""

import logging
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn

from switchyard.cli.config.user_config import get_user_config_dir
from switchyard.lib.route_table import SwitchyardApp
from switchyard.server.switchyard_app import build_switchyard_app

if TYPE_CHECKING:
    from switchyard.lib.profiles.deterministic_routing_config import DeterministicRoutingConfig

_debug_file_handler: logging.FileHandler | None = None
log = logging.getLogger(__name__)

#: Opener that ignores env proxies — loopback probes to the in-process proxy
#: must never be routed through a configured HTTP_PROXY.
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


#: System CA bundle paths to try (Debian/Ubuntu, RHEL/CentOS/Fedora).
_SYSTEM_CA_BUNDLE_CANDIDATES = (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/cert.pem",
)


def ensure_system_ssl_trust() -> None:
    """Point Python's TLS at the system CA bundle when one is available.

    The classifier path uses the OpenAI Python SDK → httpx, which defaults
    to certifi's CA bundle. On Linux dev environments behind a corporate
    outbound SSL intercept (NVIDIA dev boxes are one), certifi doesn't
    include the intercept CA — every classifier call fails with
    ``Connection error`` at TLS handshake. Curl and the Rust backend
    (``rustls-tls-native-roots`` feature) use the OS keystore at
    ``/etc/ssl/certs/ca-certificates.crt`` which *does* include the
    intercept CA. Setting ``SSL_CERT_FILE`` tells Python's httpx /
    OpenAI SDK to do the same.

    No-op when ``SSL_CERT_FILE`` is already set (user override wins) or
    when no known system bundle path exists (macOS without an explicit
    bundle; we don't touch the Mac keychain path).
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    for candidate in _SYSTEM_CA_BUNDLE_CANDIDATES:
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return


def find_free_port() -> int:
    """Bind to ``127.0.0.1:0``, let the OS pick, return the chosen port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


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


def spawn_proxy_thread(
    switchyard: SwitchyardApp,
    port: int,
    *,
    thread_name: str,
) -> tuple[uvicorn.Server, threading.Thread]:
    """Run uvicorn in a background daemon thread."""
    app = build_switchyard_app(switchyard)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    def _run_server() -> None:
        try:
            server.run()
        except Exception:
            log.debug("uvicorn server thread crashed on port %d", port, exc_info=True)
            raise

    thread = threading.Thread(target=_run_server, name=thread_name, daemon=True)
    thread.start()
    log.debug(
        "spawned proxy thread name=%s port=%d alive=%s",
        thread.name,
        port,
        thread.is_alive(),
    )
    return server, thread


def configure_debug_file_logging(*, display_model: str) -> Path:
    """Move launcher diagnostics to a per-run debug log and return its path."""
    global _debug_file_handler

    log_dir = get_user_config_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"switchyard-{os.getpid()}.log"

    if _debug_file_handler is not None:
        _debug_file_handler.close()

    # Truncate any prior file under this pid (only matters if a stale launch
    # left one behind), then open the FileHandler in append + delay mode.
    # Background: uvicorn's startup runs `logging.config.dictConfig(...)`,
    # which internally calls `logging.shutdown()` on every registered handler.
    # `FileHandler.close()` then sets `stream=None` AND sets `_closed=True`.
    # If we'd opened in mode="w", `FileHandler.emit()` refuses to reopen the
    # stream (to avoid silently truncating the file a second time), and every
    # subsequent log line vanishes — the handler is still attached but
    # streamless. Append + delay sidesteps this: the handler is allowed to
    # reopen its stream on demand, and the file content is preserved.
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

    for name in (
        "switchyard",
        "httpx",
        "httpcore",
        "openai",
        "anthropic",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ):
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    logging.getLogger("switchyard").info(
        "=== switchyard debug log: model=%s pid=%d ===",
        display_model,
        os.getpid(),
    )
    return log_path


def silence_launch_loggers(*, local_logger: logging.Logger) -> None:
    """Keep dependency chatter out of a child process terminal UI."""
    for noisy in (
        "switchyard",
        "httpx",
        "httpcore",
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "openai",
        "anthropic",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    local_logger.setLevel(logging.INFO)


def suppress_uvicorn_stream_handlers() -> None:
    """Remove uvicorn stdout/stderr handlers after startup."""
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                continue
            logger.removeHandler(handler)
            handler.close()
        if _debug_file_handler is not None and _debug_file_handler not in logger.handlers:
            logger.addHandler(_debug_file_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False


def stdin_is_tty() -> bool:
    """Return whether stdin is a usable TTY."""
    try:
        return os.isatty(sys.stdin.fileno())
    except Exception:
        return False


def passthrough_strategy_summary(model: str) -> str:
    """Return the strategy summary string for a single-model passthrough."""
    return f"passthrough → {model}"


def routing_profiles_strategy_summary(routing_profiles: str, default_model: str) -> str:
    """Return the strategy summary for the default route in a routing-profiles launch.

    Parses the YAML to find the default route (first key) and describes its type.
    Falls back to ``routing-profiles: <default_model>`` if parsing fails.
    """
    try:
        from collections.abc import Mapping as _Mapping
        from importlib import import_module
        yaml = import_module("yaml")
        raw = yaml.safe_load(Path(routing_profiles).read_text())
        routes = raw.get("routes") if isinstance(raw, dict) else None
        if isinstance(routes, _Mapping) and routes:
            first_key = next(iter(routes))
            route = routes[first_key]
            route_type = route.get("type") if isinstance(route, _Mapping) else None
            if isinstance(route_type, str):
                return _route_type_summary(route_type.lower().replace("-", "_"), route, first_key)
    except Exception:
        pass
    return f"routing-profiles: {default_model}"


def _route_type_summary(route_type: str, route: object, route_key: str) -> str:
    """Describe a single route by type for the startup banner."""
    from collections.abc import Mapping as _Mapping
    r = route if isinstance(route, _Mapping) else {}

    def _model(tier: object) -> str:
        return tier.get("model", "") if isinstance(tier, _Mapping) else ""

    def _classifier_part(r: _Mapping) -> str:  # type: ignore[type-arg]
        clf = r.get("classifier")
        m = _model(clf)
        return f", llm-classifier={m}" if m else ""

    if route_type == "stage_router":
        clf = _classifier_part(r)
        threshold = r.get("confidence_threshold")
        threshold_part = f", confidence_threshold={threshold}" if threshold is not None else ""
        return f"stage_router: strong={_model(r.get('strong'))}, weak={_model(r.get('weak'))}{clf}{threshold_part}"
    if route_type in ("deterministic", "llm_classifier"):
        clf = _classifier_part(r)
        profile = r.get("profile")
        profile_part = f", profile={profile}" if profile else ""
        return f"llm-classifier: strong={_model(r.get('strong'))}, weak={_model(r.get('weak'))}{clf}{profile_part}"
    if route_type == "plan_execute":
        cadence = r.get("cadence_n")
        cadence_part = f", cadence_n={cadence}" if cadence is not None else ""
        return f"plan-execute: strong={_model(r.get('strong'))}, weak={_model(r.get('weak'))}{cadence_part}"
    if route_type == "random_routing":
        p = r.get("strong_probability", "")
        return f"random-routing: strong={_model(r.get('strong'))}, weak={_model(r.get('weak'))}, p_strong={p}"
    if route_type in ("model", "passthrough"):
        target = r.get("model") or r.get("target") or route_key
        return f"passthrough → {target}"
    return f"{route_type}: {route_key}"


def deterministic_strategy_summary(config: "DeterministicRoutingConfig") -> str:
    """Return the strategy summary string for a deterministic (LLM-classifier) launch."""
    return (
        f"llm-classifier: classifier={config.classifier.model}, "
        f"strong={config.strong.model}, weak={config.weak.model}, "
        f"profile={config.profile_name}"
    )


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
    ``passthrough → model`` → single line (no key=value pairs to split).
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
    profile_routes: list[str] | None = None,
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

    if profile_routes:
        col = "  profiles  "
        pad = " " * len(col)
        lines.append("")
        shown, rest = profile_routes[:5], profile_routes[5:]
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
        f"  {dim('stats')}     {dim(f'curl -s {base}/v1/routing/stats | python3 -m json.tool')}",
    ]
    if log_path is not None:
        lines.append(f"  {dim('debug')}     {dim(str(log_path))}")
    lines += [sep, ""]

    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


def banner_pause(timeout: float = 10.0) -> None:
    """Hold the banner on screen for up to *timeout* seconds.

    Returns early if the user presses any key. Only call when stdin is a TTY.
    """
    import os
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
