# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embed a child process in a PTY with a live footer drawn below it.

Uses DECSTBM to confine the child's output to the upper rows and a
background thread to repaint the footer at a fixed refresh rate.

The footer function signature::

    footer_fn(cols: int) -> list[tuple[str, int]]

Each tuple is ``(styled_line, visible_width)`` — the visible width
excludes ANSI escape sequences so padding is applied correctly for
lines that contain colour codes or multi-cell emoji.
"""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import threading
import time
import tty
import types
from collections.abc import Callable

ESC = b"\x1b"
CSI = ESC + b"["

# Defer footer paints until the child has been silent for at least this long.
# Ink (Claude Code's renderer) emits an in-place update as a burst of small
# write() calls — cursor move, SGR colour, partial line text, clear-to-EOL,
# repeat — and between any two of those writes our non-blocking drain can hit
# BlockingIOError. If the footer paint barges in there, its leading ESC
# resets the terminal's CSI parser and the child's continuation bytes land
# naked, producing visible CSI fragments (``[3A``, ``M s``, etc.) wherever the
# tree was being redrawn. 30 ms comfortably outlasts a typical Ink burst
# (sub-10 ms locally) while keeping the footer feeling responsive between
# requests.
_FOOTER_QUIET_THRESHOLD_S = 0.030
_FOOTER_POLL_INTERVAL_S = 0.05


def _get_winsize(fd: int) -> tuple[int, int]:
    data = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
    rows, cols, _, _ = struct.unpack("HHHH", data)
    return rows, cols


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _writeall(fd: int, data: bytes) -> None:
    """Loop ``os.write`` until every byte is flushed.

    A single ``os.write`` to a tty can return fewer bytes than requested when
    the kernel's tty output buffer is full; the caller is expected to retry
    with the remaining slice. Without this loop, large writes (e.g. a
    coalesced 100-KiB Ink frame) can be truncated, which manifests as
    visibly garbled output.
    """
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


class ShellTUI:
    """Run a child process in a PTY with a live footer drawn below it.

    The terminal is split into two regions via DECSTBM (scroll-region):

    * rows 1 … (total - footer_height): the child process's viewport.
    * rows (total - footer_height + 1) … total: redrawn by a background
      thread at ``refresh_hz`` times per second.

    The child inherits a PTY that reports a reduced row count so its own
    TUI doesn't overlap the footer.  SIGWINCH is forwarded to the child
    and triggers an immediate footer repaint.

    Parameters
    ----------
    command:
        Argv for the child process (``[binary, *args]``).
    footer_fn:
        Called with the current terminal column width; returns a list of
        ``(styled_line, visible_width)`` tuples, one per footer row.
    footer_height:
        Callable returning the number of rows reserved at the bottom of the
        terminal. Called on every paint so the height can grow dynamically.
    refresh_hz:
        Footer repaint frequency in Hz.
    env:
        Optional dict of environment variable overrides merged on top of
        ``os.environ`` before ``execvpe``.  ``TERM`` is set to
        ``xterm-256color`` if not already present.
    """

    def __init__(
        self,
        command: list[str],
        footer_fn: Callable[[int], list[tuple[str, int]]] | None = None,
        footer_height: Callable[[], int] | None = None,
        refresh_hz: float = 2.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = command
        # No footer by default: the embedded child gets the full terminal.
        # Callers can still supply a footer renderer + height.
        self.footer_fn = footer_fn if footer_fn is not None else (lambda _cols: [])
        self._footer_height_fn: Callable[[], int] = (
            footer_height if footer_height is not None else (lambda: 0)
        )
        self.refresh_interval = 1.0 / refresh_hz
        self._env = env
        self._out_lock = threading.Lock()
        self._stop = threading.Event()
        self._master_fd: int | None = None
        # Monotonic time of the most recent child-output write to stdout.
        # Used by ``_maybe_paint_footer`` to debounce paints away from
        # the middle of Ink's multi-write frame bursts. ``-inf`` so the
        # very first paint (after startup, with no traffic yet) is not
        # blocked.
        self._last_activity_ts: float = float("-inf")
        # Monotonic time of the most recent footer paint. Throttled by
        # ``refresh_interval`` so we don't repaint at the polling rate.
        self._last_paint_ts: float = float("-inf")
        # Set by ``_on_winch`` (the SIGWINCH handler) and drained by the main
        # loop, which calls ``_handle_winch`` to perform the deferred resize
        # work. The handler must do no I/O of its own: CPython delivers
        # signals to the main thread between bytecodes — including bytecodes
        # inside the main thread's ``with self._out_lock:`` block in
        # ``_drain`` — and re-acquiring a non-reentrant lock from the same
        # thread deadlocks the input-forwarding loop. Doing the work in the
        # main loop sidesteps that entirely.
        self._winch_pending: bool = False

    @property
    def footer_height(self) -> int:
        return self._footer_height_fn()

    def _write(self, data: bytes) -> None:
        with self._out_lock:
            _writeall(sys.stdout.fileno(), data)

    def _apply_scroll_region(self, shell_rows: int) -> None:
        self._write(CSI + f"1;{shell_rows}r".encode() + CSI + b"1;1H")

    def _build_footer_bytes(self, rows: int, cols: int) -> bytes:
        """Render the footer as a single escape-sequence blob (no I/O).

        Order matters:

        * ``DECSC`` first — saves the child's cursor + SGR attrs so we can
          restore them at the end.
        * ``DECSTBM`` second — reapplies the scroll region. The child enters
          the alt screen with ``\\x1b[?1049h`` early on, and per xterm spec
          alt-screen entry resets DECSTBM. Reapplying it on every footer
          paint keeps the scroll region in force for the lifetime of the
          session, even after alt-screen toggles. DECSTBM also moves the
          cursor to home as a side effect — harmless because we explicitly
          move to the footer rows next, and DECRC at the end restores the
          original (saved-before-DECSTBM) position.
        * Footer rows.
        * ``DECRC`` last — restores the child's cursor.
        """
        lines = list(self.footer_fn(cols))[: self.footer_height]
        while len(lines) < self.footer_height:
            lines.append(("", 0))
        shell_rows = max(1, rows - self.footer_height)
        parts = [
            ESC + b"7",                            # DECSC: save cursor + attrs
            CSI + f"1;{shell_rows}r".encode(),     # DECSTBM: keep scroll region in force
        ]
        for i, (line, visible) in enumerate(lines):
            row = shell_rows + 1 + i
            padding = max(0, cols - visible)
            parts.append(CSI + f"{row};1H".encode())
            parts.append(CSI + b"2K")
            parts.append(line.encode("utf-8", "replace"))
            if padding:
                parts.append(b" " * padding)
            parts.append(CSI + b"0m")
        parts.append(ESC + b"8")                   # DECRC: restore cursor + attrs
        return b"".join(parts)

    def _draw_footer(self, rows: int, cols: int) -> None:
        """Paint the footer under ``_out_lock``. Called from the footer thread."""
        self._write(self._build_footer_bytes(rows, cols))

    def _drain(self, master_fd: int) -> bool:
        """Drain ``master_fd`` non-blocking and write the batch to stdout.

        Returns ``False`` if the child closed (EIO or EOF), else ``True``.

        The drain does NOT paint the footer. Painting between child writes
        regularly lands inside an Ink frame burst — between two of the
        child's small write() calls, where the next bytes would have been
        the continuation of an unfinished CSI — and the paint's own ESC
        resets the terminal's parser. The footer thread paints separately,
        debounced on ``_FOOTER_QUIET_THRESHOLD_S`` of child silence so it
        only ever falls in a gap between bursts.
        """
        chunks: list[bytes] = []
        eof = False
        while True:
            try:
                data = os.read(master_fd, 65536)
            except BlockingIOError:
                break
            except OSError as e:
                if e.errno == errno.EIO:
                    eof = True
                    break
                raise
            if not data:
                eof = True
                break
            chunks.append(data)
        if chunks:
            batch = b"".join(chunks)
            with self._out_lock:
                _writeall(sys.stdout.fileno(), batch)
                # Stamp inside the lock so the footer thread, if it
                # acquires the lock next, observes a fresh timestamp on
                # its inside-lock re-check and skips painting.
                self._last_activity_ts = time.monotonic()
        return not eof

    def _maybe_paint_footer(self) -> bool:
        """Paint the footer if the child has been quiet long enough.

        Returns ``True`` iff a paint actually happened. Callable from
        unit tests as a single deterministic step in place of running
        the footer thread.

        The check is double-guarded: an optimistic outside-lock check
        avoids paying lock contention when the child is busy, and an
        inside-lock re-check handles the race where a drain wins the
        lock between the optimistic check and our acquisition (in which
        case it just bumped ``_last_activity_ts`` and the paint must
        skip).
        """
        now = time.monotonic()
        if (now - self._last_activity_ts) < _FOOTER_QUIET_THRESHOLD_S:
            return False
        if (now - self._last_paint_ts) < self.refresh_interval:
            return False
        with self._out_lock:
            now = time.monotonic()
            if (now - self._last_activity_ts) < _FOOTER_QUIET_THRESHOLD_S:
                return False
            try:
                rows, cols = _get_winsize(sys.stdout.fileno())
            except OSError:
                return False
            _writeall(sys.stdout.fileno(), self._build_footer_bytes(rows, cols))
            self._last_paint_ts = now
            return True

    def _footer_loop(self) -> None:
        while not self._stop.is_set():
            self._maybe_paint_footer()
            self._stop.wait(_FOOTER_POLL_INTERVAL_S)

    def _on_winch(self, _signum: int, _frame: types.FrameType | None) -> None:
        # Async-signal-safe: only set a flag. ``_handle_winch`` runs from the
        # main loop where ``_out_lock`` is uncontended.
        self._winch_pending = True

    def _handle_winch(self) -> None:
        """Perform deferred resize work scheduled by ``_on_winch``.

        Runs from the main loop, never from a signal handler. Steps:

        * Re-query the outer terminal size.
        * Forward the new size to the child PTY so the child receives its
          own SIGWINCH and redraws — the child's redraw is what repaints
          the upper region.
        * Reapply DECSTBM for the new shell-rows count.
        * Repaint the footer at the new geometry.
        * Stamp ``_last_activity_ts`` so the footer thread's quiet-window
          debounce blocks paints during the child's resize burst.

        Deliberately does NOT issue ``CSI 2J``: clearing the user's screen
        on resize would wipe primary-screen content the user can scroll
        back to and serves no purpose for alt-screen TUIs (the child's
        own SIGWINCH-driven full redraw covers the in-region content).
        """
        if self._master_fd is None:
            return
        try:
            rows, cols = _get_winsize(sys.stdout.fileno())
        except OSError:
            return
        shell_rows = max(1, rows - self.footer_height)
        try:
            _set_winsize(self._master_fd, shell_rows, cols)
        except OSError:
            return
        with self._out_lock:
            _writeall(
                sys.stdout.fileno(),
                CSI + f"1;{shell_rows}r".encode() + CSI + b"1;1H",
            )
            _writeall(sys.stdout.fileno(), self._build_footer_bytes(rows, cols))
            now = time.monotonic()
            self._last_paint_ts = now
            self._last_activity_ts = now

    def run(self) -> int:
        """Start the child and block until it exits.  Returns its exit code."""
        if not os.isatty(sys.stdin.fileno()):
            raise RuntimeError("ShellTUI requires a TTY on stdin")

        rows, cols = _get_winsize(sys.stdout.fileno())
        shell_rows = max(1, rows - self.footer_height)

        pid, master_fd = pty.fork()
        if pid == 0:
            child_env = os.environ.copy()
            child_env.setdefault("TERM", "xterm-256color")
            if self._env:
                child_env.update(self._env)
            try:
                os.execvpe(self.command[0], self.command, child_env)
            except OSError as e:
                sys.stderr.write(f"exec failed: {e}\n")
                os._exit(127)

        self._master_fd = master_fd
        _set_winsize(master_fd, shell_rows, cols)

        # Non-blocking master so ``_drain_and_paint`` can read until
        # ``BlockingIOError`` and emit the whole batch + footer atomically.
        fcntl.fcntl(
            master_fd,
            fcntl.F_SETFL,
            fcntl.fcntl(master_fd, fcntl.F_GETFL) | os.O_NONBLOCK,
        )

        # Disable echo + canonical mode on the PTY. Without this the line
        # discipline echoes anything we forward into the child's stdin
        # (``os.write(master_fd, ...)``) back through ``master_fd``'s read
        # side, where we then paint it to the user's stdout. The user sees
        # CSI-fragment artifacts (``[3;153R``, ``[?62;...c`` etc.) whenever
        # their real terminal answers a DSR/DA query the child issued.
        # Cbreak is the minimal-surgery option: clears ``ECHO`` and
        # ``ICANON`` but leaves ``ISIG`` on so Ctrl-C still signals the
        # child. Most TUIs (Claude Code's Ink included) reset their tty
        # to raw mode shortly after exec, but during the startup window
        # — and any time the terminal answers asynchronously — echo
        # would otherwise leak.
        try:
            tty.setcbreak(master_fd)
        except termios.error:
            pass  # best-effort — never fail launch on a quirky platform

        stdin_fd = sys.stdin.fileno()
        old_termios = termios.tcgetattr(stdin_fd)
        tty.setraw(stdin_fd)
        old_winch = signal.signal(signal.SIGWINCH, self._on_winch)

        self._write(CSI + b"2J")
        self._apply_scroll_region(shell_rows)
        self._draw_footer(rows, cols)

        footer_thread = threading.Thread(target=self._footer_loop, daemon=True)
        footer_thread.start()

        try:
            while True:
                if self._winch_pending:
                    self._winch_pending = False
                    self._handle_winch()
                try:
                    r, _, _ = select.select([stdin_fd, master_fd], [], [], 0.1)
                except (InterruptedError, OSError):
                    continue
                if master_fd in r:
                    if not self._drain(master_fd):
                        break
                if stdin_fd in r:
                    data = os.read(stdin_fd, 65536)
                    if not data:
                        break
                    os.write(master_fd, data)
                try:
                    rpid, _ = os.waitpid(pid, os.WNOHANG)
                    if rpid == pid:
                        break
                except ChildProcessError:
                    break
        finally:
            self._stop.set()
            footer_thread.join(timeout=0.5)
            signal.signal(signal.SIGWINCH, old_winch)
            try:
                self._write(CSI + b"r" + CSI + b"?25h")
                rows, _ = _get_winsize(sys.stdout.fileno())
                self._write(CSI + f"{rows};1H".encode() + b"\n")
            except OSError:
                pass
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_termios)
            try:
                os.close(master_fd)
            except OSError:
                pass

        try:
            _, status = os.waitpid(pid, 0)
            return os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1
        except ChildProcessError:
            return 0
