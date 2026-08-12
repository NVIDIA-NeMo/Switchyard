# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``ShellTUI`` drain + debounced footer painting + resize handling.

The visible failure mode these tests guard against: Claude Code's Ink
renderer emits each in-place update as a burst of small ``write()`` calls
— cursor move, SGR colour, partial line text, clear-to-EOL, repeat — and
between any two of those the kernel buffer can be empty. If we paint the
footer in that gap, our paint's leading ESC resets the terminal's CSI
parser and the child's continuation bytes (``[3A``, ``M s``, parameter
digits, …) render naked as visible text. The fix is to debounce footer
paints away from active write bursts.

These tests pin three contracts:

* ``_drain`` writes child bytes only; it does NOT inline the footer
  (which would put the paint in the worst possible place — right after
  whatever fragmentary tail the child happened to have flushed).
* ``_maybe_paint_footer`` skips while the child has written within
  ``_FOOTER_QUIET_THRESHOLD_S`` and paints only after the burst settles.
* ``_on_winch`` does no I/O — it only sets a pending flag, so the main
  thread cannot deadlock against itself by re-acquiring ``_out_lock``
  in a signal handler. ``_handle_winch`` performs the deferred resize
  work from the main loop, where the lock is uncontended.
"""

from __future__ import annotations

import fcntl
import os
import pty
import signal
import struct
import sys
import termios
import tty

import pytest

from switchyard.cli.launchers.shell_tui import CSI, ShellTUI


def _make_tui(
    footer_lines: list[tuple[str, int]] | None = None,
    footer_height: int = 1,
) -> ShellTUI:
    _h = footer_height
    return ShellTUI(
        command=["true"],
        footer_fn=lambda _cols: footer_lines or [("FOOTER-MARKER", len("FOOTER-MARKER"))],
        footer_height=lambda: _h,
    )


@pytest.fixture
def fixed_winsize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "switchyard.cli.launchers.shell_tui._get_winsize",
        lambda _fd: (24, 80),
    )


def test_build_footer_bytes_reapplies_decstbm_after_decsc() -> None:
    """Footer paint must save cursor BEFORE reapplying the scroll region.

    The child enters the alt screen with ``\\x1b[?1049h``, which (per xterm
    spec) resets DECSTBM. Reapplying ``CSI 1;<shell_rows> r`` on every
    paint keeps the scroll region in force so the child's natural scroll
    can't clobber the footer area. DECSC must come first because DECSTBM
    moves the cursor to home as a side effect; if DECSC ran second it
    would save (1,1) and DECRC would drop the child's cursor there.
    """
    tui = _make_tui(footer_lines=[("X", 1)], footer_height=2)
    blob = tui._build_footer_bytes(rows=30, cols=80)

    decsc = b"\x1b7"
    decrc = b"\x1b8"
    decstbm = b"\x1b[1;28r"  # 30 rows total - 2 footer rows = 28 shell rows

    assert decsc in blob
    assert decstbm in blob
    assert decrc in blob
    assert blob.index(decsc) < blob.index(decstbm), (
        "DECSC must precede DECSTBM so the child's cursor is saved before "
        "DECSTBM's home-cursor side effect clobbers it"
    )
    assert blob.endswith(decrc), "DECRC must be the last byte sequence"


@pytest.mark.usefixtures("fixed_winsize")
def test_drain_writes_payload_only_no_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_drain`` must NOT inline the footer.

    Inlining the footer right after the child's bytes is exactly the failure
    mode the debounce design is meant to avoid: the child's last burst can
    end mid-CSI (kernel buffer happens to be empty between two of Ink's
    short ``write()`` calls), and a footer paint emitted "right after" lands
    on top of an unfinished sequence whose continuation the terminal then
    renders naked. Drain emits child bytes only; ``_maybe_paint_footer``
    decides separately when the moment is safe.
    """
    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd, when=termios.TCSANOW)

    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    payload = b"FRAME-CONTENT-" * 64
    os.write(slave_fd, payload)

    write_calls: list[bytes] = []
    real_write = os.write
    stdout_fd = sys.stdout.fileno()

    def tracking_write(fd: int, data: bytes) -> int:
        if fd == stdout_fd:
            write_calls.append(bytes(data))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", tracking_write)
    try:
        tui = _make_tui()
        result = tui._drain(master_fd)
    finally:
        monkeypatch.setattr(os, "write", real_write)
        os.close(slave_fd)
        os.close(master_fd)

    assert result is True, "slave still open at drain time → must not signal EOF"

    stream = b"".join(write_calls)
    assert stream == payload, (
        "drain must write the child payload verbatim — nothing else, "
        "and in particular no footer escape sequences"
    )
    assert b"FOOTER-MARKER" not in stream, (
        "drain must not inline the footer; if it does, it lands inside Ink "
        "frame bursts and breaks the child's escape sequences"
    )


@pytest.mark.usefixtures("fixed_winsize")
def test_drain_returns_false_on_child_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed slave → master read raises EIO (Linux) or returns b'' (BSD); both → False."""
    master_fd, slave_fd = pty.openpty()
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    os.close(slave_fd)

    real_write = os.write
    stdout_fd = sys.stdout.fileno()
    leaked: list[bytes] = []

    def tracking_write(fd: int, data: bytes) -> int:
        if fd == stdout_fd:
            leaked.append(bytes(data))
            return len(data)
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", tracking_write)
    try:
        tui = _make_tui()
        result = tui._drain(master_fd)
    finally:
        monkeypatch.setattr(os, "write", real_write)
        try:
            os.close(master_fd)
        except OSError:
            pass

    assert result is False, "EOF on master must signal child-closed → caller breaks"
    assert leaked == [], "no chunks read → no stdout writes"


@pytest.mark.usefixtures("fixed_winsize")
def test_drain_coalesces_back_to_back_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two slave writes still produce ONE coalesced stdout write."""
    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd, when=termios.TCSANOW)
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    chunk_a = b"A" * 256
    chunk_b = b"B" * 256
    os.write(slave_fd, chunk_a)
    os.write(slave_fd, chunk_b)

    write_calls: list[bytes] = []
    real_write = os.write
    stdout_fd = sys.stdout.fileno()

    def tracking_write(fd: int, data: bytes) -> int:
        if fd == stdout_fd:
            write_calls.append(bytes(data))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", tracking_write)
    try:
        tui = _make_tui()
        tui._drain(master_fd)
    finally:
        monkeypatch.setattr(os, "write", real_write)
        os.close(slave_fd)
        os.close(master_fd)

    assert len(write_calls) == 1, (
        f"two slave writes must coalesce into one stdout write; "
        f"got {len(write_calls)}"
    )
    assert write_calls[0] == chunk_a + chunk_b


@pytest.mark.usefixtures("fixed_winsize")
def test_maybe_paint_footer_skips_when_child_recently_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Within the quiet window, the footer must not paint.

    The bug reproduced by the screenshot was the footer painting in the
    middle of an Ink burst. This test pins the inverse: when the child has
    written within ``_FOOTER_QUIET_THRESHOLD_S`` (simulated by stamping
    ``_last_activity_ts`` to right now), ``_maybe_paint_footer`` must
    return ``False`` and emit nothing.
    """
    import time as _time

    write_calls: list[bytes] = []
    real_write = os.write
    stdout_fd = sys.stdout.fileno()

    def tracking_write(fd: int, data: bytes) -> int:
        if fd == stdout_fd:
            write_calls.append(bytes(data))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", tracking_write)

    tui = _make_tui()
    tui._last_activity_ts = _time.monotonic()  # child wrote "right now"

    painted = tui._maybe_paint_footer()

    monkeypatch.setattr(os, "write", real_write)

    assert painted is False, (
        "footer must defer while the child is still in an active write burst"
    )
    assert write_calls == [], "no stdout writes during the quiet window"


@pytest.mark.usefixtures("fixed_winsize")
def test_maybe_paint_footer_paints_after_quiet_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the quiet window has elapsed, the footer paints.

    Stamping ``_last_activity_ts`` in the far past (and ``_last_paint_ts``
    likewise so the rate-limit isn't what blocks the paint) simulates "the
    child has been silent long enough — safe to paint". Asserts a single
    stdout write whose contents are the footer blob (DECSC … DECRC,
    DECSTBM reapply, marker text).
    """
    write_calls: list[bytes] = []
    real_write = os.write
    stdout_fd = sys.stdout.fileno()

    def tracking_write(fd: int, data: bytes) -> int:
        if fd == stdout_fd:
            write_calls.append(bytes(data))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", tracking_write)

    tui = _make_tui()
    # _last_activity_ts and _last_paint_ts default to -inf, so the first
    # _maybe_paint_footer call satisfies both gates without further setup.
    painted = tui._maybe_paint_footer()

    monkeypatch.setattr(os, "write", real_write)

    assert painted is True, "quiet window elapsed → paint must happen"
    assert len(write_calls) == 1
    blob = write_calls[0]
    assert blob.startswith(b"\x1b7"), "DECSC opens the paint"
    assert blob.endswith(b"\x1b8"), "DECRC closes the paint"
    assert b"\x1b[1;23r" in blob, "DECSTBM reapply (24 rows - 1 footer = 23)"
    assert b"FOOTER-MARKER" in blob


def test_on_winch_only_sets_pending_flag_does_no_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGWINCH handler must do no I/O — just flip ``_winch_pending``.

    CPython delivers signals to the main thread between bytecodes, including
    bytecodes inside the main thread's ``with self._out_lock:`` block in
    ``_drain``. If the handler tries to re-acquire ``_out_lock`` (non-reentrant)
    or do any blocking I/O, the main thread deadlocks against itself —
    freezing the input-forwarding loop. That's the "stops accepting input"
    half of the deadlock.

    The handler's only allowed effect is setting ``_winch_pending``. The
    deferred work happens from the main loop in ``_handle_winch``, where
    the lock is uncontended.
    """
    tui = _make_tui()
    tui._master_fd = 999  # non-None so the handler doesn't early-return on no-pty

    write_calls: list[tuple[int, bytes]] = []
    real_write = os.write

    def tracking_write(fd: int, data: bytes) -> int:
        write_calls.append((fd, bytes(data)))
        return real_write(fd, data)

    ioctl_calls: list[int] = []

    def tracking_ioctl(fd: int, *_args: object, **_kwargs: object) -> bytes:
        # The handler must not reach this — record and return a
        # zero-filled winsize struct in case it does, so the test
        # fails on the count assertion rather than on a kernel error.
        ioctl_calls.append(fd)
        return b"\x00" * 8

    monkeypatch.setattr(os, "write", tracking_write)
    monkeypatch.setattr(fcntl, "ioctl", tracking_ioctl)

    assert tui._winch_pending is False
    tui._on_winch(signal.SIGWINCH, None)

    assert tui._winch_pending is True, "handler must mark a resize as pending"
    assert write_calls == [], (
        "SIGWINCH handler must not write to any fd — that would risk grabbing "
        "_out_lock from the same thread that already holds it inside _drain"
    )
    assert ioctl_calls == [], (
        "SIGWINCH handler must not issue ioctls — defer ALL I/O to _handle_winch"
    )


def test_handle_winch_propagates_size_and_repaints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_handle_winch`` does the deferred resize work the signal handler used to do.

    Required steps:

    * Re-query the outer terminal size.
    * Forward the new size to the child PTY (``TIOCSWINSZ`` on master_fd) so
      the child receives its own SIGWINCH and redraws the upper region itself.
    * Reapply DECSTBM for the new shell-rows count.
    * Repaint the footer.

    Test uses a real PTY pair so the ``TIOCSWINSZ`` ioctl on the master fd
    actually succeeds.

    Deliberately does NOT issue ``CSI 2J``: clearing the user's screen on
    resize would wipe primary-screen content visible in scrollback. The
    child's own SIGWINCH redraw repaints the in-region content.
    """
    master_fd, slave_fd = pty.openpty()
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    monkeypatch.setattr(
        "switchyard.cli.launchers.shell_tui._get_winsize",
        lambda _fd: (40, 100),
    )

    write_calls: list[bytes] = []
    real_write = os.write
    stdout_fd = sys.stdout.fileno()

    def tracking_write(fd: int, data: bytes) -> int:
        if fd == stdout_fd:
            write_calls.append(bytes(data))
            return len(data)
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", tracking_write)

    try:
        tui = _make_tui(footer_height=1)
        tui._master_fd = master_fd
        tui._winch_pending = True

        tui._handle_winch()

        # Master winsize must be propagated. shell_rows = 40 - 1 footer = 39.
        ws = fcntl.ioctl(master_fd, termios.TIOCGWINSZ, b"\x00" * 8)
        rows, cols, _, _ = struct.unpack("HHHH", ws)
        assert rows == 39, f"child PTY rows = total - footer_height; got {rows}"
        assert cols == 100, f"child PTY cols = outer cols; got {cols}"
    finally:
        monkeypatch.setattr(os, "write", real_write)
        os.close(slave_fd)
        os.close(master_fd)

    stream = b"".join(write_calls)
    assert CSI + b"2J" not in stream, (
        "must NOT clear the screen — that would wipe primary-screen content "
        "the user could otherwise scroll back to. The child's own SIGWINCH "
        "redraw covers the in-region content."
    )
    assert CSI + b"1;39r" in stream, (
        "must reapply DECSTBM for the new shell-rows count (40 - 1 footer = 39)"
    )
    assert b"FOOTER-MARKER" in stream, "footer must be repainted at the new geometry"


def test_handle_winch_bumps_activity_ts_to_block_footer_thread_barge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a resize, the footer thread must not paint until the child redraws.

    The child gets its own SIGWINCH (via ``TIOCSWINSZ`` on the master) and
    will emit a redraw burst over the next several ms. If the footer thread's
    next poll lands in the middle of that burst — which is overwhelmingly
    likely for a 50 ms poll interval — its paint resets the terminal's CSI
    parser mid-frame and produces the visible-fragments corruption the
    existing ``_FOOTER_QUIET_THRESHOLD_S`` design exists to prevent.

    ``_handle_winch`` must therefore stamp ``_last_activity_ts`` to "now" so
    the footer thread's debounce defers paints for at least the quiet window.
    """
    import time as _time

    master_fd, slave_fd = pty.openpty()
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    monkeypatch.setattr(
        "switchyard.cli.launchers.shell_tui._get_winsize",
        lambda _fd: (24, 80),
    )

    real_write = os.write
    stdout_fd = sys.stdout.fileno()

    def swallow_write(fd: int, data: bytes) -> int:
        if fd == stdout_fd:
            return len(data)
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", swallow_write)

    try:
        tui = _make_tui()
        tui._master_fd = master_fd
        tui._last_activity_ts = float("-inf")

        before = _time.monotonic()
        tui._handle_winch()
        after = _time.monotonic()
    finally:
        monkeypatch.setattr(os, "write", real_write)
        os.close(slave_fd)
        os.close(master_fd)

    assert before <= tui._last_activity_ts <= after, (
        "_handle_winch must stamp _last_activity_ts to 'now' so the footer "
        "thread's quiet-window debounce blocks paints during the child's "
        "post-resize redraw burst"
    )


def test_pty_echo_path_does_not_round_trip_after_cbreak() -> None:
    """``tty.setcbreak`` on the master fd must turn off echo on the PTY.

    Reproduces the leak that produced ``[3;153R``-style fragments on the
    user's screen: when the user's terminal answers a DSR/DA query the
    child issued, the response arrives via stdin; the launcher forwards
    it to the child by writing into ``master_fd``; with the line discipline
    in default (echo + canonical) mode that write would be echoed back to
    the master's read side and re-emitted on stdout.

    Test setup mirrors what ``run()`` does after ``pty.fork``:

    * Open a PTY, drain any startup boilerplate.
    * Take a baseline by writing into the master with default modes and
      reading what comes back — the kernel echoes it (proves the failure
      mode is real and our test setup observes it).
    * Apply ``tty.setcbreak(master_fd)`` and verify a fresh write produces
      *no* echoed bytes within a short window.

    The baseline write also sanity-checks that we'd otherwise see the leak,
    so a regression that made cbreak a no-op (e.g. wrong fd, swallowed
    error) would fail this test rather than silently passing.
    """
    master_fd, slave_fd = pty.openpty()
    try:
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Drain whatever junk the kernel may have buffered on open.
        try:
            os.read(master_fd, 65536)
        except BlockingIOError:
            pass

        # Baseline: default mode echoes our writes back through master read.
        os.write(master_fd, b"baseline\n")
        echoed = b""
        for _ in range(50):  # up to ~250 ms for the line discipline to flush
            try:
                echoed += os.read(master_fd, 65536)
                break
            except BlockingIOError:
                import time
                time.sleep(0.005)
        assert b"baseline" in echoed, (
            "baseline check: PTY in default mode is supposed to echo writes back; "
            "if this assertion fails the test environment is unusual and the "
            "negative assertion below is meaningless"
        )

        # Apply the fix and confirm no echo.
        tty.setcbreak(master_fd, when=termios.TCSANOW)
        # Drain any in-flight state.
        try:
            while True:
                os.read(master_fd, 65536)
        except BlockingIOError:
            pass

        os.write(master_fd, b"after-cbreak\n")
        leaked = b""
        import time
        for _ in range(50):
            try:
                leaked += os.read(master_fd, 65536)
            except BlockingIOError:
                time.sleep(0.005)
        assert b"after-cbreak" not in leaked, (
            f"cbreak must disable echo; got leaked bytes: {leaked!r}"
        )
    finally:
        os.close(slave_fd)
        os.close(master_fd)
