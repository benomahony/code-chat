"""Keep EmbeddingGemma 2 loaded between searches.

Loading the model takes several seconds; re-indexing a repository whose
vectors are cached takes a fraction of one. So `code-chat daemon` loads the
model once and serves searches over a Unix socket, one at a time, each in the
JSON-lines protocol of server.py after an opening

    {"type": "open", "root": "/abs/path/to/repo"}

It exits after IDLE seconds with no one connecting. Clients start it when
it isn't running, and restart it when code-chat's code has changed since it
started, so an editable install never serves stale code.

Whoever holds an exclusive lock on LOCK is the daemon, for as long as it
runs: taking the lock is how a daemon starts, so two can never serve at once,
and testing it is how clients tell whether one is running. The daemon writes
its pid into the lock file for people to read, and removes its own socket
when it exits, however it exits. Clients never delete anything or signal a
pid: they ask the daemon itself, over its socket, to restart or shut down.
"""

import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import NoReturn, TextIO

from code_chat import index as search
from code_chat import server

SOCKET = search.CACHE / "daemon.sock"
LOCK = search.CACHE / "daemon.lock"
LOG = search.CACHE / "daemon.log"
IDLE = 30 * 60
# macOS refuses Unix socket paths longer than this.
MAX_SOCKET_PATH = 104
# How long a client waits for a daemon to start taking connections, or to
# exit after being told to stop.
WAIT_ATTEMPTS = 100
WAIT = 0.1


def fingerprint() -> str:
    """Changes whenever code-chat's own code does. By content, so copies of
    the same code, such as Neovim's plugin and an editable install, share a
    daemon instead of restarting each other's."""
    sources = sorted(Path(__file__).parent.glob("*.py"))
    assert sources, f"found no code in {Path(__file__).parent}; reinstall code-chat"
    digest = hashlib.sha256()
    for source in sources:
        digest.update(f"{source.name}:".encode() + source.read_bytes() + b";")
    out = digest.hexdigest()[:16]
    assert len(out) == 16, f"fingerprint {out!r} should be 16 hex digits; keep the [:16] slice"
    return out


# --- Serving ---------------------------------------------------------------------

# The code this process started with, which is the code it serves, however
# the files change afterwards.
STARTED_WITH = fingerprint()


def serve_forever():
    """Serve searches until IDLE seconds pass with no one connecting, or
    exit straight away if another daemon is already serving."""
    path = str(SOCKET).encode()
    assert len(path) < MAX_SOCKET_PATH, f"socket path {SOCKET} is {len(path)} bytes, over macOS's {MAX_SOCKET_PATH}; set a shorter XDG_CACHE_HOME"
    search.CACHE.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another daemon took the lock first; it serves, this one doesn't.
            return
        # Exit through the finally below on `code-chat stop`, too.
        signal.signal(signal.SIGTERM, exit_cleanly)
        try:
            server.protect_stdout()
            claim(lock)
            listen_until_idle()
        finally:
            SOCKET.unlink(missing_ok=True)
            lock.truncate(0)
    assert not SOCKET.exists(), f"{SOCKET} outlived the daemon; delete it"


def exit_cleanly(signum: int, frame: FrameType | None) -> NoReturn:
    """On SIGTERM, exit through serve_forever()'s cleanup rather than dying."""
    assert signum == signal.SIGTERM, f"exit_cleanly() handles SIGTERM, got signal {signum}; register it for SIGTERM only"
    assert frame is None or frame.f_code.co_filename, "the interrupted frame has no file; Python passes the frame it interrupted"
    raise SystemExit(0)


def claim(lock: TextIO):
    """Record this daemon in the lock file it holds."""
    assert lock.writable(), f"{LOCK} must be open for writing to record the daemon in it; open it with 'a+'"
    lock.seek(0)
    lock.truncate()
    lock.write(f"{os.getpid()} {STARTED_WITH}")
    lock.flush()
    assert lock.tell() > 0, f"wrote nothing to {LOCK}; check {search.CACHE} has free space"


def listen_until_idle():
    """Accept searches one at a time until IDLE seconds pass without one.
    Called holding the lock, so a socket left behind is a dead daemon's."""
    assert search.CACHE.is_dir(), f"{search.CACHE} is gone; serve_forever() creates it before listening"
    assert LOCK.stat().st_size > 0, f"{LOCK} names no daemon; claim() it before listening"
    SOCKET.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(SOCKET))
        listener.listen()
        listener.settimeout(IDLE)
        idle = False
        while not idle:
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                idle = True
                continue
            with connection:
                serve_connection(connection)


def serve_connection(connection: socket.socket):
    """One search, from its opening message until the client hangs up; or a
    shutdown, or a restart when the client runs newer code than this daemon."""
    assert connection.family == socket.AF_UNIX, f"connections come over a Unix socket, got family {connection.family}; connect to {SOCKET}"
    assert SOCKET.exists(), f"serving a connection though {SOCKET} is gone; only listen_until_idle() should accept them"
    connection.settimeout(None)
    # The socket only really closes once these do too, and the client waits
    # for that to know the search is over.
    with connection.makefile("r", encoding="utf-8") as reader, connection.makefile("w", encoding="utf-8") as writer:
        answer(line=reader.readline(), reader=reader, writer=writer)


def answer(line: str, reader: TextIO, writer: TextIO):
    """Act on a connection's opening line."""
    assert reader.readable() and writer.writable(), f"answer() needs the connection's read and write sides, got {reader} and {writer}"
    assert "\n" not in line.rstrip("\n"), f"answer() takes the opening line alone, got {line[:80]!r}; pass one readline()"
    # A client that hangs up without a word.
    if not line:
        return
    try:
        opening = server.parse(line)
    except json.JSONDecodeError:
        opening = server.Command(type="unreadable")
    kind = opening["type"]
    root = Path(opening.get("root", "")).resolve()
    if kind == "shutdown":
        # Exits through serve_forever()'s cleanup.
        raise SystemExit(0)
    if kind == "open" and opening.get("fingerprint") != STARTED_WITH:
        reply(writer, {"type": "restarting"})
        raise SystemExit(0)
    if kind == "open" and root.is_dir():
        reply(writer, {"type": "opened"})
        server.Daemon(root, writer).serve(reader)
        return
    reply(writer, {"type": "status", "text": f"Start with {{\"type\": \"open\", \"root\": \"/repo\", \"fingerprint\": \"...\"}}, not {line[:80]!r}"})


def reply(writer: TextIO, message: dict[str, str]):
    """Answer a client's opening message, unless it has already hung up."""
    assert message.get("type"), f"reply {message} has no type; clients dispatch on it"
    assert writer.writable(), f"replying on a stream that isn't writable ({writer}); pass the connection's write side"
    try:
        writer.write(json.dumps(message) + "\n")
        writer.flush()
    except (BrokenPipeError, ConnectionResetError):
        # It hung up before hearing back; there's no one left to tell.
        return


# --- Clients -----------------------------------------------------------------------


def running() -> bool:
    """Whether a daemon holds the lock, tested atomically by trying to share it."""
    try:
        lock = LOCK.open("r")
    except FileNotFoundError:
        return False
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    assert lock.closed, f"{LOCK} should be closed after testing it; use it in a with block"
    assert not LOCK.is_dir(), f"{LOCK} is a directory; delete it so the daemon can lock it"
    return False


def connect(root: Path) -> tuple[socket.socket, TextIO]:
    """A connection to a daemon running this code, opened on `root`, and the
    stream of its messages. Starts a daemon if none is taking connections;
    one running older code restarts itself when it sees this code's
    fingerprint, and the next attempt starts a fresh one."""
    assert root.is_absolute(), f"repository root {root} must be absolute; resolve() it first"
    assert root.is_dir(), f"{root} isn't a directory; pass the repository to search"
    opening = json.dumps({"type": "open", "root": str(root), "fingerprint": fingerprint()}) + "\n"
    needs_start = True
    for _ in range(WAIT_ATTEMPTS):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(str(SOCKET))
        except (FileNotFoundError, ConnectionRefusedError):
            connection.close()
            # Two clients may both start one; the lock lets only the first serve.
            if needs_start:
                start()
                needs_start = False
            time.sleep(WAIT)
            continue
        connection.sendall(opening.encode())
        messages = connection.makefile("r", encoding="utf-8")
        if json.loads(messages.readline() or "{}").get("type") == "opened":
            return connection, messages
        # It's restarting for newer code, or was exiting anyway.
        messages.close()
        connection.close()
        needs_start = True
        time.sleep(WAIT)
    raise SystemExit(f"code-chat: no daemon started within {WAIT_ATTEMPTS * WAIT:.0f} s; see {LOG}")


def start():
    """Start a daemon in the background, logging to LOG. If another holds the
    lock by the time it starts, serve_forever() makes this one exit at once."""
    search.CACHE.mkdir(parents=True, exist_ok=True)
    assert search.CACHE.is_dir(), f"couldn't create {search.CACHE}; check its parent is writable"
    with LOG.open("a") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "code_chat.cli", "daemon"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=search.CACHE,
            # Its own session, so closing the terminal doesn't take it down.
            start_new_session=True,
        )
    assert process.pid > 0, f"starting the daemon gave pid {process.pid}; see {LOG}"


def stop() -> bool:
    """Ask the running daemon to shut down, over its own socket, so only it
    can receive the request. True if one was running. A daemon serving a
    search reads the request once that search ends."""
    path = str(SOCKET).encode()
    assert len(path) < MAX_SOCKET_PATH, f"socket path {SOCKET} is {len(path)} bytes, over macOS's {MAX_SOCKET_PATH}; set a shorter XDG_CACHE_HOME"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with connection:
        try:
            connection.connect(str(SOCKET))
        except (FileNotFoundError, ConnectionRefusedError):
            return False
        connection.sendall(b'{"type": "shutdown"}\n')
    for _ in range(WAIT_ATTEMPTS):
        if not running():
            break
        time.sleep(WAIT)
    else:
        raise SystemExit(f"code-chat: the daemon is still busy after {WAIT_ATTEMPTS * WAIT:.0f} s; finish its search and try again")
    # It removes its socket before letting go of the lock.
    assert not SOCKET.exists(), f"the daemon exited without removing {SOCKET}; delete it, and check {LOG}"
    return True
