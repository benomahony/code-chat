"""The voice search engine editors talk to, one JSON message per line.

    code-chat serve [ROOT]     over stdin and stdout, for one editor session
    code-chat daemon           over a Unix socket, shared and kept warm (see daemon.py)

In:  {"type": "listen"} / {"type": "stop"}       start or end listening
     {"type": "update", "file": "/abs/path"}     re-index a saved file
     {"type": "exit"}                            stop serving (as does hanging up)
Out: {"type": "status", "text": "..."}
     {"type": "ready", "chunks": 1234}
     {"type": "listening", "on": bool} and {"type": "level", "value": 0..1}
     {"type": "results", "final": bool, "seconds": 2.5, "hits": [...]}
"""

import json
import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import NotRequired, TextIO, TypedDict, cast

import numpy as np

from code_chat import index as search
from code_chat import voice

LEVEL_EVERY = 0.05
# How often to check the microphone while listening.
TICK = 0.01


class Command(TypedDict):
    type: str
    file: NotRequired[str]
    root: NotRequired[str]
    fingerprint: NotRequired[str]


def parse(line: str) -> Command:
    """A command from one line of JSON."""
    data: object = json.loads(line)
    assert isinstance(data, dict), f"each line must be a JSON object, got {line[:80]!r}; send one object per line"
    command = cast(Command, data)
    assert isinstance(command.get("type"), str), f"command {line[:80]!r} has no \"type\" string; add one"
    return command


def rank(index: search.Index, session: voice.Session, audio: voice.Audio) -> tuple[list[tuple[int, float]], float]:
    """Re-rank against everything said so far: (chunks, seconds heard)."""
    assert index.chunks, f"nothing is indexed in {index.root}, so there's nothing to rank; search a repository with code in it"
    assert len(audio) > 0, f"rank() got {len(audio)} samples; pass what Session.take() returned"
    if session.smoothed is None:
        session.smoothed = np.full(len(index.chunks), np.nan, dtype=np.float32)
    ranked = index.spoken(search.embed_audio(audio), session.smoothed)
    return ranked, round(len(audio) / voice.RATE, 1)


class Daemon:
    """Serves one editor's searches.

    The thread running serve() does the work. The reader thread only puts
    commands on the queue; send() writes under its own lock; the session is
    handed over under another, and stopping is an Event, so no check and the
    action it decides can be split by another thread.
    """

    def __init__(self, root: Path, out: TextIO):
        assert root.is_absolute(), f"repository root {root} must be absolute; resolve() it first"
        assert out is not sys.stderr, "protocol messages can't share stderr with logs; pass stdout or a socket"
        self.root = root
        self.out = out
        self.write = threading.Lock()
        self.commands: queue.Queue[Command] = queue.Queue()
        self.done = threading.Event()
        self.index: search.Index | None = None
        self.state = threading.Lock()
        self.session: voice.Session | None = None
        self.level_at = 0.0
        self.handlers: dict[str, Callable[[Command], None]] = {
            "exit": self.exit,
            "listen": self.listen,
            "stop": self.stop,
            "update": self.update,
        }

    def current(self) -> voice.Session | None:
        """The session listening now, if any."""
        with self.state:
            session = self.session
        assert session is None or not self.done.is_set() or session.closing is not None, f"the daemon is done but session {session} is still listening; exit() must stop it"
        assert self.index is not None or session is None, f"session {session} started before the index was built; listen() needs the index"
        return session

    def swap(self, new: voice.Session | None, expected: voice.Session | None = None) -> voice.Session | None:
        """Make `new` the session and return the one it replaces; with
        `expected`, only if that's still the session."""
        assert new is None or new is not expected, f"swapping session {new} for itself does nothing; pass a different one"
        assert new is None or self.index is not None, "a new session needs the index; build it in serve() first"
        with self.state:
            old = self.session
            if expected is None or old is expected:
                self.session = new
        return old

    def send(self, message: Mapping[str, object]):
        assert message.get("type"), f"message {message} has no type; editors dispatch on it"
        line = json.dumps(message)
        assert "\n" not in line, f"message {line[:80]!r} spans lines; the protocol is one message per line"
        with self.write:
            try:
                self.out.write(line + "\n")
                self.out.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The editor or terminal hung up; stop serving it.
                self.commands.put({"type": "exit"})

    def read(self, lines: Iterable[str]):
        """Queue each command from the editor, then exit when it hangs up."""
        assert self.commands.empty(), "read() must start before any command is queued; serve() starts it first"
        assert not self.done.is_set(), "read() is for a daemon that's serving; call serve()"
        try:
            for line in lines:
                self.receive(line)
        except (OSError, ValueError):
            # The connection closed under the reader: the same as hanging up.
            self.commands.put({"type": "exit"})
            return
        self.commands.put({"type": "exit"})

    def receive(self, line: str):
        """Queue one line's command, or say why it can't be read. Runs on the
        reader thread, so it only touches the queue and send()."""
        assert "\n" not in line.rstrip("\n"), f"receive() takes one line, got {line[:80]!r}; split the input into lines"
        assert len(line) < 1_000_000, f"a {len(line)}-character command is too long; commands are small JSON objects"
        if not line.strip():
            return
        try:
            self.commands.put(parse(line))
        except json.JSONDecodeError as error:
            self.send({"type": "status", "text": f"Couldn't read {line[:80]!r} ({error}); send one JSON object per line"})

    def serve(self, lines: Iterable[str]):
        assert not self.done.is_set(), "a daemon that has exited can't serve again; make a new one"
        assert self.index is None, "serve() builds the index; call it once"
        threading.Thread(target=self.read, args=(lines,), daemon=True).start()
        self.send({"type": "status", "text": "Loading EmbeddingGemma 2…"})
        search.model()
        self.send({"type": "status", "text": "Splitting the code into chunks…"})
        index = search.Index(self.root)
        index.build(lambda done, total: self.send({"type": "status", "text": f"Embedding {done}/{total} new chunks…"}))
        self.index = index
        self.send({"type": "ready", "chunks": len(index.chunks)})
        while not self.done.is_set():
            command = self.next_command()
            if command:
                self.handlers.get(command["type"], self.unknown)(command)
            session = self.current()
            if session:
                self.tick(session, index)

    def next_command(self) -> Command | None:
        """The next command: waits for one when idle, but only briefly while
        listening, so the microphone keeps being checked."""
        assert not self.done.is_set(), "no commands are read after exit; check done first"
        assert self.index is not None, "commands are handled once the index is built; build it in serve()"
        try:
            return self.commands.get(timeout=TICK if self.current() else None)
        except queue.Empty:
            return None

    def unknown(self, command: Command):
        assert command["type"] not in self.handlers, f"{command['type']} has a handler; dispatch through self.handlers"
        assert not self.done.is_set(), "commands are only answered while serving; check done first"
        self.send({"type": "status", "text": f"Unknown command {command['type']!r}; send one of {sorted(self.handlers)}"})

    def exit(self, command: Command):
        assert command["type"] == "exit", f"exit() got a {command['type']} command; dispatch through self.handlers"
        assert not self.done.is_set(), "the daemon has already exited; don't queue a second exit"
        session = self.swap(None)
        if session:
            # Let the microphone close before the process exits under it.
            session.stop().join(timeout=2)
        self.done.set()

    def listen(self, command: Command):
        assert command["type"] == "listen", f"listen() got a {command['type']} command; dispatch through self.handlers"
        assert self.index is not None, f"listen() on {self.root} needs the index; it's built in serve()"
        previous = self.swap(None)
        if previous:
            previous.stop()
        try:
            session = voice.Session()
        except voice.MicrophoneError as error:
            self.send({"type": "status", "text": f"Couldn't open the microphone ({error}); check it's connected and allowed"})
            return
        self.swap(session)
        self.send({"type": "listening", "on": True})

    def stop(self, command: Command):
        assert command["type"] == "stop", f"stop() got a {command['type']} command; dispatch through self.handlers"
        assert not self.done.is_set(), "stop() only applies while serving; check done first"
        session = self.current()
        # Stopping twice is safe, so a session that stops itself meanwhile is fine.
        if session:
            session.stop()

    def update(self, command: Command):
        assert command["type"] == "update", f"update() got a {command['type']} command; dispatch through self.handlers"
        assert self.index is not None, "update() needs the index; it's built in serve()"
        file = Path(command.get("file", "")).resolve()
        # Saves outside the repository don't change its index.
        if file.is_relative_to(self.root) and file != self.root:
            self.index.update(str(file.relative_to(self.root)))

    def tick(self, session: voice.Session, index: search.Index):
        """Report the level, re-rank when there's new speech, finish once
        listening stops."""
        assert index is self.index, "tick() ranks against this daemon's index; pass self.index"
        assert index.chunks, f"nothing is indexed in {self.root}; search a repository with code in it"
        taken = session.take()
        if taken:
            audio, final = taken
            ranked, seconds = rank(index, session, audio)
            self.send({"type": "results", "final": final, "seconds": seconds, "hits": index.hits(ranked)})
        speaking, done = session.state()
        now = time.monotonic()
        if done:
            if not speaking:
                self.send({"type": "status", "text": "Didn't hear anything. Try again?"})
            # Only if no new session has started since.
            self.swap(None, expected=session)
            self.send({"type": "listening", "on": False})
        elif now - self.level_at > LEVEL_EVERY:
            self.level_at = now
            self.send({"type": "level", "value": round(session.loudness(), 3)})


def main(root: Path):
    assert root.is_absolute(), f"repository root {root} must be absolute; resolve() it first"
    assert root.is_dir(), f"{root} isn't a directory; pass the repository to search"
    Daemon(root, protect_stdout()).serve(sys.stdin)


_protected: TextIO | None = None
_protecting = threading.Lock()


def protect_stdout() -> TextIO:
    """A private copy of stdout for protocol messages. Everything else
    written there, by Python libraries or C ones like PortAudio, goes to
    stderr instead, so it can't break the protocol. Calling it again returns
    the same copy."""
    global _protected
    with _protecting:
        if _protected is None:
            sys.stdout.flush()
            _protected = os.fdopen(os.dup(1), "w", buffering=1)
            os.dup2(2, 1)
            sys.stdout = sys.stderr
        out = _protected
    assert out.fileno() > 2, f"the protocol stream got descriptor {out.fileno()}, one of the standard three; check descriptor 1 was open"
    assert sys.stdout is sys.stderr, "Python's stdout should now be stderr; only protect_stdout() should change sys.stdout"
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve())
