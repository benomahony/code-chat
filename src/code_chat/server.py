"""The voice search daemon editors talk to, one JSON message per line.

    code-chat serve [ROOT]

In:  {"type": "listen"} / {"type": "stop"}       start or end listening
     {"type": "update", "file": "/abs/path"}     re-index a saved file
     {"type": "exit"}                            stop the daemon (as does closing stdin)
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
from pathlib import Path

import numpy as np

from code_chat import index as search
from code_chat import voice

# Give up listening if nobody speaks within this long.
SILENT_TIMEOUT = 8.0
LEVEL_EVERY = 0.05
# How often to check the microphone while listening.
TICK = 0.01


def rank(index: search.Index, session: voice.Session) -> tuple[list[tuple[int, float]], float]:
    """Re-rank against everything said so far: (chunks, seconds heard)."""
    assert index.chunks, f"nothing is indexed in {index.root}, so there's nothing to rank"
    assert session.speaking, "rank() needs speech; check Session.due() first"
    audio = session.audio()
    if session.smoothed is None:
        session.smoothed = np.full(len(index.chunks), np.nan, dtype=np.float32)
    ranked = index.spoken(search.embed_audio(audio), session.smoothed)
    assert len(ranked) == min(search.LIMIT, len(index.chunks)), f"ranked {len(ranked)} of {len(index.chunks)} chunks; Index.spoken() should return up to LIMIT"
    return ranked, round(len(audio) / voice.RATE, 1)


class Daemon:
    def __init__(self, root: Path, out):
        assert root.is_absolute(), f"repository root {root} must be absolute"
        assert out is not sys.stderr, "messages go to stdout; stderr is for logs"
        self.root = root
        self.out = out
        self.write = threading.Lock()
        self.commands: queue.Queue = queue.Queue()
        self.running = True
        self.session: voice.Session | None = None
        self.started = self.level_at = 0.0
        self.handlers = {"exit": self.exit, "listen": self.listen, "stop": self.stop, "update": self.update}

    def send(self, message: dict):
        assert message.get("type"), f"message {message} has no type for the editor to dispatch on"
        line = json.dumps(message)
        assert "\n" not in line, "a message must fit on one line of the JSON-lines protocol"
        with self.write:
            self.out.write(line + "\n")
            self.out.flush()

    def read(self, lines):
        """Queue each command from the editor, then exit when it hangs up."""
        assert hasattr(lines, "__iter__"), f"read() needs lines to iterate, got {type(lines).__name__}"
        assert self.commands.empty(), "read() must start before any command is queued"
        for line in lines:
            if line.strip():
                self.commands.put(json.loads(line))
        self.commands.put({"type": "exit"})

    def serve(self, lines):
        assert self.running, "a daemon that has exited can't serve again"
        threading.Thread(target=self.read, args=(lines,), daemon=True).start()
        self.send({"type": "status", "text": "Loading EmbeddingGemma 2…"})
        search.model()
        self.send({"type": "status", "text": "Splitting the code into chunks…"})
        self.index = search.Index(self.root)
        self.index.build(lambda done, total: self.send({"type": "status", "text": f"Embedding {done}/{total} new chunks…"}))
        self.send({"type": "ready", "chunks": len(self.index.chunks)})
        while self.running:
            command = self.next_command()
            if command:
                self.handlers.get(command["type"], self.unknown)(command)
            if self.session:
                self.tick(self.session)
        assert self.session is None, "the microphone must be closed before the daemon exits"

    def next_command(self) -> dict | None:
        """The next command: waits for one when idle, but only briefly while
        listening, so the microphone keeps being checked."""
        assert self.running, "no commands are read after exit"
        try:
            command = self.commands.get(timeout=TICK if self.session else None)
        except queue.Empty:
            return None
        assert "type" in command, f"command {command} has no type; send one of {sorted(self.handlers)}"
        return command

    def unknown(self, command: dict):
        assert command["type"] not in self.handlers, f"{command['type']} has a handler; dispatch to it"
        assert self.running, "commands are only answered while running"
        self.send({"type": "status", "text": f"Unknown command {command['type']!r}; send one of {sorted(self.handlers)}"})

    def exit(self, command: dict):
        assert command["type"] == "exit", f"exit() got a {command['type']} command; dispatch through self.handlers"
        assert self.running, "the daemon has already exited"
        if self.session:
            # Let the microphone close before the process exits under it.
            self.session.stop().join(timeout=2)
            self.session = None
        self.running = False

    def listen(self, command: dict):
        assert command["type"] == "listen", f"listen() got a {command['type']} command; dispatch through self.handlers"
        assert getattr(self, "index", None) is not None, f"listen() on {self.root} needs the index serve() builds; call serve() first"
        if self.session:
            self.session.stop()
        try:
            self.session = voice.Session()
        except voice.MicrophoneError as error:
            self.session = None
            self.send({"type": "status", "text": f"Couldn't open the microphone ({error}); check it's connected and allowed"})
            return
        self.started = time.monotonic()
        self.send({"type": "listening", "on": True})

    def stop(self, command: dict):
        assert command["type"] == "stop", f"stop() got a {command['type']} command; dispatch through self.handlers"
        assert self.running, "stop() only applies while running"
        if self.session:
            self.session.stop()

    def update(self, command: dict):
        assert command["type"] == "update", f"update() got a {command['type']} command; dispatch through self.handlers"
        file = Path(command["file"]).resolve()
        assert file.is_absolute(), f"update needs an absolute path, got {command['file']!r}"
        # Saves outside the repository don't change its index.
        if file.is_relative_to(self.root):
            self.index.update(str(file.relative_to(self.root)))

    def tick(self, session: voice.Session):
        """Report the level, re-rank when there's new speech, finish after a pause."""
        assert session is self.session, "tick() is for the current session"
        assert not session.stopped or session.stream is not None, "a stopped session must still have its stream"
        now = time.monotonic()
        if now - self.level_at > LEVEL_EVERY and not session.stopped:
            self.level_at = now
            self.send({"type": "level", "value": round(session.loudness(), 3)})
        if not session.speaking and now - self.started > SILENT_TIMEOUT:
            session.stop()
        if session.stopped:
            self.finish(session)
        elif session.due():
            self.hear(session, final=False)

    def finish(self, session: voice.Session):
        assert session.stopped, "finish() is for a session that has stopped listening"
        assert self.session is session, "finish() is for the current session"
        if session.speaking:
            # One last pass over everything said.
            self.hear(session, final=True)
        else:
            self.send({"type": "status", "text": "Didn't hear anything. Try again?"})
        self.session = None
        self.send({"type": "listening", "on": False})

    def hear(self, session: voice.Session, final: bool):
        assert session.speaking, "hear() needs speech to rank against"
        assert final == session.stopped, f"final ({final}) must match whether listening stopped ({session.stopped})"
        ranked, seconds = rank(self.index, session)
        self.send({"type": "results", "final": final, "seconds": seconds, "hits": self.index.hits(ranked)})


def main(root: Path):
    assert root.is_absolute(), f"repository root {root} must be absolute"
    assert root.is_dir(), f"{root} isn't a directory; pass the repository to search"
    # Messages to the editor get a private copy of stdout. Everything else
    # written there, by Python libraries or C ones like PortAudio, goes to
    # stderr instead, so it can't break the protocol.
    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    Daemon(root, out).serve(sys.stdin)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve())
