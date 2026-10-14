"""The voice search daemon Neovim talks to, one JSON message per line.

    code-chat [ROOT]

In:  {"type": "listen"} / {"type": "stop"}       start or end listening
     {"type": "update", "file": "/abs/path"}     re-index a saved file
Out: {"type": "status", "text": "..."}
     {"type": "ready", "chunks": 1234}
     {"type": "listening", "on": bool} and {"type": "level", "value": 0..1}
     {"type": "results", "final": bool, "seconds": 2.5, "hits": [...]}
"""

import json
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


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    # Libraries print to stdout; keep it for messages to Neovim.
    out = sys.stdout
    sys.stdout = sys.stderr
    write = threading.Lock()

    def send(message: dict):
        with write:
            out.write(json.dumps(message) + "\n")
            out.flush()

    commands: queue.Queue = queue.Queue()

    def read():
        for line in sys.stdin:
            if line.strip():
                commands.put(json.loads(line))
        commands.put({"type": "exit"})

    threading.Thread(target=read, daemon=True).start()

    send({"type": "status", "text": "Loading EmbeddingGemma 2…"})
    search.model()
    send({"type": "status", "text": "Splitting the code into chunks…"})
    index = search.Index(root)
    index.build(lambda done, total: send({"type": "status", "text": f"Embedding {done}/{total} new chunks…"}))
    send({"type": "ready", "chunks": len(index.chunks)})

    session: voice.Session | None = None
    started = level_at = 0.0

    while True:
        try:
            command = commands.get(timeout=0.01 if session else None)
        except queue.Empty:
            command = None
        kind = command and command["type"]
        if kind == "exit":
            if session:
                session.stop()
            return
        if kind == "listen":
            if session:
                session.stop()
            try:
                session = voice.Session()
            except Exception as error:
                send({"type": "status", "text": f"Couldn't open the microphone: {error}"})
                continue
            started = time.monotonic()
            send({"type": "listening", "on": True})
        elif kind == "stop" and session:
            session.stop()
        elif kind == "update":
            try:
                index.update(str(Path(command["file"]).resolve().relative_to(root)))
            except ValueError:
                pass

        if not session:
            continue
        now = time.monotonic()
        if now - level_at > LEVEL_EVERY and not session.stopped:
            level_at = now
            send({"type": "level", "value": round(session.loudness(), 3)})
        if not session.speaking and now - started > SILENT_TIMEOUT:
            session.stop()
        if session.stopped:
            # One last pass over everything said.
            if session.speaking:
                hear(index, session, send, final=True)
            else:
                send({"type": "status", "text": "Didn't hear anything. Try again?"})
            session = None
            send({"type": "listening", "on": False})
        elif session.due():
            hear(index, session, send, final=False)


def hear(index: search.Index, session: voice.Session, send, final: bool):
    audio = session.audio()
    vector = search.embed_audio(audio)
    if session.smoothed is None:
        session.smoothed = np.full(len(index.chunks), np.nan, dtype=np.float32)
    ranked = index.spoken(vector, session.smoothed)
    send({"type": "results", "final": final, "seconds": round(len(audio) / voice.RATE, 1), "hits": index.hits(ranked)})


if __name__ == "__main__":
    main()
