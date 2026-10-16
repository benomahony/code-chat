"""Search code by describing it out loud, from any terminal or editor.

    code-chat [ROOT]           listen, show matches live, open the chosen one in
                               $VISUAL / $EDITOR at its line, or print path:line
                               when neither is set or stdout isn't a terminal
    code-chat --print [ROOT]   ... always print path:line
    code-chat --open [ROOT]    ... always open it, in vi if no editor is set
    code-chat --list [ROOT]    listen, then print every match as path:line:col: text
    code-chat serve [ROOT]     the JSON-lines engine, over stdin and stdout, for an editor
    code-chat daemon           the same engine over a socket, keeping the model loaded
    code-chat stop             stop the daemon

The live view draws on the terminal (stderr), so stdout carries only the
result: `nvim $(code-chat)`, `code -g $(code-chat)`, `vim -q <(code-chat --list)`.
Searches go through the daemon, which the first one starts, so only that one
waits for the model to load.
"""

import argparse
import json
import math
import os
import queue
import select
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO, cast

# Keep model-loading chatter out of the daemon's log.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

from rich.console import Console, Group, RenderableType  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from rich.text import Text  # noqa: E402

from code_chat import daemon, server  # noqa: E402
from code_chat.index import Hit  # noqa: E402

Message = dict[str, object]

CARD_LINES = 9
BLOCKS = "▁▂▃▄▅▆▇█"
QUIT = ("\x1b", "q", "\x03")
DOWN = ("j", "\x1b[B", "\x0e", "\t")
UP = ("k", "\x1b[A", "\x10", "\x1b[Z")
# Editors that take file:line as one argument; the rest take +line file.
COLON_EDITORS = ("zed", "subl", "hx", "helix", "micro")
GOTO_EDITORS = ("code", "code-insiders", "cursor", "windsurf", "codium")
COMMANDS = ("serve", "daemon", "stop")
DESCRIPTION = "Search code by describing it out loud, from any terminal or editor."


def main():
    assert sys.argv, "main() needs the command line in sys.argv; run it as the code-chat script"
    command = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in COMMANDS else None
    if command:
        return run_command(command, sys.argv[2:])
    parser = argparse.ArgumentParser(prog="code-chat", description=DESCRIPTION)
    parser.add_argument("root", nargs="?", help="repository to search (default: the git repo you're in)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--open", action="store_true", help="open the chosen match in $VISUAL / $EDITOR, even when stdout isn't a terminal")
    output.add_argument("--print", action="store_true", help="print the chosen match as path:line, even when an editor is set")
    output.add_argument("--list", action="store_true", help="print every match, grep style, without choosing")
    args = parser.parse_args()
    out = result_stream()
    given = cast(str | None, args.root)
    root = Path(given).resolve() if given else repository(Path.cwd().resolve())
    assert root.is_dir(), f"{root} isn't a directory; pass the repository to search"
    search = Search(root, choose=not args.list)
    hits = search.run()
    if not hits:
        # The live view is gone by now, so say why there's nothing.
        if search.reason:
            print(f"code-chat: {search.reason}", file=sys.stderr)
        sys.exit(1)
    if args.list:
        print("\n".join(grep_line(hit) for hit in hits), file=out)
    elif editor := chosen_editor(out, opening=cast(bool, args.open), printing=cast(bool, args.print)):
        out.close()
        command = editor_command(editor, hits[0]["file"], hits[0]["start"])
        os.execvp(command[0], command)
    else:
        print(f"{shown(hits[0]['file'])}:{hits[0]['start']}", file=out)


def chosen_editor(out: TextIO, opening: bool, printing: bool) -> str | None:
    """The editor to open the match in, or None to print it. An editor set in
    $VISUAL or $EDITOR opens it, unless stdout is captured, as in
    `nvim $(code-chat)`, which wants the path."""
    assert not (opening and printing), f"got --open={opening} and --print={printing}; pass at most one, as argparse enforces"
    assert not out.closed, "chosen_editor() checks whether the result stream is a terminal; call it before closing that"
    editor = (os.environ.get("VISUAL") or os.environ.get("EDITOR") or "").strip()
    if opening:
        return editor or "vi"
    if printing or not editor or not out.isatty():
        return None
    return editor


def run_command(command: str, rest: list[str]):
    """code-chat serve, daemon or stop."""
    assert command in COMMANDS, f"{command!r} isn't a command; use one of {COMMANDS}"
    assert len(rest) <= 1, f"{command} takes at most a repository, got {rest}; pass one path"
    if command == "serve":
        server.main(Path(rest[0] if rest else ".").resolve())
    elif command == "daemon":
        daemon.serve_forever()
    else:
        print("stopped" if daemon.stop() else "no daemon was running", file=sys.stderr)


def result_stream() -> TextIO:
    """Only the result goes to stdout; anything else written there, by Python
    libraries or C ones like PortAudio, goes to stderr, so `$(code-chat)`
    gets just the path."""
    assert sys.stdout is not sys.stderr, "stdout is already redirected; call result_stream() once"
    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    assert out.fileno() > 2, f"the result stream got descriptor {out.fileno()}, one of the standard three; call result_stream() once"
    return out


def repository(cwd: Path) -> Path:
    """The git repository `cwd` is in, or `cwd` itself outside one."""
    assert cwd.is_absolute(), f"working directory {cwd} must be absolute; resolve() it first"
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True, check=True)
        root = Path(top.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        root = cwd
    assert cwd.is_relative_to(root), f"repository {root} should contain the working directory {cwd}; pass the repository explicitly"
    return root


def shown(file: str) -> str:
    """Relative to where you are when it's inside, so output reads well."""
    path = Path(file)
    assert path.is_absolute(), f"hit paths come from the daemon as absolute paths, got {file!r}; send Hit['file']"
    out = str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else file
    assert out.endswith(path.name), f"shortening {file} to {out} lost the file name; keep the last path part"
    return out


def grep_line(hit: Hit) -> str:
    """path:line:col: text, as grep -n and vim's errorformat read it."""
    assert hit["start"] >= 1, f"hit in {hit['file']} starts at line {hit['start']}; send 1-based lines"
    symbols = ", ".join(hit["symbols"])
    out = f"{shown(hit['file'])}:{hit['start']}:1: {symbols + ': ' if symbols else ''}{hit['preview']}"
    assert "\n" not in out, f"grep output for {hit['file']} must be one line, got {out!r}; strip newlines from the preview"
    return out


def editor_command(editor: str, file: str, line: int) -> list[str]:
    """The command that opens `file` at `line` in `editor`."""
    assert editor.strip(), f"no editor to open with (got {editor!r}); set $VISUAL or $EDITOR"
    assert line >= 1, f"line {line} in {file} must count from 1; pass Hit['start']"
    command = editor.split()
    name = Path(command[0]).name
    if name in GOTO_EDITORS:
        return command + ["--goto", f"{file}:{line}"]
    if name in COLON_EDITORS:
        return command + [f"{file}:{line}"]
    # vi, vim, nvim, emacs, nano, kak and most others.
    return command + [f"+{line}", file]


# --- Input -------------------------------------------------------------------------


@contextmanager
def keys() -> Generator["queue.Queue[str]"]:
    """Keypresses from the terminal, without echo or waiting for Enter."""
    try:
        fd = os.open("/dev/tty", os.O_RDONLY)
    except OSError:
        # No terminal, as in a pipeline: no keys, but listening still works.
        yield queue.Queue()
        return
    assert fd >= 0, f"opening /dev/tty gave descriptor {fd}; run code-chat from a terminal"
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    pressed: queue.Queue[str] = queue.Queue()
    # Writing to this pipe tells the reader to stop, in the same select() it
    # waits for keys in, so a stop can't slip between checking and reading.
    stop_reading, stop_writing = os.pipe()
    reader = threading.Thread(target=read_keys, args=(fd, stop_reading, pressed), daemon=True)
    reader.start()
    try:
        yield pressed
    finally:
        os.write(stop_writing, b"x")
        reader.join()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        for open_fd in (fd, stop_reading, stop_writing):
            os.close(open_fd)
    assert not reader.is_alive(), "the key reader must stop before the terminal is restored; signal it through the pipe"


def read_keys(fd: int, stop: int, pressed: "queue.Queue[str]"):
    """Queue each keypress (or escape sequence) until `stop` is readable."""
    assert fd >= 0 and stop >= 0, f"read_keys() needs open descriptors, got terminal {fd} and stop pipe {stop}"
    assert fd != stop, f"the terminal and the stop pipe must be different descriptors, both are {fd}"
    ready: list[int] = []
    while stop not in ready:
        ready = select.select([fd, stop], [], [])[0]
        if fd in ready:
            pressed.put(os.read(fd, 8).decode(errors="ignore"))


def read_messages(stream: TextIO, messages: "queue.Queue[Message]"):
    """Queue each message from the daemon, then a hang-up when it closes."""
    assert stream.readable(), f"read_messages() needs the daemon's message stream, got {stream}; use daemon.connect()"
    assert messages.empty(), "read_messages() must start before any message is queued; start it once per connection"
    try:
        for line in stream:
            messages.put(cast(Message, json.loads(line)))
    except (OSError, ValueError) as error:
        messages.put({"type": "hangup", "error": str(error)})
        return
    messages.put({"type": "hangup", "error": f"the daemon closed the connection; see {daemon.LOG}"})


def drain[T](items: "queue.Queue[T]") -> list[T]:
    """Everything queued since the last call."""
    assert items.maxsize == 0, f"drain() expects an unbounded queue, got maxsize {items.maxsize}; make it with queue.Queue()"
    out: list[T] = []
    while not items.empty():
        out.append(items.get())
    nones = sum(item is None for item in out)
    assert nones == 0, f"{nones} of {len(out)} queued items are None; put a value or nothing"
    return out


# --- The search ----------------------------------------------------------------------


class Search:
    def __init__(self, root: Path, choose: bool):
        assert root.is_absolute(), f"repository root {root} must be absolute; resolve() it first"
        assert root.is_dir(), f"repository root {root} isn't a directory; pass the repository to search"
        self.root = root
        self.choose = choose
        self.console = Console(stderr=True)
        self.status = "Connecting to the code-chat daemon…"
        self.ready = False
        self.listening = False
        self.level = 0.0
        self.seconds = 0.0
        self.hits: list[Hit] = []
        self.selected = 0
        self.heights = [0.0] * 24
        self.connection: socket.socket | None = None
        # Set when the search ends: the chosen hits, or [] when cancelled,
        # with the reason when it ended without any.
        self.outcome: list[Hit] | None = None
        self.reason: str | None = None

    def run(self) -> list[Hit]:
        assert self.outcome is None, "a Search runs once; make a new one to search again"
        assert self.connection is None, "run() opens its own connection; don't connect first"
        messages: queue.Queue[Message] = queue.Queue()
        with keys() as pressed, Live(self.view(), console=self.console, transient=True, refresh_per_second=25) as live:
            live.update(self.view())
            connection, stream = daemon.connect(self.root)
            self.connection = connection
            threading.Thread(target=read_messages, args=(stream, messages), daemon=True).start()
            while self.outcome is None:
                self.handle(drain(pressed), drain(messages))
                live.update(self.view())
                time.sleep(0.03)
        # Hanging up stops the daemon listening, and frees it for the next search.
        connection.close()
        return self.outcome

    def handle(self, pressed: list[str], messages: list[Message]):
        """Act on keys, then messages, until one of them ends the search: a
        batch can hold more after that, like the final results followed by
        listening stopping."""
        assert self.outcome is None, "handle() is for a search that's still going; stop the loop when it ends"
        assert all(pressed), f"got empty keypresses {pressed}; read_keys() should only queue what it read"
        for key in pressed:
            self.press(key)
            if self.outcome is not None:
                return
        for message in messages:
            self.receive(message)
            if self.outcome is not None:
                return

    def send(self, message: Message):
        assert self.connection is not None, "send() needs the connection run() opens"
        assert message.get("type"), f"message {message} has no type; the daemon dispatches on it"
        try:
            self.connection.sendall((json.dumps(message) + "\n").encode())
        except (BrokenPipeError, ConnectionResetError) as error:
            self.status = f"Lost the daemon ({error}); see {daemon.LOG}"

    def press(self, key: str):
        assert key, "press() needs a key; skip empty reads"
        assert self.outcome is None, f"key {key!r} arrived after the search ended; stop reading keys when it does"
        if key in QUIT:
            self.outcome = []
            self.reason = None
        elif key in ("\r", "\n") and self.hits:
            self.outcome = [self.hits[self.selected]]
        elif key in DOWN and self.hits:
            self.selected = min(self.selected + 1, len(self.hits) - 1)
        elif key in UP:
            self.selected = max(self.selected - 1, 0)
        elif key == " " and self.ready and not self.listening:
            self.send({"type": "listen"})

    def receive(self, message: Message):
        """Act on one message from the daemon."""
        kind = message.get("type")
        assert isinstance(kind, str), f"message {message} has no type; the daemon must send one"
        assert self.outcome is None, f"message {kind!r} arrived after the search ended; stop reading when it does"
        if kind == "status":
            self.status = str(message["text"])
        elif kind == "ready":
            self.ready = True
            self.send({"type": "listen"})
        elif kind == "listening":
            self.listening = bool(message["on"])
            self.level = 0.0
            self.status = "Listening… describe the code  (esc cancels)" if self.listening else self.status
            # Listing has no one to press space and try again.
            if not self.listening and not self.choose and self.outcome is None:
                self.end(self.status)
        elif kind == "level":
            self.level = float(cast(float, message["value"]))
        elif kind == "results":
            self.show(cast(list[Hit], message["hits"]), float(cast(float, message["seconds"])), bool(message["final"]))
        elif kind == "hangup":
            self.end(str(message["error"]))

    def end(self, reason: str):
        """End the search without a result, saying why."""
        assert reason, "end() needs a reason to tell the user; pass the status or error"
        assert self.outcome is None, f"the search already ended; can't end it again because {reason!r}"
        self.status = self.reason = reason
        self.outcome = []

    def show(self, hits: list[Hit], seconds: float, final: bool):
        """New rankings, keeping a match you've moved to selected."""
        backwards = [(hit["path"], hit["start"], hit["end"]) for hit in hits if hit["start"] > hit["end"]]
        assert not backwards, f"hits end before they start: {backwards[:3]}; check Index.hit()"
        assert seconds >= 0, f"heard {seconds} s of speech; the daemon reports a length, which can't be negative"
        current = self.hits[self.selected] if self.hits and self.selected else None
        self.hits, self.seconds = hits, seconds
        self.selected = next(
            (i for i, h in enumerate(hits) if current and (h["file"], h["start"]) == (current["file"], current["start"])), 0
        )
        if not final:
            self.status = f"Listening… {seconds:.1f} s"
        elif self.choose:
            self.status = "Closest to what you said  (↑/↓ choose · enter · space speak again · esc)"
        else:
            self.outcome = hits

    # --- Drawing -------------------------------------------------------------

    def waveform(self) -> Text:
        """Bars driven by the microphone level, tallest in the middle and each
        wobbling at its own pace, so it reads as a voice."""
        t = time.monotonic()
        level = self.level if self.listening else 0.0
        assert 0 <= level <= 1, f"loudness {level} must be between 0 and 1; the daemon sends Session.loudness()"
        n = len(self.heights)
        bars: list[str] = []
        for i in range(n):
            x = (i - (n - 1) / 2) / (n * 0.3)
            wobble = 0.55 + 0.45 * math.sin(t * (5 + (i * 7) % 5) + i * 2.399)
            target = level * (0.15 + 0.85 * math.exp(-x * x)) * wobble
            h = self.heights[i]
            # Rise quickly, fall back gently.
            self.heights[i] = h + (target - h) * (0.5 if target > h else 0.15)
            bars.append(BLOCKS[min(len(BLOCKS) - 1, int(self.heights[i] * len(BLOCKS)))])
        assert len(bars) == n, f"drew {len(bars)} bars for a {n}-bar waveform; draw one per height"
        return Text("󰍬 " + "".join(bars) + "  ", style="green" if level else "dim")

    def view(self) -> RenderableType:
        count_hits = len(self.hits)
        assert 0 <= self.selected < max(1, count_hits), f"selection {self.selected} is outside {count_hits} hits; clamp it in press()"
        header = Panel(self.waveform() + Text(self.status, style="dim"), border_style="dim", title="code chat", title_align="left")
        if not self.hits:
            return header
        rows = self.console.size.height - 4
        count = max(1, min(count_hits, rows // (CARD_LINES + 2)))
        first = min(max(0, self.selected - count + 1), count_hits - count)
        assert first <= self.selected < first + count, f"cards {first}-{first + count} must include selection {self.selected}; fix the scroll arithmetic"
        return Group(header, *[self.card(i) for i in range(first, first + count)])

    def card(self, i: int) -> Panel:
        assert 0 <= i < len(self.hits), f"card {i} is outside {len(self.hits)} hits; only draw cards for hits"
        hit = self.hits[i]
        file, start, end = hit["file"], hit["start"], hit["end"]
        assert start <= end, f"hit in {file} ends at {end}, before it starts at {start}; check Index.hit()"
        try:
            lines = Path(file).read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            lines = [f"(can't read {file} any more)"]
        code = "\n".join(lines[start - 1 : start - 1 + CARD_LINES])
        selected = i == self.selected
        syntax = Syntax(
            code,
            Syntax.guess_lexer(file, code),
            line_numbers=True,
            start_line=start,
            highlight_lines=set(range(start, end + 1)) if selected else None,
            theme="ansi_dark",
            background_color="default",
            word_wrap=False,
        )
        symbols = f"  [bold]{', '.join(hit['symbols'])}[/]" if hit["symbols"] else ""
        best, worst = self.hits[0]["score"], self.hits[-1]["score"]
        filled = round(10 * ((hit["score"] - worst) / (best - worst) if best > worst else 1))
        return Panel(
            syntax,
            title=f"[{'reverse ' if selected else ''}bold] #{i + 1} [/] {shown(file)}:{start}{symbols}",
            title_align="left",
            subtitle=f"[green]{'▰' * filled}[/][dim]{'▱' * (10 - filled)} {hit['score']:.3f}[/]",
            subtitle_align="right",
            border_style="cyan" if selected else "dim",
            height=CARD_LINES + 2,
        )


if __name__ == "__main__":
    main()
