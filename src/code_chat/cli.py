"""Search code by describing it out loud, from any terminal or editor.

    code-chat [ROOT]           listen, show matches live, print the chosen path:line
    code-chat --open [ROOT]    ... and open it in $VISUAL / $EDITOR at that line
    code-chat --list [ROOT]    listen, then print every match as path:line:col: text
    code-chat serve [ROOT]     the JSON-lines daemon editor plugins talk to

The live view draws on the terminal (stderr), so stdout carries only the
result: `nvim $(code-chat)`, `code -g $(code-chat)`, `vim -q <(code-chat --list)`.
"""

import argparse
import math
import os
import queue
import select
import subprocess
import sys
import termios
import threading
import time
import tty
import warnings
from contextlib import contextmanager
from pathlib import Path

# Keep model-loading chatter off the terminal the live view draws on.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

from rich.console import Console, Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from rich.text import Text  # noqa: E402

from code_chat import index as search  # noqa: E402
from code_chat import server, voice  # noqa: E402

CARD_LINES = 9
BLOCKS = "▁▂▃▄▅▆▇█"
QUIT = ("\x1b", "q", "\x03")
DOWN = ("j", "\x1b[B", "\x0e", "\t")
UP = ("k", "\x1b[A", "\x10", "\x1b[Z")
# Editors that take file:line as one argument; the rest take +line file.
COLON_EDITORS = ("zed", "subl", "hx", "helix", "micro")
GOTO_EDITORS = ("code", "code-insiders", "cursor", "windsurf", "codium")


def main():
    assert sys.argv, "main() needs the command line in sys.argv"
    if sys.argv[1:2] == ["serve"]:
        return server.main(Path(sys.argv[2] if len(sys.argv) > 2 else ".").resolve())
    parser = argparse.ArgumentParser(prog="code-chat", description=__doc__.split("\n")[0])
    parser.add_argument("root", nargs="?", help="repository to search (default: the git repo you're in)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--open", action="store_true", help="open the chosen match in $VISUAL / $EDITOR")
    output.add_argument("--list", action="store_true", help="print every match, grep style, without choosing")
    args = parser.parse_args()
    # Only the result goes to stdout; anything else written there, by Python
    # libraries or C ones like PortAudio, goes to stderr, so `$(code-chat)`
    # gets just the path.
    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    root = Path(args.root).resolve() if args.root else repository(Path.cwd().resolve())
    assert root.is_dir(), f"{root} isn't a directory; pass the repository to search"
    hits = Search(root, choose=not args.list).run()
    if not hits:
        sys.exit(1)
    if args.list:
        print("\n".join(grep_line(hit) for hit in hits), file=out)
    elif args.open:
        out.close()
        command = editor_command(os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi", hits[0]["file"], hits[0]["start"])
        os.execvp(command[0], command)
    else:
        print(f"{shown(hits[0]['file'])}:{hits[0]['start']}", file=out)


def repository(cwd: Path) -> Path:
    """The git repository `cwd` is in, or `cwd` itself outside one."""
    assert cwd.is_absolute(), f"working directory {cwd} must be absolute"
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True, check=True)
        root = Path(top.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        root = cwd
    assert cwd.is_relative_to(root), f"repository {root} should contain the working directory {cwd}"
    return root


def shown(file: str) -> str:
    """Relative to where you are when it's inside, so output reads well."""
    path = Path(file)
    assert path.is_absolute(), f"hit paths come from the daemon as absolute paths, got {file!r}"
    out = str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else file
    assert out.endswith(path.name), f"shortening {file} to {out} lost the file name"
    return out


def grep_line(hit: dict) -> str:
    """path:line:col: text, as grep -n and vim's errorformat read it."""
    assert hit["start"] >= 1, f"hit in {hit['file']} starts at line {hit['start']}; lines count from 1"
    symbols = ", ".join(hit["symbols"])
    out = f"{shown(hit['file'])}:{hit['start']}:1: {symbols + ': ' if symbols else ''}{hit['preview']}"
    assert "\n" not in out, f"grep output for {hit['file']} must be one line, got {out!r}"
    return out


def editor_command(editor: str, file: str, line: int) -> list[str]:
    """The command that opens `file` at `line` in `editor`."""
    assert editor.strip(), f"no editor to open with (got {editor!r}); set $VISUAL or $EDITOR"
    assert line >= 1, f"line {line} in {file} must count from 1"
    command = editor.split()
    name = Path(command[0]).name
    if name in GOTO_EDITORS:
        return command + ["--goto", f"{file}:{line}"]
    if name in COLON_EDITORS:
        return command + [f"{file}:{line}"]
    # vi, vim, nvim, emacs, nano, kak and most others.
    return command + [f"+{line}", file]


@contextmanager
def keys():
    """Keypresses from the terminal, without echo or waiting for Enter."""
    try:
        fd = os.open("/dev/tty", os.O_RDONLY)
    except OSError:
        # No terminal, as in a pipeline: no keys, but listening still works.
        yield queue.Queue()
        return
    assert fd >= 0, f"opening /dev/tty gave descriptor {fd}"
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    pressed: queue.Queue = queue.Queue()
    done = threading.Event()
    reader = threading.Thread(target=read_keys, args=(fd, pressed, done), daemon=True)
    reader.start()
    try:
        yield pressed
    finally:
        done.set()
        reader.join()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        os.close(fd)
    assert not reader.is_alive(), "the key reader must stop before the terminal is restored"


def read_keys(fd: int, pressed: queue.Queue, done: threading.Event):
    """Queue each keypress (or escape sequence) until `done` is set."""
    assert fd >= 0, f"read_keys() needs an open terminal, got descriptor {fd}"
    assert not done.is_set(), "read_keys() started after it was told to stop; start it before setting done"
    while not done.is_set():
        if select.select([fd], [], [], 0.05)[0]:
            pressed.put(os.read(fd, 8).decode(errors="ignore"))


def drain(pressed: queue.Queue) -> list[str]:
    """Every key pressed since the last call."""
    assert pressed is not None, "drain() needs the key queue keys() yields"
    out = []
    while not pressed.empty():
        out.append(pressed.get())
    assert all(out), f"got empty keypresses {out}; read_keys() should only queue what it read"
    return out


class Search:
    def __init__(self, root: Path, choose: bool):
        assert root.is_absolute(), f"repository root {root} must be absolute"
        assert root.is_dir(), f"repository root {root} isn't a directory"
        self.root = root
        self.choose = choose
        self.console = Console(stderr=True)
        self.status = "Loading EmbeddingGemma 2…"
        self.index: search.Index | None = None
        self.hits: list[dict] = []
        self.selected = 0
        self.session: voice.Session | None = None
        self.closing: threading.Thread | None = None
        self.listened = False
        self.started = 0.0
        self.seconds = 0.0
        self.heights = [0.0] * 24
        # Set when the search ends: the chosen hits, or [] when cancelled.
        self.outcome: list[dict] | None = None

    def run(self) -> list[dict]:
        assert self.outcome is None, "a Search runs once; make a new one to search again"
        assert self.index is None, "run() loads the index itself"
        with keys() as pressed, Live(self.view(), console=self.console, transient=True, refresh_per_second=25) as live:
            threading.Thread(target=self.load, daemon=True).start()
            while self.outcome is None:
                for key in drain(pressed):
                    self.press(key)
                # Start listening as soon as the index is ready.
                if self.index and not self.listened:
                    self.listen()
                self.step()
                live.update(self.view())
                if not self.session:
                    time.sleep(0.04)
        if self.session:
            self.closing = self.session.stop()
        # Let the microphone close before the process exits under it.
        if self.closing:
            self.closing.join(timeout=2)
        return self.outcome

    def load(self):
        assert self.index is None, "the index is already loaded"
        search.model()
        self.status = "Splitting the code into chunks…"
        index = search.Index(self.root)
        index.build(lambda done, total: setattr(self, "status", f"Embedding {done}/{total} new chunks…"))
        assert index.chunks, f"found no code to search in {self.root}; is it the right directory?"
        self.index = index

    def press(self, key: str):
        assert key, "press() needs a key"
        assert self.outcome is None, f"key {key!r} arrived after the search ended"
        if key in QUIT:
            self.outcome = []
        elif key in ("\r", "\n") and self.hits:
            self.outcome = [self.hits[self.selected]]
        elif key in DOWN and self.hits:
            self.selected = min(self.selected + 1, len(self.hits) - 1)
        elif key in UP:
            self.selected = max(self.selected - 1, 0)
        elif key == " " and self.index and not self.session:
            self.listen()

    def listen(self):
        assert self.index is not None, "listen() needs the index load() builds"
        assert self.session is None, "already listening; let this session finish first"
        self.listened = True
        try:
            self.session = voice.Session()
        except voice.MicrophoneError as error:
            self.status = f"Couldn't open the microphone ({error}); check it's connected and allowed"
            return
        self.started = time.monotonic()
        self.status = "Listening… describe the code  (esc cancels)"

    def step(self):
        """Re-rank when there's new speech; finish after a pause."""
        session = self.session
        if not session:
            return
        assert self.index is not None, "step() needs the index to rank against"
        assert self.started > 0, f"step() needs the time listening started, got {self.started}"
        if not session.speaking and time.monotonic() - self.started > server.SILENT_TIMEOUT:
            session.stop()
        if session.stopped:
            self.finish(session)
        elif session.due():
            self.update(session)
            self.status = f"Listening… {self.seconds:.1f} s"

    def finish(self, session: voice.Session):
        assert session.stopped, "finish() is for a session that has stopped listening"
        assert session is self.session, "finish() is for the current session"
        self.closing = session.stop()
        self.session = None
        if not session.speaking:
            self.status = "Didn't hear anything. Space to try again, esc to quit."
            return
        self.update(session)
        self.status = "Closest to what you said  (↑/↓ choose · enter · space speak again · esc)"
        if not self.choose:
            self.outcome = self.hits

    def update(self, session: voice.Session):
        assert self.index is not None, "update() needs the index to rank against"
        assert session.speaking, "update() needs speech to rank against"
        current = self.hits[self.selected] if self.hits and self.selected else None
        ranked, self.seconds = server.rank(self.index, session)
        self.hits = self.index.hits(ranked)
        # Keep a match you've moved to selected while results keep moving.
        self.selected = next(
            (i for i, h in enumerate(self.hits) if current and (h["file"], h["start"]) == (current["file"], current["start"])), 0
        )

    # --- Drawing -------------------------------------------------------------

    def waveform(self) -> Text:
        """Bars driven by the microphone level, tallest in the middle and each
        wobbling at its own pace, so it reads as a voice."""
        t = time.monotonic()
        level = self.session.loudness() if self.session and not self.session.stopped else 0
        assert 0 <= level <= 1, f"loudness {level} must be between 0 and 1"
        n = len(self.heights)
        bars = []
        for i in range(n):
            x = (i - (n - 1) / 2) / (n * 0.3)
            wobble = 0.55 + 0.45 * math.sin(t * (5 + (i * 7) % 5) + i * 2.399)
            target = level * (0.15 + 0.85 * math.exp(-x * x)) * wobble
            h = self.heights[i]
            # Rise quickly, fall back gently.
            self.heights[i] = h + (target - h) * (0.5 if target > h else 0.15)
            bars.append(BLOCKS[min(len(BLOCKS) - 1, int(self.heights[i] * len(BLOCKS)))])
        assert len(bars) == n, f"drew {len(bars)} bars for a {n}-bar waveform"
        return Text("󰍬 " + "".join(bars) + "  ", style="green" if level else "dim")

    def view(self):
        count_hits = len(self.hits)
        assert 0 <= self.selected < max(1, count_hits), f"selection {self.selected} is outside {count_hits} hits"
        header = Panel(self.waveform() + Text(self.status, style="dim"), border_style="dim", title="code chat", title_align="left")
        if not self.hits:
            return header
        rows = self.console.size.height - 4
        count = max(1, min(count_hits, rows // (CARD_LINES + 2)))
        first = min(max(0, self.selected - count + 1), count_hits - count)
        assert first <= self.selected < first + count, f"cards {first}-{first + count} must include selection {self.selected}"
        return Group(header, *[self.card(i) for i in range(first, first + count)])

    def card(self, i: int) -> Panel:
        assert 0 <= i < len(self.hits), f"card {i} is outside {len(self.hits)} hits"
        hit = self.hits[i]
        file, start, end = hit["file"], hit["start"], hit["end"]
        assert start <= end, f"hit in {file} ends at {end}, before it starts at {start}"
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
