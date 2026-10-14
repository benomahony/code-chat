"""Split a repository into chunks of code to embed.

Files come from git (tracked and untracked, minus ignored), so the index
matches what the project considers source. Each file is cut along its syntax
tree into pieces of roughly a function or two, with line windows for files
tree-sitter can't parse.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_language_pack as tslp

# About a function or two: small enough that a hit points at one thing, big
# enough to carry what that thing does.
MAX_CHUNK = 1500
MAX_FILE = 256_000
# Fallback windows for files tree-sitter doesn't split.
WINDOW_LINES = 40
SKIP_SUFFIXES = (".lock", ".min.js", ".min.css", ".map", ".svg", ".snap")
SKIP_NAMES = {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "uv.lock", "Cargo.lock", "go.sum"}


@dataclass
class Chunk:
    file: str  # Relative to the repository root.
    start: int  # First line, 1-based.
    end: int  # Last line, 1-based and inclusive.
    text: str
    symbols: list[str] = field(default_factory=list)

    def document(self) -> str:
        assert 1 <= self.start <= self.end, f"chunk of {self.file} has lines {self.start}-{self.end}; they must count up from 1"
        assert self.file and not self.file.startswith("/"), f"chunk path {self.file!r} must be relative to the repository"
        # EmbeddingGemma's document format, with the path as the title.
        out = plain(f"title: {self.file} | text: {self.text}")
        assert "<|" not in out, f"special tokens left in the document for {self.file}; plain() must break them up"
        return out


def plain(text: str) -> str:
    """Break up special tokens like <|audio|>, which code can contain and the
    model would otherwise read as a request for audio input."""
    out = text.replace("<|", "< |")
    assert "<|" not in out, f"plain() left a special token in {out[:80]!r}"
    assert len(out) >= len(text), f"plain() shortened {len(text)} characters to {len(out)}; it only inserts them"
    return out


def files(root: Path) -> list[str]:
    assert root.is_absolute(), f"repository root {root} must be absolute, so chunk paths resolve the same everywhere"
    assert root.is_dir(), f"repository root {root} isn't a directory; pass the repository to search"
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.decode()
        names = [name for name in out.split("\0") if name]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repository, or no git: every file under the root.
        names = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    kept = sorted(name for name in names if wanted(root, name))
    absolute = [name for name in kept if name.startswith("/")]
    assert not absolute, f"file names must be relative to {root}, but got {absolute[:3]}"
    return kept


def wanted(root: Path, name: str) -> bool:
    assert root.is_absolute(), f"repository root {root} must be absolute"
    assert name and not name.startswith("/"), f"file name {name!r} must be relative to {root}"
    path = root / name
    if path.name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES) or language(name) is None:
        return False
    try:
        return 0 < path.stat().st_size <= MAX_FILE
    except OSError:
        # Listed by git but deleted, or a broken link: nothing to index.
        return False


def language(name: str) -> str | None:
    assert name, "language() needs a file name to detect a language from"
    detected = tslp.detect_language_from_path(name)
    assert detected is None or detected.strip() == detected, f"language for {name} came back as {detected!r}"
    return detected


def split(root: Path, name: str) -> list[Chunk]:
    assert root.is_absolute(), f"repository root {root} must be absolute"
    assert language(name) is not None, f"{name} has no language tree-sitter knows; files() should have skipped it"
    try:
        source = (root / name).read_text()
    except (UnicodeDecodeError, OSError):
        # Binary, unreadable or gone since it was listed: nothing to index.
        return []
    chunks = syntax_chunks(name, source) if source.strip() else []
    if source.strip() and not chunks:
        chunks = windows(name, source)
    assert all(c.file == name for c in chunks), f"every chunk of {name} must name it as its file"
    return chunks


def syntax_chunks(name: str, source: str) -> list[Chunk]:
    """Chunks cut along the syntax tree, or none if tree-sitter can't parse it."""
    assert source.strip(), f"{name} is blank; there's nothing to split"
    try:
        result = tslp.process(source, tslp.ProcessConfig(language(name), chunk_max_size=MAX_CHUNK))
    except tslp.Error:
        return []
    chunks = [
        # tree-sitter lines are 0-based and inclusive.
        Chunk(name, c.start_line + 1, c.end_line + 1, c.content, list(dict.fromkeys(c.metadata.symbols_defined)))
        for c in result.chunks
        if c.content.strip()
    ]
    assert all(c.start <= c.end for c in chunks), f"tree-sitter returned a chunk of {name} that ends before it starts"
    return chunks


def windows(name: str, source: str) -> list[Chunk]:
    """Fixed windows of lines, for files tree-sitter couldn't split."""
    lines = source.splitlines()
    assert lines, f"{name} has no lines to window"
    chunks = []
    for start in range(0, len(lines), WINDOW_LINES):
        stop = min(start + WINDOW_LINES, len(lines))
        text = "\n".join(lines[start:stop])
        if text.strip():
            chunks.append(Chunk(name, start + 1, stop, text))
    assert all(c.end - c.start < WINDOW_LINES for c in chunks), f"a window of {name} is longer than {WINDOW_LINES} lines"
    return chunks
