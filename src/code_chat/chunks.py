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
        # EmbeddingGemma's document format, with the path as the title.
        return plain(f"title: {self.file} | text: {self.text}")


def plain(text: str) -> str:
    """Break up special tokens like <|audio|>, which code can contain and the
    model would otherwise read as a request for audio input."""
    return text.replace("<|", "< |")


def files(root: Path) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.decode()
        names = [name for name in out.split("\0") if name]
    except (subprocess.CalledProcessError, FileNotFoundError):
        names = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    return sorted(name for name in names if wanted(root, name))


def wanted(root: Path, name: str) -> bool:
    path = root / name
    if path.name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
        return False
    if language(name) is None:
        return False
    try:
        return 0 < path.stat().st_size <= MAX_FILE
    except OSError:
        return False


def language(name: str) -> str | None:
    try:
        return tslp.detect_language_from_path(name)
    except Exception:
        return None


def split(root: Path, name: str) -> list[Chunk]:
    try:
        source = (root / name).read_text()
    except (UnicodeDecodeError, OSError):
        return []
    if not source.strip():
        return []
    try:
        result = tslp.process(source, tslp.ProcessConfig(language(name), chunk_max_size=MAX_CHUNK))
        chunks = [
            # tree-sitter lines are 0-based and inclusive.
            Chunk(name, c.start_line + 1, c.end_line + 1, c.content, list(dict.fromkeys(c.metadata.symbols_defined)))
            for c in result.chunks
            if c.content.strip()
        ]
        if chunks:
            return chunks
    except Exception:
        pass
    lines = source.splitlines()
    return [
        Chunk(name, start + 1, min(start + WINDOW_LINES, len(lines)), "\n".join(lines[start : start + WINDOW_LINES]))
        for start in range(0, len(lines), WINDOW_LINES)
        if "".join(lines[start : start + WINDOW_LINES]).strip()
    ]
