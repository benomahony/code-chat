"""Embed a repository's chunks as text, and rank them against speech.

The same scheme as the voice search on benomahony.com: what someone has said
so far is embedded straight from the audio, with no transcription, and chunks
rank by cosine, smoothed across live updates so results only move when
they're consistently better.
"""

import hashlib
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import numpy.typing as npt

from code_chat.chunks import Chunk, files, split, wanted

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

Vectors = npt.NDArray[np.float32]
Progress = Callable[[int, int], None]

MODEL = "google/embeddinggemma-2"
DIMS = 768
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "code-chat"
LIMIT = 30
SMOOTHING = 0.4  # Weight of the newest update.


class Hit(TypedDict):
    """A ranked chunk, as editors receive it."""

    file: str  # Absolute.
    path: str  # Relative to the repository.
    start: int  # 1-based, inclusive.
    end: int
    end_col: int  # Byte column where the chunk ends on its last line.
    symbols: list[str]
    preview: str
    score: float


# --- Model -------------------------------------------------------------------


_model: "SentenceTransformer | None" = None
_loading = threading.Lock()


def model() -> "SentenceTransformer":
    """EmbeddingGemma 2, loaded once per process however many threads ask."""
    global _model
    with _loading:
        if _model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            _model = SentenceTransformer(
                MODEL,
                device=device,
                # Code as text, queries as speech; leave out the image encoder.
                config_kwargs={"vision_config": None},
                # bfloat16 on Apple GPUs; float16 overflows this model.
                model_kwargs={"torch_dtype": "bfloat16" if device == "mps" else "float32"},
            )
        loaded = _model
    dims = loaded.get_embedding_dimension()
    assert dims == DIMS, f"{MODEL} gives {dims}-dimensional vectors but the cache stores {DIMS}; set DIMS to match"
    assert (loaded.max_seq_length or 0) > 0, f"{MODEL} loaded with input length {loaded.max_seq_length}; delete it from the Hugging Face cache to download it again"
    return loaded


def unit(vectors: Vectors) -> Vectors:
    norms: Vectors = np.sqrt(np.sum(vectors * vectors, axis=-1, keepdims=True))
    assert np.all(norms > 0), f"{np.count_nonzero(norms == 0)} zero vectors can't be normalised; drop empty inputs before embedding"
    out: Vectors = vectors / norms
    assert out.shape == vectors.shape, f"normalising changed the shape from {vectors.shape} to {out.shape}; norms must keep their last axis"
    return out


def embed(inputs: Sequence[str | dict[str, npt.NDArray[np.float32]]]) -> Vectors:
    assert inputs, "embed() needs at least one input; skip the call when there's nothing to embed"
    # encode()'s overloads include types sentence-transformers leaves unknown,
    # so call it through a signature naming what code-chat gets back.
    encode = cast(Callable[..., npt.NDArray[np.float32]], getattr(model(), "encode"))
    vectors = unit(encode(list(inputs), batch_size=8, convert_to_numpy=True).astype(np.float32))
    assert vectors.shape == (len(inputs), DIMS), f"embedded {len(inputs)} inputs into shape {vectors.shape}; check the model is {MODEL}"
    return vectors


def embed_audio(audio: npt.NDArray[np.float32]) -> Vectors:
    assert audio.ndim == 1, f"audio must be mono samples, got shape {audio.shape}; take one channel"
    assert len(audio) > 0, f"there's no audio to embed (got {len(audio)} samples); wait for Session.take()"
    return embed([{"audio": audio}])[0]


def key(chunk: Chunk) -> str:
    """Cache key: vectors are cached by the content they embed."""
    document = chunk.document()
    assert document, f"chunk of {chunk.file} has an empty document; split() should drop blank chunks"
    out = hashlib.sha256((MODEL + document).encode()).hexdigest()[:32]
    assert len(out) == 32, f"cache key {out!r} should be 32 hex digits; keep the [:32] slice"
    return out


def first_line(text: str) -> str:
    assert text.strip(), "a chunk with no text has no first line to preview; split() should drop blank chunks"
    out = next(line.strip() for line in text.splitlines() if line.strip())[:120]
    assert 0 < len(out) <= 120, f"preview {out!r} must be 1 to 120 characters; keep the [:120] slice"
    return out


# --- Index -------------------------------------------------------------------


class Index:
    def __init__(self, root: Path):
        assert root.is_absolute(), f"repository root {root} must be absolute; resolve() it first"
        assert root.is_dir(), f"repository root {root} isn't a directory; pass the repository to search"
        self.root = root
        self.cache_file = CACHE / f"{hashlib.sha256(str(root).encode()).hexdigest()[:16]}.npz"
        # Guards the cache and the index, which a build and an update could
        # otherwise change at once.
        self.lock = threading.Lock()
        self.cache: dict[str, Vectors] = {}
        self.chunks: list[Chunk] = []
        self.vectors: Vectors = np.zeros((0, DIMS), dtype=np.float32)

    def _load_cache(self):
        """Fill the in-memory cache from disk, once. Called with the lock held."""
        assert self.lock.locked(), "_load_cache() changes the cache; call it with self.lock held"
        if self.cache:
            return
        try:
            data = np.load(self.cache_file)
        except FileNotFoundError:
            return
        keys, vectors = data["keys"], data["vectors"]
        assert len(keys) == len(vectors), f"{self.cache_file} has {len(keys)} keys for {len(vectors)} vectors; delete it to rebuild"
        self.cache = dict(zip(keys.tolist(), vectors.astype(np.float32)))

    def _save_cache(self, keep: set[str]):
        """Write the cache, keeping only `keep`. Called with the lock held."""
        assert self.lock.locked(), "_save_cache() changes the cache; call it with self.lock held"
        self.cache = {k: v for k, v in self.cache.items() if k in keep}
        shapes = {v.shape for v in self.cache.values()}
        assert shapes <= {(DIMS,)}, f"cached vectors must all have {DIMS} dimensions, found {shapes}; delete {self.cache_file} to rebuild"
        CACHE.mkdir(parents=True, exist_ok=True)
        keys = list(self.cache)
        vectors = np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, DIMS))
        np.savez(self.cache_file, keys=np.array(keys), vectors=vectors.astype(np.float16))

    def _vectors(self, chunks: list[Chunk], progress: Progress | None = None) -> Vectors:
        """Vectors for `chunks`, embedding only those not already cached.
        Called with the lock held."""
        assert self.lock.locked(), "_vectors() changes the cache; call it with self.lock held"
        self._load_cache()
        keys = [key(c) for c in chunks]
        missing = [(k, c) for k, c in zip(keys, chunks) if k not in self.cache]
        total = len(missing)
        batch = 32
        for start in range(0, total, batch):
            part = missing[start : start + batch]
            for (k, _), v in zip(part, embed([c.document() for _, c in part])):
                self.cache[k] = v
            if progress:
                progress(min(start + batch, total), total)
        if missing:
            self._save_cache(set(keys) | set(self.cache))
        out = np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, DIMS), dtype=np.float32)
        assert len(out) == len(chunks), f"{len(out)} vectors for {len(chunks)} chunks; every missing chunk must be embedded above"
        return out

    def build(self, progress: Progress | None = None):
        chunks = [c for name in files(self.root) for c in split(self.root, name)]
        with self.lock:
            self._set(chunks, self._vectors(chunks, progress))
            # Drop vectors for code that no longer exists.
            self._save_cache({key(c) for c in chunks})
            cached = len(self.cache)
        assert len(self.chunks) == len(chunks), f"indexed {len(self.chunks)} of {len(chunks)} chunks; _set() must take them all"
        assert cached <= len(chunks), f"cache holds {cached} vectors for {len(chunks)} chunks; _save_cache() should drop the rest"

    def update(self, name: str):
        """Re-split and re-embed one file, after it's saved."""
        assert not name.startswith("/"), f"{name} must be relative to {self.root}; pass the path from the repository root"
        # wanted() is False for a deleted file, and split() returns nothing for
        # one deleted after it, so either way only the kept chunks remain.
        fresh = split(self.root, name) if wanted(self.root, name) else []
        with self.lock:
            kept = [(c, v) for c, v in zip(self.chunks, self.vectors) if c.file != name]
            vectors = self._vectors(fresh)
            old = np.stack([v for _, v in kept]) if kept else np.zeros((0, DIMS), dtype=np.float32)
            self._set([c for c, _ in kept] + fresh, np.concatenate([old, vectors]).astype(np.float32))
            count = len(self.chunks)
        expected = len(kept) + len(fresh)
        assert count == expected, f"{name}: expected {expected} chunks, have {count}; _set() must take kept and fresh chunks"

    def _set(self, chunks: list[Chunk], vectors: Vectors):
        assert len(chunks) == len(vectors), f"{len(chunks)} chunks but {len(vectors)} vectors; embed every chunk before indexing it"
        assert vectors.ndim == 2 and vectors.shape[1] == DIMS, f"vectors must be (n, {DIMS}), got {vectors.shape}; stack one vector per chunk"
        self.chunks = chunks
        self.vectors = vectors

    def spoken(self, vector: Vectors, smoothed: Vectors | None = None) -> list[tuple[int, float]]:
        """Chunks by cosine to what was said, best first, averaged with
        earlier updates when `smoothed` is given."""
        assert vector.shape == (DIMS,), f"query vector must have {DIMS} dimensions, got {vector.shape}; embed it with embed_audio()"
        count = len(self.chunks)
        assert smoothed is None or len(smoothed) == count, f"smoothed scores cover {len(smoothed)} chunks, index has {count}; start a new session after re-indexing"
        cosine = self.vectors @ vector
        if smoothed is not None:
            fresh = np.isnan(smoothed)
            smoothed[fresh] = cosine[fresh]
            smoothed += SMOOTHING * (cosine - smoothed)
            cosine = smoothed
        return [(int(i), float(cosine[i])) for i in np.argsort(-cosine)[:LIMIT]]

    def hits(self, ranked: list[tuple[int, float]]) -> list[Hit]:
        """What editors show for each ranked chunk."""
        outside = [i for i, _ in ranked if not 0 <= i < len(self.chunks)]
        assert not outside, f"ranked chunks {outside[:3]} are out of range for {len(self.chunks)} chunks; rank against this index"
        out = [self.hit(self.chunks[i], score) for i, score in ranked[:LIMIT]]
        assert len(out) <= LIMIT, f"{len(out)} hits is more than the limit of {LIMIT}; keep the [:LIMIT] slice"
        return out

    def hit(self, chunk: Chunk, score: float) -> Hit:
        assert chunk.start <= chunk.end, f"chunk of {chunk.file} ends at {chunk.end}, before it starts at {chunk.start}; check split()"
        assert -1.001 <= score <= 1.001, f"cosine score {score} must be between -1 and 1; normalise vectors with unit()"
        return {
            "file": str(self.root / chunk.file),
            "path": chunk.file,
            "start": chunk.start,
            "end": chunk.end,
            "end_col": len(chunk.text.rstrip("\n").split("\n")[-1].encode()),
            "symbols": chunk.symbols[:4],
            "preview": first_line(chunk.text),
            "score": round(score, 4),
        }
