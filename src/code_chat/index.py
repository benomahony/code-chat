"""Embed a repository's chunks as text, and rank them against speech.

The same scheme as the voice search on benomahony.com: what someone has said
so far is embedded straight from the audio, with no transcription, and chunks
rank by cosine, smoothed across live updates so results only move when
they're consistently better.
"""

import hashlib
import os
from pathlib import Path

import numpy as np

from code_chat.chunks import Chunk, files, split, wanted

MODEL = "google/embeddinggemma-2"
DIMS = 768
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "code-chat"
LIMIT = 30
SMOOTHING = 0.4  # Weight of the newest update.

# --- Model -------------------------------------------------------------------

_model = None


def model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # bfloat16 on Apple GPUs; float16 overflows this model.
        dtype = torch.bfloat16 if device == "mps" else torch.float32
        _model = SentenceTransformer(
            MODEL,
            device=device,
            # Code as text, queries as speech; leave out the image encoder.
            config_kwargs={"vision_config": None},
            model_kwargs={"torch_dtype": dtype},
        )
    dims = _model.get_sentence_embedding_dimension()
    assert dims == DIMS, f"{MODEL} gives {dims}-dimensional vectors; the cache expects {DIMS}"
    assert _model.max_seq_length > 0, f"{MODEL} loaded with no input length ({_model.max_seq_length})"
    return _model


def unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    assert np.all(norms > 0), f"{int(np.sum(norms == 0))} zero vectors can't be normalised"
    out = vectors / norms
    assert out.shape == vectors.shape, f"normalising changed the shape from {vectors.shape} to {out.shape}"
    return out


def embed(inputs: list) -> np.ndarray:
    assert inputs, "embed() needs at least one input"
    vectors = unit(model().encode(inputs, batch_size=8, convert_to_numpy=True).astype(np.float32))
    assert vectors.shape == (len(inputs), DIMS), f"embedded {len(inputs)} inputs into shape {vectors.shape}"
    return vectors


def embed_audio(audio: np.ndarray) -> np.ndarray:
    assert audio.ndim == 1, f"audio must be mono samples, got shape {audio.shape}"
    assert len(audio) > 0, f"there's no audio to embed (got {len(audio)} samples); wait for Session.due()"
    return embed([{"audio": audio}])[0]


def key(chunk: Chunk) -> str:
    """Cache key: vectors are cached by the content they embed."""
    document = chunk.document()
    assert document, f"chunk of {chunk.file} has an empty document"
    out = hashlib.sha256((MODEL + document).encode()).hexdigest()[:32]
    assert len(out) == 32, f"cache key {out!r} should be 32 hex digits"
    return out


def first_line(text: str) -> str:
    assert text.strip(), "a chunk with no text has no first line to preview"
    out = next(line.strip() for line in text.splitlines() if line.strip())[:120]
    assert out and len(out) <= 120, f"preview {out!r} must be non-empty and at most 120 characters"
    return out


# --- Index -------------------------------------------------------------------


class Index:
    def __init__(self, root: Path):
        assert root.is_absolute(), f"repository root {root} must be absolute"
        assert root.is_dir(), f"repository root {root} isn't a directory"
        self.root = root
        self.cache_file = CACHE / f"{hashlib.sha256(str(root).encode()).hexdigest()[:16]}.npz"
        self.cache: dict[str, np.ndarray] = {}
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, DIMS), dtype=np.float32)

    def _load_cache(self):
        assert self.cache_file.suffix == ".npz", f"cache file {self.cache_file} must be .npz"
        if self.cache or not self.cache_file.exists():
            return
        data = np.load(self.cache_file)
        keys, vectors = data["keys"], data["vectors"]
        assert len(keys) == len(vectors), f"{self.cache_file} has {len(keys)} keys for {len(vectors)} vectors; delete it to rebuild"
        self.cache = dict(zip(keys.tolist(), vectors.astype(np.float32)))

    def _save_cache(self, keep: set[str]):
        self.cache = {k: v for k, v in self.cache.items() if k in keep}
        shapes = {v.shape for v in self.cache.values()}
        assert shapes <= {(DIMS,)}, f"cached vectors must all have {DIMS} dimensions, found shapes {shapes}"
        CACHE.mkdir(parents=True, exist_ok=True)
        keys = list(self.cache)
        vectors = np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, DIMS))
        np.savez(self.cache_file, keys=np.array(keys), vectors=vectors.astype(np.float16))
        assert self.cache_file.exists(), f"saving the cache didn't write {self.cache_file}"

    def _vectors(self, chunks: list[Chunk], progress=None) -> np.ndarray:
        """Vectors for `chunks`, embedding only those not already cached."""
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
        uncached = [k for k in keys if k not in self.cache]
        assert not uncached, f"{len(uncached)} chunks still have no vector after embedding"
        out = np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, DIMS), dtype=np.float32)
        assert len(out) == len(chunks), f"{len(out)} vectors for {len(chunks)} chunks"
        return out

    def build(self, progress=None):
        chunks = [c for name in files(self.root) for c in split(self.root, name)]
        self._set(chunks, self._vectors(chunks, progress))
        # Drop vectors for code that no longer exists.
        self._save_cache({key(c) for c in chunks})
        assert len(self.chunks) == len(chunks), f"indexed {len(self.chunks)} of {len(chunks)} chunks"
        assert len(self.cache) <= len(chunks), f"cache holds {len(self.cache)} vectors for {len(chunks)} chunks"

    def update(self, name: str):
        """Re-split and re-embed one file, after it's saved."""
        assert not name.startswith("/"), f"{name} must be relative to {self.root}"
        kept = [(c, v) for c, v in zip(self.chunks, self.vectors) if c.file != name]
        # A deleted file, or one the index skips, leaves only what was kept.
        fresh = split(self.root, name) if (self.root / name).exists() and wanted(self.root, name) else []
        vectors = self._vectors(fresh)
        old = np.stack([v for _, v in kept]) if kept else np.zeros((0, DIMS), dtype=np.float32)
        self._set([c for c, _ in kept] + fresh, np.concatenate([old, vectors]).astype(np.float32))
        expected = len(kept) + len(fresh)
        assert len(self.chunks) == expected, f"{name}: expected {expected} chunks, have {len(self.chunks)}"

    def _set(self, chunks: list[Chunk], vectors: np.ndarray):
        assert len(chunks) == len(vectors), f"{len(chunks)} chunks but {len(vectors)} vectors; embed every chunk before indexing it"
        assert vectors.ndim == 2 and vectors.shape[1] == DIMS, f"vectors must be (n, {DIMS}), got {vectors.shape}"
        self.chunks = chunks
        self.vectors = vectors

    def spoken(self, vector: np.ndarray, smoothed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Chunks by cosine to what was said, best first, averaged with
        earlier updates when `smoothed` is given."""
        assert vector.shape == (DIMS,), f"query vector must have {DIMS} dimensions, got {vector.shape}"
        count = len(self.chunks)
        assert smoothed is None or len(smoothed) == count, f"smoothed scores for {len(smoothed)} chunks, index has {count}"
        cosine = self.vectors @ vector
        if smoothed is not None:
            fresh = np.isnan(smoothed)
            smoothed[fresh] = cosine[fresh]
            smoothed += SMOOTHING * (cosine - smoothed)
            cosine = smoothed
        return [(int(i), float(cosine[i])) for i in np.argsort(-cosine)[:LIMIT]]

    def hits(self, ranked: list[tuple[int, float]]) -> list[dict]:
        """What editors show for each ranked chunk."""
        outside = [i for i, _ in ranked if not 0 <= i < len(self.chunks)]
        assert not outside, f"ranked chunks {outside[:3]} are out of range for {len(self.chunks)} chunks"
        out = [self.hit(self.chunks[i], score) for i, score in ranked[:LIMIT]]
        assert len(out) <= LIMIT, f"{len(out)} hits is more than the limit of {LIMIT}"
        return out

    def hit(self, chunk: Chunk, score: float) -> dict:
        assert chunk.start <= chunk.end, f"chunk of {chunk.file} ends at {chunk.end}, before it starts at {chunk.start}"
        assert -1.001 <= score <= 1.001, f"cosine score {score} must be between -1 and 1"
        return {
            "file": str(self.root / chunk.file),
            "path": chunk.file,
            "start": chunk.start,
            "end": chunk.end,
            # Byte column where the chunk ends on its last line.
            "end_col": len(chunk.text.rstrip("\n").split("\n")[-1].encode()),
            "symbols": chunk.symbols[:4],
            "preview": first_line(chunk.text),
            "score": round(score, 4),
        }
