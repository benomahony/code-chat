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

from code_chat.chunks import Chunk, files, split

MODEL = "google/embeddinggemma-2"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "code-chat"

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
    return _model


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def embed(inputs: list) -> np.ndarray:
    return unit(model().encode(inputs, batch_size=8, convert_to_numpy=True).astype(np.float32))


def embed_audio(audio: np.ndarray) -> np.ndarray:
    return embed([{"audio": audio}])[0]


# --- Index -------------------------------------------------------------------

LIMIT = 30


class Index:
    def __init__(self, root: Path):
        self.root = root
        self.cache_file = CACHE / f"{hashlib.sha256(str(root).encode()).hexdigest()[:16]}.npz"
        self.cache: dict[str, np.ndarray] = {}
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    # Vectors are cached by the content they embed, so a rebuild only embeds
    # what changed.
    def _load_cache(self):
        if self.cache or not self.cache_file.exists():
            return
        data = np.load(self.cache_file)
        self.cache = dict(zip(data["keys"].tolist(), data["vectors"].astype(np.float32)))

    def _save_cache(self, keep: set[str]):
        self.cache = {k: v for k, v in self.cache.items() if k in keep}
        CACHE.mkdir(parents=True, exist_ok=True)
        keys = list(self.cache)
        vectors = np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, 768))
        np.savez(self.cache_file, keys=np.array(keys), vectors=vectors.astype(np.float16))

    def _vectors(self, chunks: list[Chunk], progress=None) -> np.ndarray:
        self._load_cache()
        keys = [hashlib.sha256((MODEL + c.document()).encode()).hexdigest()[:32] for c in chunks]
        missing = [(k, c) for k, c in zip(keys, chunks) if k not in self.cache]
        batch = 32
        for start in range(0, len(missing), batch):
            part = missing[start : start + batch]
            for (k, _), v in zip(part, embed([c.document() for _, c in part])):
                self.cache[k] = v
            if progress:
                progress(min(start + batch, len(missing)), len(missing))
        if missing:
            self._save_cache(set(keys) | set(self.cache))
        return np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, 768), dtype=np.float32)

    def build(self, progress=None):
        chunks = [c for name in files(self.root) for c in split(self.root, name)]
        self._set(chunks, self._vectors(chunks, progress))
        # Drop vectors for code that no longer exists.
        keys = {hashlib.sha256((MODEL + c.document()).encode()).hexdigest()[:32] for c in chunks}
        self._save_cache(keys)

    def update(self, name: str):
        """Re-split and re-embed one file, after it's saved."""
        kept = [(c, v) for c, v in zip(self.chunks, self.vectors) if c.file != name]
        fresh = split(self.root, name) if (self.root / name).exists() else []
        vectors = self._vectors(fresh)
        chunks = [c for c, _ in kept] + fresh
        matrix = np.concatenate([np.stack([v for _, v in kept]) if kept else np.zeros((0, vectors.shape[1])), vectors])
        self._set(chunks, matrix.astype(np.float32))

    def _set(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors

    # --- Ranking ---------------------------------------------------------------

    def spoken(self, vector: np.ndarray, smoothed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Chunks by cosine to what was said, best first, averaged with
        earlier updates when `smoothed` is given."""
        cosine = self.vectors @ vector
        if smoothed is not None:
            fresh = np.isnan(smoothed)
            smoothed[fresh] = cosine[fresh]
            smoothed += SMOOTHING * (cosine - smoothed)
            cosine = smoothed
        return [(int(i), float(cosine[i])) for i in np.argsort(-cosine)[:LIMIT]]

    def hits(self, ranked: list[tuple[int, float]]) -> list[dict]:
        return [
            {
                "file": str(self.root / self.chunks[i].file),
                "path": self.chunks[i].file,
                "start": self.chunks[i].start,
                "end": self.chunks[i].end,
                "symbols": self.chunks[i].symbols[:4],
                "preview": first_line(self.chunks[i].text),
                "score": round(score, 4),
            }
            for i, score in ranked[:LIMIT]
        ]


SMOOTHING = 0.4  # Weight of the newest update.


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")[:120]
