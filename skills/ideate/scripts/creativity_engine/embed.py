"""Pluggable text embeddings + near-duplicate suppression.

Three providers, selected by the ``CREATIVITY_EMBEDDER`` environment variable:

* ``hash``  — deterministic char-n-gram hashing vectorizer (no downloads). Used
  by the test suite and non-live ``selftest``. Lexically similar text → similar
  vectors, so dedup is meaningful.
* ``local`` — sentence-transformers ``BAAI/bge-small-en-v1.5`` (CPU, ~33M params,
  a **different model family** from the agent → satisfies the lineage hedge).
  This is the default for real runs. Lazily downloaded on first use.
* ``api``   — a stub for a hosted provider (Voyage/Cohere/OpenAI), selected via
  env so callers never change. Constructing it is cheap; embedding raises until
  wired up.

All embedders return an ``(n, d)`` float32 array of **L2-normalized** rows, so
cosine similarity is a plain dot product.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

ENV_VAR = "CREATIVITY_EMBEDDER"
DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
HASH_DIM = 512

# Per-embedder near-duplicate cosine thresholds. Cosine scale is family-specific:
# "the same idea, reworded" sits at different similarities under a char-n-gram
# hashing vectorizer vs. a sentence model, so one global tau misfires when the
# embedder changes. Keyed by ``Embedder.name``; unknown families fall back to the
# default.
DEFAULT_DEDUP_TAU = 0.92
DEDUP_TAU_BY_EMBEDDER = {
    "hash": 0.92,   # char-n-gram cosines: near-dupes cluster ~0.92+
    "local": 0.94,  # sentence-transformer cosines run higher; raise the bar
    "api": 0.92,    # unknown backend: conservative default
}


def default_dedup_tau(embedder_name: str) -> float:
    """Near-duplicate cosine threshold calibrated to the embedder family."""
    return DEDUP_TAU_BY_EMBEDDER.get(embedder_name, DEFAULT_DEDUP_TAU)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32)


class Embedder:
    """Interface: turn texts into an ``(n, d)`` array of normalized rows.

    ``embed`` is a **template method** — it coerces the inputs to ``str``, short-
    circuits the empty case, calls the subclass hook :meth:`_embed_raw`, and then
    L2-normalizes. Centralizing the normalization here means a new provider only
    implements ``_embed_raw`` and *cannot forget* to return unit rows, which the
    rest of the math relies on (cosine == dot product).
    """

    name: str = "base"
    dim: int = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(self._embed_raw(texts))

    def _embed_raw(self, texts: List[str]) -> np.ndarray:  # pragma: no cover
        """Return UNnormalized ``(n, d)`` rows; the base class normalizes them."""
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic, dependency-light embedder over character n-grams."""

    name = "hash"

    def __init__(self, dim: int = HASH_DIM):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vec = HashingVectorizer(
            n_features=dim,
            analyzer="char_wb",
            ngram_range=(2, 4),
            alternate_sign=True,
            norm=None,  # we normalize ourselves so zero rows are handled
        )

    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray()


class LocalEmbedder(Embedder):
    """sentence-transformers embedder (lazy model load)."""

    name = "local"

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL):
        self.model_name = model_name
        self._model = None
        self.dim = 0

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            # method was renamed across sentence-transformers versions
            get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension"
            )
            self.dim = int(get_dim())
        return self._model

    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        model = self._ensure()
        return model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class APIEmbedder(Embedder):
    """Stub for a hosted embedding provider, selected via env vars.

    Construction is intentionally cheap so a provider switch loads without import
    errors; actually embedding raises until a backend is wired up.
    """

    name = "api"

    def __init__(self):
        self.provider = os.environ.get("CREATIVITY_EMBED_API", "voyage")
        self.api_key = os.environ.get("CREATIVITY_EMBED_API_KEY", "")
        self.dim = 0

    def _embed_raw(self, texts: List[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(
            f"API embedder provider {self.provider!r} is a stub; set "
            f"{ENV_VAR}=local or =hash, or wire up a backend in APIEmbedder."
        )


_CACHE: dict = {}


def get_embedder(provider: Optional[str] = None) -> Embedder:
    """Return the embedder selected by ``provider`` or ``$CREATIVITY_EMBEDDER``.

    Cached per-provider so repeated calls reuse the (lazily loaded) model.
    """
    provider = (provider or os.environ.get(ENV_VAR) or DEFAULT_PROVIDER).strip().lower()
    if provider in _CACHE:
        return _CACHE[provider]
    if provider == "hash":
        emb: Embedder = HashingEmbedder()
    elif provider == "local":
        emb = LocalEmbedder()
    elif provider == "api":
        emb = APIEmbedder()
    else:
        raise ValueError(
            f"unknown embedder provider {provider!r}; expected hash|local|api"
        )
    _CACHE[provider] = emb
    return emb


def reset_cache() -> None:
    """Clear the embedder cache (tests switching providers)."""
    _CACHE.clear()


# --------------------------------------------------------------------------- #
# Near-duplicate suppression
# --------------------------------------------------------------------------- #
def dedupe(
    vecs: np.ndarray,
    tau: float = 0.92,
    existing: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[int]]:
    """Greedy near-duplicate removal over normalized rows.

    Keeps a row unless its cosine similarity to an already-kept row (or to any
    ``existing`` row) exceeds ``tau``.

    Returns ``(keep_indices, drop_indices)`` into ``vecs`` (row order preserved).
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    n = vecs.shape[0]
    if n == 0:
        return [], []
    dim = vecs.shape[1]
    # Running buffer of kept rows (existing seeds first, then survivors) so we
    # never re-vstack the whole set on each step — one preallocated matrix.
    n_existing = 0 if existing is None else len(existing)
    kept = np.empty((n_existing + n, dim), dtype=np.float32)
    count = 0
    if n_existing:
        kept[:n_existing] = np.asarray(existing, dtype=np.float32)
        count = n_existing
    keep: List[int] = []
    drop: List[int] = []
    for i in range(n):
        v = vecs[i]
        if count and float(np.max(kept[:count] @ v)) > tau:
            drop.append(i)
            continue
        keep.append(i)
        kept[count] = v
        count += 1
    return keep, drop
