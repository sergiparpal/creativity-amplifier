"""Pluggable text embeddings + near-duplicate suppression.

Four providers, selected by the ``CAMBRIAN_EMBEDDER`` environment variable:

* ``static`` — model2vec ``minishlab/potion-multilingual-128M`` (256-dim, **101
  languages**, distilled from ``BAAI/bge-m3``, MIT). Static embeddings, so
  **inference needs only numpy — no torch**, and the weights are ~120 MB (vs.
  ~2 GB for the torch stack). A **different model family** from the agent →
  satisfies the lineage hedge. This is the **default** for real runs; lazily
  downloaded on first use.
* ``local`` — sentence-transformers ``BAAI/bge-small-en-v1.5`` (CPU, ~33M params,
  384-dim, **English-only**). The opt-in **high-fidelity** option; needs the
  torch stack (``pip install -r requirements-local.txt``). Lazily downloaded.
* ``hash``  — deterministic char-n-gram hashing vectorizer (no downloads). Used
  by the test suite and non-live ``selftest``. Lexically similar text → similar
  vectors, so dedup is meaningful.
* ``api``   — a stub for a hosted provider (Voyage/Cohere/OpenAI), selected via
  env so callers never change. Constructing it is cheap; embedding raises until
  wired up.

All embedders return an ``(n, d)`` float32 array of **L2-normalized** rows, so
cosine similarity is a plain dot product.

Note: ``static`` (256-dim) and ``local`` (384-dim) produce different-width,
incompatible geometries; ``pipeline._guard_embedding_dim`` refuses to mix them
within one project, so switching the default is breaking for projects persisted
under the old default (re-embed, or pin ``CAMBRIAN_EMBEDDER=local``).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

ENV_VAR = "CAMBRIAN_EMBEDDER"
DEFAULT_PROVIDER = "static"
DEFAULT_STATIC_MODEL = "minishlab/potion-multilingual-128M"
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
HASH_DIM = 512

# Per-embedder near-duplicate cosine thresholds. Cosine scale is family-specific:
# "the same idea, reworded" sits at different similarities under a char-n-gram
# hashing vectorizer vs. a sentence model, so one global tau misfires when the
# embedder changes. Keyed by ``Embedder.name``; unknown families fall back to the
# default.
DEFAULT_DEDUP_TAU = 0.92
DEDUP_TAU_BY_EMBEDDER = {
    "hash": 0.92,    # char-n-gram cosines: near-dupes cluster ~0.92+
    "static": 0.93,  # potion (model2vec): trivial restatements ~0.96-0.99 EN/ES,
                     # genuine synonym variations ~0.86, distinct ideas <=0.43 —
                     # 0.93 drops the former, keeps the latter (calibrated on a
                     # near-dup/distinct EN+ES sample).
    "local": 0.94,   # sentence-transformer cosines run higher; raise the bar
    "api": 0.92,     # unknown backend: conservative default
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
            return np.zeros((0, self._dim_for_empty()), dtype=np.float32)
        return l2_normalize(self._embed_raw(texts))

    def _dim_for_empty(self) -> int:
        """Width of the empty-input result. Defaults to ``self.dim``; embedders whose
        ``dim`` is resolved LAZILY (model loaded on first real embed) override this to
        resolve it, so an accidental ``embed([])`` before the first real call can't
        return a 0-width array that would later trip ``_guard_embedding_dim``. Eager-dim
        providers (hash / api / tests) keep the default and load nothing."""
        return self.dim

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


class StaticEmbedder(Embedder):
    """model2vec static embedder (lazy model load).

    Static token embeddings averaged per text, so **inference needs only numpy**
    (no torch). The default real-run embedder: multilingual and ~120 MB on disk.
    """

    name = "static"

    def __init__(self, model_name: str = DEFAULT_STATIC_MODEL):
        self.model_name = model_name
        self._model = None
        self.dim = 0

    def _ensure(self):
        if self._model is None:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(self.model_name)
            self.dim = int(self._model.dim)
        return self._model

    def _dim_for_empty(self) -> int:
        self._ensure()  # dim is unknown until the model loads
        return self.dim

    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        model = self._ensure()
        # StaticModel.encode already returns a float32 ndarray; the base class
        # L2-normalizes, so we only need the raw rows here.
        return np.asarray(model.encode(list(texts)), dtype=np.float32)


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

    def _dim_for_empty(self) -> int:
        self._ensure()  # dim is unknown until the model loads
        return self.dim

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
        self.provider = os.environ.get("CAMBRIAN_EMBED_API", "voyage")
        self.api_key = os.environ.get("CAMBRIAN_EMBED_API_KEY", "")
        self.dim = 0

    def _embed_raw(self, texts: List[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(
            f"API embedder provider {self.provider!r} is a stub; set "
            f"{ENV_VAR}=local or =hash, or wire up a backend in APIEmbedder."
        )


_CACHE: dict = {}


def get_embedder(provider: Optional[str] = None) -> Embedder:
    """Return the embedder selected by ``provider`` or ``$CAMBRIAN_EMBEDDER``.

    Cached per-provider so repeated calls reuse the (lazily loaded) model.
    """
    provider = (provider or os.environ.get(ENV_VAR) or DEFAULT_PROVIDER).strip().lower()
    if provider in _CACHE:
        return _CACHE[provider]
    if provider == "static":
        emb: Embedder = StaticEmbedder()
    elif provider == "hash":
        emb = HashingEmbedder()
    elif provider == "local":
        emb = LocalEmbedder()
    elif provider == "api":
        emb = APIEmbedder()
    else:
        raise ValueError(
            f"unknown embedder provider {provider!r}; expected static|hash|local|api"
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
