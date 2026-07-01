"""Shared pytest fixtures.

Tests are hermetic: state goes to a temp dir (via ``CAMBRIAN_HOME``)
and embeddings use the deterministic hashing embedder (via
``CAMBRIAN_EMBEDDER=hash``) so no model is ever downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _hash_embedder(monkeypatch):
    """Force the deterministic embedder for the whole suite."""
    monkeypatch.setenv("CAMBRIAN_EMBEDDER", "hash")


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """An isolated state home for one test."""
    h = tmp_path / "state-home"
    h.mkdir()
    monkeypatch.setenv("CAMBRIAN_HOME", str(h))
    return h
