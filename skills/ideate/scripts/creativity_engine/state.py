"""File-based local state for a project.

State is written **outside** the plugin so reinstalls don't wipe it. The base
directory is ``~/.creativity-amplifier`` by default, overridable with the
``CREATIVITY_AMPLIFIER_HOME`` environment variable (used by the test suite to
isolate runs). Layout::

    <home>/<project>/
        meta.json                 # project metadata
        axes.json                 # snapshot of the resolved axes
        archive.json              # MAP-Elites: niche_id -> niche record
        candidates.json           # id -> candidate record (genealogy kept)
        embeddings.json           # id -> embedding vector
        memory/<domain>/
            comparisons.jsonl     # appended preference events
            pins.json             # pinned "stepping stone" ids

Preferences are namespaced per domain so switching domains keeps memories
separate.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_HOME_ENV = "CREATIVITY_AMPLIFIER_HOME"
_DEFAULT_BASE = "~/.creativity-amplifier"

_PATH_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _path_slug(name: str, fallback: str = "default") -> str:
    """Filesystem-safe slug for a project or domain id."""
    s = _PATH_SLUG_RE.sub("-", str(name).strip()).strip("-_.")
    return s or fallback


def base_dir() -> Path:
    env = os.environ.get(DEFAULT_HOME_ENV)
    root = Path(env) if env else Path(_DEFAULT_BASE)
    return root.expanduser()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class State:
    """Handle to one project's on-disk state."""

    def __init__(self, project: str, home: Optional[Path] = None):
        self.project = project
        self.base = Path(home).expanduser() if home else base_dir()
        self.root = self.base / _path_slug(project)

    # -- paths -------------------------------------------------------------- #
    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def axes_path(self) -> Path:
        return self.root / "axes.json"

    @property
    def archive_path(self) -> Path:
        return self.root / "archive.json"

    @property
    def candidates_path(self) -> Path:
        return self.root / "candidates.json"

    @property
    def embeddings_path(self) -> Path:
        return self.root / "embeddings.json"

    @property
    def open_nicher_path(self) -> Path:
        return self.root / "open_nicher.json"

    def memory_dir(self, domain: str) -> Path:
        return self.root / "memory" / _path_slug(domain)

    def comparisons_path(self, domain: str) -> Path:
        return self.memory_dir(domain) / "comparisons.jsonl"

    def pins_path(self, domain: str) -> Path:
        return self.memory_dir(domain) / "pins.json"

    # -- lifecycle ---------------------------------------------------------- #
    def exists(self) -> bool:
        return self.meta_path.exists()

    def ensure(self) -> "State":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def paths(self) -> Dict[str, str]:
        return {
            "root": str(self.root),
            "meta": str(self.meta_path),
            "axes": str(self.axes_path),
            "archive": str(self.archive_path),
            "candidates": str(self.candidates_path),
            "embeddings": str(self.embeddings_path),
        }

    # -- generic json helpers ---------------------------------------------- #
    def read_json(self, path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, path: Path, obj: Any) -> None:
        _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))

    # -- typed accessors ---------------------------------------------------- #
    def read_meta(self) -> Dict[str, Any]:
        return self.read_json(self.meta_path, {}) or {}

    def write_meta(self, meta: Dict[str, Any]) -> None:
        self.write_json(self.meta_path, meta)

    def read_axes(self) -> Optional[Dict[str, Any]]:
        return self.read_json(self.axes_path, None)

    def write_axes(self, axes: Dict[str, Any]) -> None:
        self.write_json(self.axes_path, axes)

    def read_archive(self) -> Dict[str, Any]:
        return self.read_json(self.archive_path, {}) or {}

    def write_archive(self, archive: Dict[str, Any]) -> None:
        self.write_json(self.archive_path, archive)

    def read_candidates(self) -> Dict[str, Any]:
        return self.read_json(self.candidates_path, {}) or {}

    def write_candidates(self, candidates: Dict[str, Any]) -> None:
        self.write_json(self.candidates_path, candidates)

    def read_embeddings(self) -> Dict[str, List[float]]:
        return self.read_json(self.embeddings_path, {}) or {}

    def write_embeddings(self, embeddings: Dict[str, List[float]]) -> None:
        self.write_json(self.embeddings_path, embeddings)

    def read_open_nicher(self) -> Optional[Dict[str, Any]]:
        """Persisted open-axis nicher: cold-start accumulation or frozen centroids."""
        return self.read_json(self.open_nicher_path, None)

    def write_open_nicher(self, data: Dict[str, Any]) -> None:
        self.write_json(self.open_nicher_path, data)

    # -- memory (namespaced by domain) ------------------------------------- #
    def append_comparison(self, domain: str, event: Dict[str, Any]) -> None:
        path = self.comparisons_path(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def read_comparisons(self, domain: str) -> List[Dict[str, Any]]:
        path = self.comparisons_path(domain)
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def read_pins(self, domain: str) -> List[str]:
        return self.read_json(self.pins_path(domain), []) or []

    def write_pins(self, domain: str, pins: List[str]) -> None:
        self.write_json(self.pins_path(domain), pins)

    def add_pin(self, domain: str, candidate_id: str) -> List[str]:
        pins = self.read_pins(domain)
        if candidate_id not in pins:
            pins.append(candidate_id)
            self.write_pins(domain, pins)
        return pins
