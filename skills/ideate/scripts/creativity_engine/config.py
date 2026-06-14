"""Core types and per-session axes resolution.

This module is the lowest-level foundation: it defines the shared dataclasses
(``Axis``, ``AxesSpec``, ``Candidate``, ``Niche``) and loads/validates the
resolved axes for a session. Axes can arrive three ways, all producing an
identical ``AxesSpec``:

* a **named** domain config (``config/domains/*.yaml``),
* an **inferred** ``axes.json`` written by the agent, or
* the **generic** fallback (``config/domains/generic.yaml``).

The engine never assumes a domain — every command receives the resolved axes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:  # pyyaml is a hard dependency, but keep the import error legible.
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "pyyaml is required (pip install -r requirements.txt)"
    ) from exc


AXIS_TYPES = ("categorical", "continuous", "open")


class ConfigError(ValueError):
    """Raised when an axes spec is malformed. Message is user-facing."""


# --------------------------------------------------------------------------- #
# Core dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Axis:
    """One descriptor axis of the idea space.

    ``type`` is one of ``categorical`` | ``continuous`` | ``open``. ``range`` is
    required for continuous axes. ``primary_novelty`` marks the single "open"
    axis that carries most of the novelty (e.g. a mechanism/approach axis); it is
    niched via CVT over embeddings.
    """

    name: str
    type: str
    range: Optional[Tuple[float, float]] = None
    primary_novelty: bool = False
    bins: int = 5  # discretization granularity for continuous axes

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "type": self.type}
        if self.range is not None:
            out["range"] = list(self.range)
        if self.primary_novelty:
            out["primary_novelty"] = True
        if self.type == "continuous" and self.bins != 5:
            out["bins"] = self.bins
        return out


@dataclass
class AxesSpec:
    """The fully-resolved descriptor space for a session."""

    domain: str
    unit_of_generation: str
    axes: List[Axis]
    slate_size: int = 6
    candidates_per_generation: int = 12
    judge_rubric: str = "references/judge_rubric.md"

    @property
    def axis_names(self) -> List[str]:
        return [a.name for a in self.axes]

    def axis(self, name: str) -> Axis:
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError(name)

    @property
    def primary_axis(self) -> Optional[Axis]:
        for a in self.axes:
            if a.primary_novelty:
                return a
        # Fall back to the first open axis if none flagged.
        for a in self.axes:
            if a.type == "open":
                return a
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "unit_of_generation": self.unit_of_generation,
            "axes": [a.to_dict() for a in self.axes],
            "slate_size": self.slate_size,
            "candidates_per_generation": self.candidates_per_generation,
            "judge_rubric": self.judge_rubric,
        }

    def __eq__(self, other: object) -> bool:  # value equality for "load identically"
        if not isinstance(other, AxesSpec):
            return NotImplemented
        return self.to_dict() == other.to_dict()


@dataclass
class Candidate:
    """An idea produced by the agent and consumed by ``ingest``."""

    id: str
    text: str
    descriptor: Dict[str, Any] = field(default_factory=dict)
    genealogy: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 1.0  # within-niche quality from the judge; NOT novelty

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Candidate":
        if not isinstance(d, dict):
            raise ConfigError(f"candidate must be an object, got {type(d).__name__}")
        cid = d.get("id")
        if not cid or not isinstance(cid, str):
            raise ConfigError("candidate missing a non-empty string 'id'")
        text = d.get("text", "")
        if not isinstance(text, str):
            raise ConfigError(f"candidate {cid!r} 'text' must be a string")
        descriptor = d.get("descriptor", {}) or {}
        if not isinstance(descriptor, dict):
            raise ConfigError(f"candidate {cid!r} 'descriptor' must be an object")
        genealogy = d.get("genealogy", {}) or {}
        if not isinstance(genealogy, dict):
            raise ConfigError(f"candidate {cid!r} 'genealogy' must be an object")
        fitness = d.get("fitness", 1.0)
        try:
            fitness = float(fitness)
        except (TypeError, ValueError):
            raise ConfigError(f"candidate {cid!r} 'fitness' must be a number")
        return cls(id=cid, text=text, descriptor=descriptor,
                   genealogy=genealogy, fitness=fitness)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "descriptor": dict(self.descriptor),
            "genealogy": dict(self.genealogy),
            "fitness": self.fitness,
        }


@dataclass
class Niche:
    """One MAP-Elites cell: at most one elite candidate per niche."""

    id: str
    coords: Dict[str, Any] = field(default_factory=dict)
    elite_id: Optional[str] = None
    fitness: float = 0.0
    novelty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "coords": dict(self.coords),
            "elite_id": self.elite_id,
            "fitness": self.fitness,
            "novelty": self.novelty,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Niche":
        return cls(
            id=d["id"],
            coords=dict(d.get("coords", {})),
            elite_id=d.get("elite_id"),
            fitness=float(d.get("fitness", 0.0)),
            novelty=float(d.get("novelty", 0.0)),
        )


# --------------------------------------------------------------------------- #
# Axes loading & validation
# --------------------------------------------------------------------------- #
def _coerce_range(value: Any, axis_name: str) -> Tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or isinstance(value, str)
    ):
        raise ConfigError(
            f"axis {axis_name!r}: 'range' must be a [lo, hi] pair"
        )
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        raise ConfigError(f"axis {axis_name!r}: 'range' bounds must be numbers")
    if not hi > lo:
        raise ConfigError(
            f"axis {axis_name!r}: 'range' hi ({hi}) must be greater than lo ({lo})"
        )
    return (lo, hi)


def _axis_from_dict(d: Any) -> Axis:
    if not isinstance(d, dict):
        raise ConfigError(f"each axis must be an object, got {type(d).__name__}")
    name = d.get("name")
    if not name or not isinstance(name, str):
        raise ConfigError("axis missing a non-empty string 'name'")
    atype = d.get("type")
    if atype not in AXIS_TYPES:
        raise ConfigError(
            f"axis {name!r}: 'type' must be one of {AXIS_TYPES}, got {atype!r}"
        )
    arange: Optional[Tuple[float, float]] = None
    if atype == "continuous":
        if "range" not in d:
            raise ConfigError(f"axis {name!r}: continuous axis requires a 'range'")
        arange = _coerce_range(d["range"], name)
    elif "range" in d and d["range"] is not None:
        arange = _coerce_range(d["range"], name)
    primary = bool(d.get("primary_novelty", False))
    bins = int(d.get("bins", 5))
    if bins < 1:
        raise ConfigError(f"axis {name!r}: 'bins' must be >= 1")
    return Axis(name=name, type=atype, range=arange,
                primary_novelty=primary, bins=bins)


def axes_spec_from_dict(d: Dict[str, Any]) -> AxesSpec:
    """Build and validate an :class:`AxesSpec` from a plain dict (json or yaml)."""
    if not isinstance(d, dict):
        raise ConfigError(f"axes spec must be an object, got {type(d).__name__}")
    raw_axes = d.get("axes")
    if not isinstance(raw_axes, list) or not raw_axes:
        raise ConfigError("axes spec must contain a non-empty 'axes' list")
    axes = [_axis_from_dict(a) for a in raw_axes]

    names = [a.name for a in axes]
    if len(names) != len(set(names)):
        raise ConfigError(f"axis names must be unique, got {names}")

    n_primary = sum(1 for a in axes if a.primary_novelty)
    if n_primary > 1:
        raise ConfigError(
            "at most one axis may set 'primary_novelty: true'; "
            f"{n_primary} did"
        )

    slate = int(d.get("slate_size", 6))
    per_gen = int(d.get("candidates_per_generation", 12))
    if slate < 1:
        raise ConfigError("'slate_size' must be >= 1")
    if per_gen < 1:
        raise ConfigError("'candidates_per_generation' must be >= 1")

    return AxesSpec(
        domain=str(d.get("domain", "ad-hoc")),
        unit_of_generation=str(d.get("unit_of_generation", "idea")),
        axes=axes,
        slate_size=slate,
        candidates_per_generation=per_gen,
        judge_rubric=str(d.get("judge_rubric", "references/judge_rubric.md")),
    )


def load_axes(source: Union[str, Path, Dict[str, Any]]) -> AxesSpec:
    """Load axes from a dict, a ``.json`` file, or a ``.yaml``/``.yml`` file.

    All three paths produce an identical :class:`AxesSpec`.
    """
    if isinstance(source, dict):
        return axes_spec_from_dict(source)
    path = Path(source)
    if not path.exists():
        raise ConfigError(f"axes source not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            # Be forgiving: try yaml (a superset of json) for unknown extensions.
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not parse axes file {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"axes file {path} is empty")
    return axes_spec_from_dict(data)


def generic_axes_path() -> Path:
    """Absolute path to the bundled neutral fallback ``generic.yaml``."""
    # creativity_engine/config.py -> .../skills/ideate/scripts/creativity_engine
    # generic.yaml -> .../skills/ideate/config/domains/generic.yaml
    skill_dir = Path(__file__).resolve().parents[2]
    return skill_dir / "config" / "domains" / "generic.yaml"


def load_generic_axes() -> AxesSpec:
    """Load the bundled generic fallback axes."""
    return load_axes(generic_axes_path())
