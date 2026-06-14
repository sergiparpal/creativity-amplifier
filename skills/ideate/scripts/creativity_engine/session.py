"""Per-session orchestration context.

A :class:`Session` bundles the four things every CLI command needs — the
on-disk :class:`~creativity_engine.state.State` handle, the resolved preference
**domain** (the memory namespace), the axes **spec**, and the **embedder** —
constructed once and resolved lazily.

It exists to hold, in *one* place, the rule the pipeline commands used to each
re-derive by hand: the preference-memory namespace is the ``domain`` of the
persisted axes snapshot. Keeping that here means ``ingest`` / ``recall`` /
``remember`` / ``parents`` can never drift on which namespace a project's
memory lives in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import config
from .config import AxesSpec
from .embed import Embedder, get_embedder
from .state import State


class Session:
    """One project's resolved context for a single CLI invocation.

    Everything is lazy: constructing a ``Session`` only builds the ``State``
    handle. ``domain`` / ``spec`` read the persisted axes snapshot on first
    access; ``embedder`` is the env-selected default unless one is injected
    (which is the seam the self-test and tests use to avoid global env state).
    """

    def __init__(
        self,
        project: str,
        *,
        home: Optional[Path] = None,
        seed: int = 0,
        embedder: Optional[Embedder] = None,
    ):
        self.project = project
        self.seed = int(seed)
        self.state = State(project, home=home)
        self._embedder = embedder
        self._spec: Optional[AxesSpec] = None
        self._domain: Optional[str] = None

    def ensure(self) -> "Session":
        """Create the project's state directory; return self for chaining."""
        self.state.ensure()
        return self

    @property
    def domain(self) -> str:
        """Preference-memory namespace = the persisted snapshot's domain.

        Resolved once and cached so the memory namespace is identical across
        every command in this invocation.
        """
        if self._domain is None:
            axes = self.state.read_axes()
            self._domain = (
                str(axes.get("domain", "default"))
                if isinstance(axes, dict) and axes
                else "default"
            )
        return self._domain

    @property
    def spec(self) -> AxesSpec:
        """Axes from the persisted snapshot, falling back to the generic axes.

        Used by commands that read an existing project (``metrics``,
        ``parents``). Commands that *receive* axes for the cycle (``init`` /
        ``ingest``) resolve their spec from that argument and call
        :meth:`adopt_spec` instead.
        """
        if self._spec is None:
            axes = self.state.read_axes()
            self._spec = (
                config.axes_spec_from_dict(axes)
                if axes
                else config.load_generic_axes()
            )
        return self._spec

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def adopt_spec(self, spec: AxesSpec) -> None:
        """Persist ``spec`` as this project's axes snapshot and make it the
        session's spec/domain.

        Used when the caller supplies the axes for the cycle (``init-project``,
        and the first ``ingest`` of a fresh project) rather than reading them
        back from disk.
        """
        self.state.write_axes(spec.to_dict())
        self._spec = spec
        self._domain = spec.domain
