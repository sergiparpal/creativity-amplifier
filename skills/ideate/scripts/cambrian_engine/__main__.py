"""Command-line entry point: ``python -m cambrian_engine <cmd> ...``.

Every command reads/writes JSON and takes ``--project`` plus, where relevant,
``--axes`` / ``--seed``. Output goes to stdout as JSON; errors print to stderr
and exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import config, pipeline, selftest


def _emit(obj: Any) -> None:
    json.dump(obj, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cambrian_engine",
        description="Deterministic diversity engine for the ideate skill.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init-project", help="create state dirs, snapshot axes")
    sp.add_argument("--project", required=True)
    sp.add_argument("--axes", required=True, help="path to axes .json/.yaml")
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser(
        "paths", help="ensure the project state dir (+ tmp/) and print resolved paths"
    )
    sp.add_argument("--project", required=True)

    sp = sub.add_parser("recall", help="return preference memory for injection")
    sp.add_argument("--project", required=True)
    sp.add_argument("--k", type=int, default=10)

    sp = sub.add_parser("ingest", help="embed -> place -> novelty -> DPP -> monitor")
    sp.add_argument("--project", required=True)
    sp.add_argument("--candidates", required=True, help="path to candidates .json")
    sp.add_argument("--axes", required=True, help="path to axes .json/.yaml")
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("remember", help="append a comparison/pin/discard to memory")
    sp.add_argument("--project", required=True)
    sp.add_argument("--event", required=True, help="path to event .json")

    sp = sub.add_parser("parents", help="diverse parents for next generation")
    sp.add_argument("--project", required=True)
    sp.add_argument("--k", type=int, default=4)
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("metrics", help="current archive health")
    sp.add_argument("--project", required=True)

    sp = sub.add_parser("selftest", help="full loop with stubbed LLM + human")
    sp.add_argument("--live", action="store_true", help="use the live embedder")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--project", default="selftest")

    return p


def _read_json_file(path: str) -> Any:
    from pathlib import Path

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Identify which path failed — ingest reads both --candidates and --axes.
        raise config.ConfigError(f"could not read JSON from {path}: {exc}") from exc


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init-project":
            _emit(pipeline.init_project(args.project, args.axes, seed=args.seed))
        elif args.command == "paths":
            _emit(pipeline.paths(args.project))
        elif args.command == "recall":
            _emit(pipeline.recall(args.project, k=args.k))
        elif args.command == "ingest":
            candidates = _read_json_file(args.candidates)
            _emit(
                pipeline.ingest(
                    args.project, candidates, args.axes, seed=args.seed
                )
            )
        elif args.command == "remember":
            event = _read_json_file(args.event)
            _emit(pipeline.remember(args.project, event))
        elif args.command == "parents":
            _emit(pipeline.parents(args.project, k=args.k, seed=args.seed))
        elif args.command == "metrics":
            _emit(pipeline.metrics(args.project))
        elif args.command == "selftest":
            report = selftest.run(
                project=args.project, live=args.live, seed=args.seed
            )
            _emit(report)
            return 0 if report.get("ok") else 1
        else:  # pragma: no cover - argparse enforces choices
            raise SystemExit(2)
    except config.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # surface a clean message, non-zero exit
        if config.debug_enabled():
            raise  # full traceback for diagnosis when debugging
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
