"""Phase 1: axes resolve identically across named / inferred / generic paths,
and malformed specs fail loudly with a non-zero exit."""

from __future__ import annotations

import json

import pytest
import yaml

from cambrian_engine import config
from cambrian_engine.__main__ import main
from cambrian_engine.config import (
    AxesSpec,
    ConfigError,
    SessionSettings,
    axes_spec_from_dict,
    load_axes,
    load_generic_axes,
    load_session_settings,
)


SPEC_DICT = {
    "domain": "campaign-ideas",
    "unit_of_generation": "concept",
    "axes": [
        {"name": "audience", "type": "categorical"},
        {"name": "register", "type": "categorical"},
        {"name": "format", "type": "categorical"},
        {"name": "edginess", "type": "continuous", "range": [0, 1]},
        {"name": "mechanism", "type": "open", "primary_novelty": True},
    ],
    "slate_size": 5,
    "candidates_per_generation": 10,
}


def _write(path, data, as_yaml):
    text = yaml.safe_dump(data) if as_yaml else json.dumps(data)
    path.write_text(text, encoding="utf-8")
    return path


def test_named_inferred_generic_paths_load_identically(tmp_path):
    # (a) a named domain config (yaml) and (b) an inferred axes.json with the
    # same content must produce identical specs, as must (c) a raw dict.
    yaml_path = _write(tmp_path / "marketing.yaml", SPEC_DICT, as_yaml=True)
    json_path = _write(tmp_path / "axes.json", SPEC_DICT, as_yaml=False)

    from_yaml = load_axes(yaml_path)
    from_json = load_axes(json_path)
    from_dict = axes_spec_from_dict(SPEC_DICT)

    assert from_yaml == from_json == from_dict
    assert isinstance(from_yaml, AxesSpec)


def test_continuous_range_normalized_to_floats():
    spec = axes_spec_from_dict(SPEC_DICT)
    edg = spec.axis("edginess")
    assert edg.type == "continuous"
    assert edg.range == (0.0, 1.0)
    assert all(isinstance(b, float) for b in edg.range)


def test_primary_axis_detection():
    spec = axes_spec_from_dict(SPEC_DICT)
    assert spec.primary_axis is not None
    assert spec.primary_axis.name == "mechanism"


def test_round_trip_to_dict_is_stable():
    spec = axes_spec_from_dict(SPEC_DICT)
    again = axes_spec_from_dict(spec.to_dict())
    assert spec == again


def test_axes_spec_is_pure_geometry():
    # Agent-/session-level settings must NOT leak into the engine's core type.
    spec = axes_spec_from_dict(SPEC_DICT)
    assert not hasattr(spec, "candidates_per_generation")
    assert not hasattr(spec, "judge_rubric")
    serialized = spec.to_dict()
    assert "candidates_per_generation" not in serialized
    assert "judge_rubric" not in serialized


def test_session_settings_load_from_same_source():
    # The settings ride alongside the axes in one file and parse out separately.
    settings = load_session_settings(SPEC_DICT)
    assert settings.candidates_per_generation == 10
    assert settings.judge_rubric == "references/judge_rubric.md"


def test_session_settings_default_when_absent():
    settings = load_session_settings({"axes": []})  # axes ignored by this loader
    assert settings == SessionSettings()
    assert settings.candidates_per_generation == 12


def test_session_settings_rejects_bad_count():
    with pytest.raises(ConfigError) as exc:
        SessionSettings.from_dict({"candidates_per_generation": 0})
    assert "candidates_per_generation" in str(exc.value)


def test_generic_fallback_loads_and_is_valid():
    spec = load_generic_axes()
    assert spec.domain == "generic"
    assert 4 <= len(spec.axes) <= 6
    assert spec.primary_axis is not None
    # generic stays domain-neutral
    assert spec.primary_axis.type == "open"


@pytest.mark.parametrize(
    "bad, needle",
    [
        ({"axes": []}, "non-empty"),
        ({"axes": [{"type": "categorical"}]}, "name"),
        ({"axes": [{"name": "x", "type": "bogus"}]}, "type"),
        ({"axes": [{"name": "x", "type": "continuous"}]}, "range"),
        (
            {"axes": [{"name": "x", "type": "continuous", "range": [1, 0]}]},
            "greater",
        ),
        (
            {"axes": [{"name": "a", "type": "open"}, {"name": "a", "type": "open"}]},
            "unique",
        ),
        (
            {
                "axes": [
                    {"name": "a", "type": "open", "primary_novelty": True},
                    {"name": "b", "type": "open", "primary_novelty": True},
                ]
            },
            "primary_novelty",
        ),
    ],
)
def test_malformed_specs_raise_clear_errors(bad, needle):
    with pytest.raises(ConfigError) as exc:
        axes_spec_from_dict(bad)
    assert needle in str(exc.value)


def test_cli_malformed_axes_exits_nonzero(tmp_path, capsys, home):
    bad_path = _write(tmp_path / "bad.json", {"axes": []}, as_yaml=False)
    code = main(["init-project", "--project", "p", "--axes", str(bad_path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err and "non-empty" in err


def test_cli_init_and_recall_round_trip(tmp_path, capsys, home):
    good = _write(tmp_path / "axes.json", SPEC_DICT, as_yaml=False)
    assert main(["init-project", "--project", "p", "--axes", str(good)]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["domain"] == "campaign-ideas"

    assert main(["recall", "--project", "p"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["domain"] == "campaign-ideas"
    assert rec["pins"] == []
    assert rec["preferences"] == []
