"""State path-slug safety: clean ids are preserved; lossy/colliding ids disambiguate.

The slug maps a project/domain id to a filesystem directory. A clean ASCII id must
round-trip unchanged (so existing state dirs are preserved), but ids that slug
*lossily* — non-ASCII-only ids, or distinct ids that reduce to the same slug — must
never collide on one directory, or they would silently merge state (archives,
embeddings, per-domain preference memory).
"""

from __future__ import annotations

import re

from cambrian_engine.state import State, _path_slug


def test_clean_ascii_ids_round_trip_unchanged():
    for name in ["p", "q", "ov", "selftest-diverse", "research_hypotheses",
                 "ad-hoc", "generic", "marketing", "product_features"]:
        assert _path_slug(name) == name


def test_unicode_only_ids_do_not_collide_or_default():
    a, b, c = _path_slug("日本語"), _path_slug("한국어"), _path_slug("español-café")
    assert len({a, b, c}) == 3            # all distinct
    assert a != "default" and b != "default"  # no collapse to the fallback


def test_lossy_ascii_ids_do_not_collide():
    # "proj A" base-slugs to "proj-A", which would collide with a literal "proj-A".
    assert _path_slug("proj A") != _path_slug("proj-A")
    assert _path_slug("proj-A") == "proj-A"   # already clean -> unchanged
    # path-separator-bearing ids must not collide with their dashed twin
    assert _path_slug("a/b") != _path_slug("a-b")


def test_slug_is_always_filesystem_safe():
    for raw in ["a/b\\c:d*e", "日本語", "  ", "...", "x" * 5]:
        assert re.fullmatch(r"[A-Za-z0-9._-]+", _path_slug(raw))


def test_distinct_unicode_projects_get_distinct_roots(home):
    r1 = State("проект-один", home=home).root
    r2 = State("проект-два", home=home).root
    assert r1 != r2
