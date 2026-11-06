"""B4, B6 and B8b — the parts that can be checked without the record.

The three traps these cover, each of which has already cost a pass somewhere in the
programme: a scope label that is silently ignored so a "module" profile is built from every
scope; an `ast` rule that is written as a rule and then quietly hand-corrected; and a map
whose colours are not the profile's because the layout permutation was applied twice or not
at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.atlas import (BINDING_FIELDS, INDENTED, SCOPES, ScopeView, _ast_categories,
                                _degree_layout, _ov, _purity, K)


class Fake:
    """The smallest npz-alike with scope-keyed `hs|` accumulators."""

    def __init__(self, entries):
        self._d = {}
        for core, scope, layer, cell, n, val in entries:
            stem = f"hs|{core}@{scope}|token|L{layer}|{cell}"
            self._d[stem + "|n"] = np.array([n])
            self._d[stem + "|mean"] = np.full(4, float(val), np.float16)
        self.files = list(self._d)

    def __getitem__(self, k):
        return self._d[k]


def test_scope_view_keeps_only_the_wanted_scope_and_counts_the_refusals():
    z = Fake([("c0", "module", 0, "b0:keyword", 3, 1.0),
              ("c0", "function", 0, "b0:keyword", 5, 2.0),
              ("c0", "method", 0, "b0:keyword", 7, 4.0)])
    v = ScopeView(z, ("module",), "module")
    assert v.read == 1 and v.wrong_scope == 2
    assert v.prompts() == 3
    assert v["hs|c0@module|token|L0|b0:keyword|mean"][0] == pytest.approx(1.0)


def test_scope_view_pools_colliding_entries_by_count_weighted_mean():
    """function + method land on one key, and must combine by n, not be overwritten."""
    z = Fake([("c0", "function", 0, "b0:keyword", 1, 0.0),
              ("c0", "method", 0, "b0:keyword", 3, 4.0)])
    v = ScopeView(z, INDENTED, "indented")
    assert v.prompts() == 4
    assert v["hs|c0@indented|token|L0|b0:keyword|mean"][0] == pytest.approx(3.0)


def test_scope_view_refuses_a_key_without_a_scope():
    class Bare:
        files = ["hs|c0|token|L0|b0:keyword|mean", "hs|c0|token|L0|b0:keyword|n"]

        def __getitem__(self, k):
            return np.array([1]) if k.endswith("|n") else np.zeros(4, np.float16)

    with pytest.raises(ValueError, match="without a scope"):
        ScopeView(Bare(), ("module",), "module")


def test_all_three_scopes_reproduce_the_unpooled_total():
    z = Fake([("c0", s, 0, "b0:keyword", n, v)
              for s, n, v in (("module", 2, 1.0), ("function", 3, 2.0), ("method", 5, 3.0))])
    v = ScopeView(z, SCOPES, "all")
    assert v.prompts() == 10
    assert v["hs|c0@all|token|L0|b0:keyword|mean"][0] == pytest.approx(
        (2 * 1.0 + 3 * 2.0 + 5 * 3.0) / 10)


def test_ast_rules_find_the_constructs_the_design_names():
    r = _ast_categories(["For", "If", "While", "Try", "With", "FunctionDef", "ClassDef",
                         "Match", "AsyncFor", "AsyncFunctionDef", "Assign", "AnnAssign",
                         "AugAssign", "Import", "ImportFrom", "NamedExpr", "Pass", "Expr"])
    for n in ("If", "For", "While", "Try", "With", "FunctionDef", "ClassDef",
              "AsyncFor", "AsyncFunctionDef"):
        assert n in r["compound"], n
    # the rule excludes Match and the design's list includes it; both must be carried
    assert "Match" not in r["compound"]
    assert "Match" in r["compound_with_Match"]
    assert "Match" in r["note_on_Match"]
    for n in ("Assign", "AnnAssign", "AugAssign", "FunctionDef", "AsyncFunctionDef",
              "ClassDef", "Import", "ImportFrom", "For", "With", "NamedExpr"):
        assert n in r["binding"], n
    assert "Pass" not in r["compound"] and "Pass" not in r["binding"]
    assert "Expr" not in r["binding"]
    assert set(BINDING_FIELDS) >= {"targets", "target", "name"}


def test_overlap_reports_both_conventions():
    a = np.arange(K)
    b = np.arange(K)
    o = _ov(a, b)
    assert o["overlap_share_of_K"] == pytest.approx(1.0)
    assert o["jaccard"] == pytest.approx(1.0)
    half = _ov(np.arange(K), np.arange(K // 2, K // 2 + K))
    assert half["overlap_share_of_K"] > half["jaccard"]


def test_purity_of_a_perfect_category_beats_the_null():
    rng = np.random.default_rng(0)
    labels = ["x"] * 10 + [None] * 10 + ["y"] * 10
    clusters = [1] * 10 + [2] * 10 + [3] * 10
    r = _purity(labels, clusters, rng, n_perm=200)
    assert r["x"]["purity"] == pytest.approx(1.0)
    assert r["x"]["p"] < 0.05
    assert "None" not in r


def test_purity_of_a_scattered_category_does_not():
    rng = np.random.default_rng(1)
    labels = ["x" if i % 3 == 0 else "y" for i in range(30)]
    clusters = list(range(1, 31))
    r = _purity(labels, clusters, rng, n_perm=200)
    assert r["x"]["p"] > 0.05


def test_degree_layout_is_a_permutation_and_orders_by_degree():
    sets = {"a": np.array([0, 1, 2]), "b": np.array([1, 2, 6144]), "c": np.array([2])}
    deg, order = _degree_layout(sets, ["a", "b", "c"])
    assert deg[2] == 3 and deg[1] == 2 and deg[0] == 1 and deg[6144] == 1
    assert order.shape == (24, 6144)
    assert np.array_equal(np.sort(order.ravel()), np.arange(24 * 6144))
    assert list(order[0][:3]) == [2, 1, 0]          # degree 3, 2, 1
    assert order[1][0] == 6144                      # row 1 starts at its own offset
