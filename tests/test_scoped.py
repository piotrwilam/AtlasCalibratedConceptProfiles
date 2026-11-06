"""The scope-restricted profile machinery — A24, S15."""
import numpy as np
import pytest

from profilebattery.profiles.scoped import GROUPS, INDENTED, SCOPES, ScopeView


class Acc:
    """The subset of the npz interface `from_accumulators` and `ScopeView` use."""

    def __init__(self, d):
        self._d = d
        self.files = list(d)

    def __getitem__(self, k):
        return self._d[k]


def build(units=4, layers=2, cores=("c0", "c1"), cells=("b0:keyword",),
          scopes=SCOPES, n=10, seed=0):
    rng = np.random.default_rng(seed)
    d = {}
    for core in cores:
        for scope in scopes:
            for li in range(layers):
                for cell in cells:
                    base = f"hs|X/{core}@{scope}|token|L{li}|{cell}"
                    d[base + "|n"] = np.array([n])
                    d[base + "|mean"] = rng.standard_normal(units)
    # the pooled family too, so the view must not pick it up
    for core in cores:
        for li in range(layers):
            for cell in cells:
                base = f"h|X/{core}|token|L{li}|{cell}"
                d[base + "|n"] = np.array([3 * n])
                d[base + "|mean"] = rng.standard_normal(units)
    return Acc(d)


def test_one_scope_keeps_its_own_cores_and_counts():
    acc = build()
    v = ScopeView(acc, ("module",))
    keys = [k for k in v.files if k.endswith("|n")]
    assert len(keys) == 2 * 2, "two cores x two layers x one cell"
    assert all("@" not in k.split("|")[1] for k in keys), "the scope is stripped off the core"
    assert {k.split("|")[1] for k in keys} == {"X/c0", "X/c1"}
    assert int(v["hs|X/c0|token|L0|b0:keyword|n"][0]) == 10


def test_the_pooled_h_family_is_never_picked_up():
    """`h` and `hs` ride the same `act` axis, and only `hs` carries the scope."""
    v = ScopeView(build(), ("module",))
    assert all(k.startswith("hs|") for k in v.files)


def test_indented_combines_function_and_method_instead_of_overwriting_one():
    """The defect this module exists to prevent.

    Stripping `@function` and `@method` makes two keys collide on (core, layer, cell).
    `from_accumulators` assigns into that triple, so unmerged keys would keep whichever came
    last -- 8 cores and half the prompts, reported as a complete profile.
    """
    acc = build()
    v = ScopeView(acc, INDENTED)
    keys = [k for k in v.files if k.endswith("|n")]
    assert len(keys) == 4, "still two cores x two layers, not four pseudo-cores"
    assert {k.split("|")[1] for k in keys} == {"X/c0", "X/c1"}, "no @scope survives in a core"

    # the count is the union's, not one scope's
    assert int(v["hs|X/c0|token|L0|b0:keyword|n"][0]) == 20

    # and the mean is the count-weighted combination, not either input
    f = acc["hs|X/c0@function|token|L0|b0:keyword|mean"]
    m = acc["hs|X/c0@method|token|L0|b0:keyword|mean"]
    got = v["hs|X/c0|token|L0|b0:keyword|mean"]
    assert np.allclose(got, (10 * f + 10 * m) / 20)
    assert not np.allclose(got, f) and not np.allclose(got, m)


def test_unequal_counts_are_weighted_not_averaged():
    """A plain mean of two scope means is wrong whenever the scopes have different n."""
    d = {}
    for scope, n, val in (("function", 30, 1.0), ("method", 10, 5.0)):
        base = f"hs|X/c0@{scope}|token|L0|b0:keyword"
        d[base + "|n"] = np.array([n])
        d[base + "|mean"] = np.full(3, val)
    v = ScopeView(Acc(d), INDENTED)
    got = v["hs|X/c0|token|L0|b0:keyword|mean"]
    assert np.allclose(got, (30 * 1.0 + 10 * 5.0) / 40), got      # 2.0
    assert not np.allclose(got, 3.0), "the unweighted mean of 1 and 5 is wrong here"


def test_a_cell_present_in_only_one_scope_survives_with_its_own_count():
    d = {}
    base = "hs|X/c0@function|token|L0|b0:keyword"
    d[base + "|n"] = np.array([7])
    d[base + "|mean"] = np.ones(3)
    base2 = "hs|X/c0@method|token|L1|b0:keyword"
    d[base2 + "|n"] = np.array([5])
    d[base2 + "|mean"] = np.full(3, 2.0)
    v = ScopeView(Acc(d), INDENTED)
    assert int(v["hs|X/c0|token|L0|b0:keyword|n"][0]) == 7
    assert int(v["hs|X/c0|token|L1|b0:keyword|n"][0]) == 5


def test_a_reference_without_the_scope_key_fails_loudly():
    """Silence here would standardise a pooled profile and call it the primary one."""
    acc = Acc({"h|X/c0|token|L0|b0:keyword|n": np.array([3]),
               "h|X/c0|token|L0|b0:keyword|mean": np.zeros(2)})
    with pytest.raises(ValueError, match="cannot be recovered later"):
        ScopeView(acc, ("module",))


def test_a_key_that_is_not_cell_keyed_fails_loudly():
    acc = Acc({"hs|X/c0@module|token|L0|n": np.array([3]),
               "hs|X/c0@module|token|L0|mean": np.zeros(2)})
    with pytest.raises(ValueError, match="not cell-keyed"):
        ScopeView(acc, ("module",))


def test_the_groups_are_the_four_s15_stores():
    assert list(GROUPS) == ["module", "function", "method", "indented"]
    assert GROUPS["indented"] == ("function", "method")
    assert set(SCOPES) == {"module", "function", "method"}


def test_scope_profiles_sum_to_the_pooled_count():
    """A sanity property that holds on the real record: the three scopes partition the set."""
    acc = build(n=11)
    ns = [sum(int(ScopeView(acc, (s,))[k][0]) for k in ScopeView(acc, (s,)).files
              if k.endswith("|n")) for s in SCOPES]
    allthree = ScopeView(acc, SCOPES)
    total = sum(int(allthree[k][0]) for k in allthree.files if k.endswith("|n"))
    assert sum(ns) == total
