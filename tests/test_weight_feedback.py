"""The calibration chain must not read its own output (1 Oct).

`fit_core_weights` writes `core_weights.json`; `weights.for_construct` reads it; `spread`
allocates cores to levels by it; and `body_signatures` -- the measurement the fit is computed
*from* -- built its cores through that path. So the fit's input was the fit's previous output.

Measured before the fix: from the committed state one run moved a weight by 0.364, and iterating
did not converge -- eight runs, eight states, a limit cycle of period 3. The weights were not
identified by the procedure; which ones the battery was built with depended on how many times
the script had been run. These tests pin the three properties that make it a function of the
corpus alone.
"""

import collections
import pathlib
import re

import numpy as np
import pytest

from profilebattery.generate import weights as W
from profilebattery.generate.families import build_a_cores
from profilebattery.generate.seeds import stable_seed

SKEWED = {"For": {"body": {"filter_continue": 0.9, "accumulate": 0.05, "transform": 0.05}}}


@pytest.fixture
def skewed_cache(monkeypatch):
    """A weights table far from balanced, as the loop's late iterations produced."""
    monkeypatch.setattr(W, "_CACHE", SKEWED)
    return SKEWED


def body_levels(construct, n=24):
    return collections.Counter(c.axes.get("body")
                               for c in build_a_cores(construct, n,
                                                      np.random.default_rng(7)))


def test_balanced_hides_the_fitted_weights(skewed_cache):
    assert W.for_construct("For") == SKEWED["For"]
    with W.balanced():
        assert W.for_construct("For") == {}
    assert W.for_construct("For") == SKEWED["For"], "the override must not leak"


def test_balanced_restores_even_when_the_block_raises(skewed_cache):
    with pytest.raises(RuntimeError):
        with W.balanced():
            raise RuntimeError("boom")
    assert W.for_construct("For") == SKEWED["For"]


def test_balanced_nests(skewed_cache):
    with W.balanced():
        with W.balanced():
            assert W.for_construct("For") == {}
        assert W.for_construct("For") == {}, "the inner block must not end the outer one"
    assert W.for_construct("For") == SKEWED["For"]


def test_core_allocation_is_weight_free_inside_balanced(skewed_cache):
    """The property that breaks the loop: the measurement's cores ignore the weights."""
    with W.balanced():
        under_skew = body_levels("For")
    W._CACHE = {}
    with W.balanced():
        under_none = body_levels("For")
    assert under_skew == under_none


def test_the_skew_really_would_have_changed_the_allocation(skewed_cache):
    """Guards the test above: without `balanced` the weights do move the cores.

    Without this, `test_core_allocation_is_weight_free_inside_balanced` could pass because
    the weights never mattered, rather than because `balanced` neutralises them.
    """
    skewed = body_levels("For")
    W._CACHE = {}
    none = body_levels("For")
    assert skewed != none, "if this fails the fixture no longer exercises the loop"


def test_no_salted_hash_survives_in_the_fitting_path():
    """`hash()` of a str is salted per process, so it cannot seed a reproducible fit."""
    root = pathlib.Path(__file__).resolve().parent.parent / "profilebattery/generate"
    for name in ("weights.py", "matching.py"):
        src = (root / name).read_text()
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        assert not re.search(r"\babs\(hash\(", code), f"{name} still seeds off a salted hash"


def test_stable_seed_is_reproducible_across_processes():
    """The property `abs(hash(...))` lacked: same inputs, same seed, any interpreter."""
    assert stable_seed("For/c4", "gaming|module|0") == stable_seed("For/c4", "gaming|module|0")
    assert stable_seed("a", "b") != stable_seed("b", "a")
    assert 0 <= stable_seed("x") < 2 ** 31
