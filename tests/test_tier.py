"""The deep tier is a union of four declared sets, not a frequency cut.

The regression that matters: the tier must not silently become "the top 40". Three
quarters of the pre-compaction discussion was about claims needing constructs that
frequency would drop, and a refactor that reduced the union to its frequency arm
would be invisible in the counts alone.
"""
import pytest

from profilebattery.inventory import tier


@pytest.fixture
def rate():
    # z is rarest, a is commonest; ties on name so the set is stable
    return {c: float(i) for i, c in enumerate("zyxwvutsrqponmlkjihgfedcba")}


def test_top_n_takes_the_commonest(rate):
    assert tier.top_n(rate, 3) == {"a", "b", "c"}


def test_top_n_breaks_ties_by_name():
    flat = {c: 1.0 for c in "dcba"}
    assert tier.top_n(flat, 2) == {"a", "b"}


def test_assign_is_a_union_not_a_cut(rate):
    deep, sets = tier.assign(rate, claim={"z"}, poly={"y"}, bridge={"x"}, n=2)
    assert {"a", "b"} <= deep          # the frequency arm
    assert {"x", "y", "z"} <= deep     # the three rarest, kept by claim
    assert deep == {"a", "b", "x", "y", "z"}
    assert set(sets) == {"bridge", "claim", "polysemous", "top_n"}


def test_claim_members_survive_a_zero_rate():
    """A construct the claim needs stays deep even with no corpus occurrences —
    the vintage zeros (Match*, TypeAlias) depend on this."""
    deep, _ = tier.assign({"a": 1.0}, claim={"MatchStar"}, poly=set(), bridge=set(), n=1)
    assert "MatchStar" in deep


def test_rate_table_drops_leaves():
    assert "Name" in tier.RATE_LEAVES and "Constant" in tier.RATE_LEAVES


def test_bridge_is_the_union_of_two_grounded_v1_sets():
    assert tier.BRIDGE == tier.BRIDGE_PROBE | tier.BRIDGE_NAMED
    # they are genuinely different sets, overlapping only in the two comprehensions
    assert tier.BRIDGE_PROBE & tier.BRIDGE_NAMED == {"ListComp", "GeneratorExp"}
    assert len(tier.BRIDGE) == 22
