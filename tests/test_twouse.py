"""The two-use sets (D1, G5), and why they need no new construct sets.

Each assertion below stands for a measurement on the pilot battery, quoted in the module's
docstring: the pairs already have their own cores, the B sets already carry the other use as
nameable sibling groups, and the only real confound -- the statement use reading a different
token piece at module scope -- is removed by restricting the read to the indented scopes.
"""

from profilebattery.generate import twouse
from profilebattery.generate.families import siblings


def test_the_pairs_are_existing_constructs():
    """Nothing to generate: `For` against the comprehensions, `If` against `IfExp`."""
    for p in twouse.PAIRS:
        assert p.statement in ("For", "If")
        assert p.expression
        for e in p.expression:
            assert e != p.statement


def test_the_members_are_each_others_siblings():
    """Which is what puts the other use in each B set, as a group the kernel can drop by name."""
    for p in twouse.PAIRS:
        for e in p.expression:
            assert e in siblings(p.statement), f"{e} is not a sibling of {p.statement}"
            assert p.statement in siblings(e)


def test_the_read_is_restricted_to_indented_scopes():
    """A statement `for` reads `for` at module scope and `▁for` when indented; a comprehension's
    reads `▁for` everywhere. Only the indented scopes are piece-matched."""
    for p in twouse.PAIRS:
        assert p.scopes == twouse.INDENTED_SCOPES
        assert "module" not in p.scopes


def test_the_cost_of_the_restriction_is_recorded():
    """Two thirds of the prompts per statement set -- 480 of 720 -- and not a battery change."""
    assert abs(twouse.SHARE_KEPT - 2 / 3) < 0.01


def test_scope_keyed_covers_the_members_and_a_control():
    ks = twouse.scope_keyed_constructs()
    assert "While" in ks, "the control needs the keys too, or there is nothing to compare"
    for p in twouse.PAIRS:
        assert p.statement in ks
        for e in p.expression:
            assert e in ks


def test_pair_lookup_works_from_either_side():
    assert twouse.pair_of("For").token == "for"
    assert twouse.pair_of("GeneratorExp").token == "for"
    assert twouse.pair_of("IfExp").token == "if"
    assert twouse.pair_of("While") is None


def test_the_if_pair_records_that_p8_cannot_be_read_on_it():
    p = twouse.pair_of("IfExp")
    assert "P8" in p.note and "dependent" in p.note
