"""Tests for the B-matching machinery: tokens, garnishes, and the marginal fit."""

import itertools

import numpy as np
import pytest

from profilebattery.generate.bpool import build_b_cores, shape_signature
from profilebattery.generate.families import CAP, MODULES, STRUCTURAL_EXEMPT, \
    build_a_cores, family_of, sibling_b_cores
from profilebattery.generate.garnish import GARNISHES, SUPPLIERS, garnishes_for
from profilebattery.generate.matching import _needed, _union, b_profile, \
    companion_profile, fit_garnishes, fit_two_placements, worst_gap
from profilebattery.generate.menu import DOMAINS, SCOPES, Cell, StubMenu
from profilebattery.inventory.tokens import NODE_TOKENS, all_tokens, sharers, siblings

CELLS = [Cell(d, s, p, 0) for d, s, p in itertools.product(DOMAINS[:2], SCOPES, (0, 2))]


# --- tokens ---------------------------------------------------------------------

def test_every_generated_construct_has_a_token():
    for mod in MODULES.values():
        for c in mod.SPECS:
            assert c in NODE_TOKENS, c


def test_shared_tokens_are_derived_not_declared():
    """Collisions are computed from the token table, so they grow with the inventory.

    Completing the table to all 63 constructs widened them: the pattern nodes collide
    with the expression constructs that share their delimiters, and `type` collides
    across member kinds — the TypeAlias statement against the `type` builtin.
    """
    s = sharers()
    assert set(s["for"]) == {"For", "AsyncFor", "ListComp", "SetComp", "DictComp",
                             "GeneratorExp"}
    assert set(s["if"]) == {"If", "IfExp"}
    assert set(s["["]) == {"List", "Subscript", "MatchSequence"}
    assert set(s["{"]) == {"Dict", "Set", "MatchMapping", "FormattedValue"}
    assert set(s["type"]) == {"TypeAlias", "type"}       # AST construct vs builtin name
    assert "For" in siblings("ListComp") and "ListComp" in siblings("For")
    assert siblings("While") == ()          # `while` belongs to one construct only


def test_builtin_names_are_their_own_tokens():
    t = all_tokens()
    assert t["len"] == "len" and t["ValueError"] == "ValueError"


# --- garnishes ------------------------------------------------------------------

def test_no_garnish_is_offered_for_a_construct_it_creates():
    for construct in NODE_TOKENS:
        for g in garnishes_for(construct):
            assert construct not in GARNISHES[g][1], f"{g} creates {construct}"


def test_every_garnish_renders_valid_python():
    import ast

    menu, cell, rng = StubMenu(), Cell("physics", "module", 0, 0), np.random.default_rng(0)
    for name, (fn, creates) in GARNISHES.items():
        src = "\n".join(fn(menu, cell, rng)) + "\n"
        tree = ast.parse(src)          # raises if the garnish does not parse
        found = {type(n).__name__ for n in ast.walk(tree)}
        for made in creates:
            assert made in found, f"{name} claims {made} but does not create it"


def test_suppliers_cover_the_companions_that_matter():
    for companion in ("Subscript", "Call", "Compare", "BinOp", "If", "For", "Attribute",
                      "Pass", "List", "Dict", "Return", "Break"):
        assert SUPPLIERS.get(companion), f"no garnish supplies {companion}"


# --- the marginal fit -----------------------------------------------------------

def test_union_probability_not_additive():
    """A garnish added to a prompt that already carries the companion raises nothing."""
    assert _needed(0.5, 0.0) == pytest.approx(0.5)
    assert _needed(0.75, 0.5) == pytest.approx(0.5)        # not 0.25
    assert _union(0.5, _needed(0.75, 0.5)) == pytest.approx(0.75)
    assert _needed(0.2, 0.4) == 0.0                        # already above target


def test_fit_respects_components_and_the_target_construct():
    target = {"Subscript": 0.8, "For": 0.5, "Call": 0.3}
    probs = fit_garnishes(target, {}, "For", components=("Call",), cap=0.15)
    for g in probs:
        assert "For" not in GARNISHES[g][1]
        assert "Call" not in GARNISHES[g][1] or "Call" in target


@pytest.mark.parametrize("construct", ["For", "If", "Assign", "BinOp", "ListComp"])
@pytest.mark.parametrize("site", ["token", "last"])
def test_matching_closes_the_worst_gap(construct, site):
    menu = StubMenu()
    token = NODE_TOKENS[construct]
    cap = CAP[family_of(construct)]
    cores = build_a_cores(construct, 8, np.random.default_rng(11))
    components = tuple(cores[0].companions) + STRUCTURAL_EXEMPT.get(construct, ())
    a = companion_profile(cores, menu, CELLS, construct, site=site)
    a_last = companion_profile(cores, menu, CELLS, construct, site="last")
    pooled = build_b_cores(construct, token, 5, np.random.default_rng(5), target=a_last)
    b0 = b_profile(pooled, token, menu, CELLS, construct, site=site)
    pre, post = fit_two_placements(cores, pooled, token, menu, CELLS, construct,
                                   components, cap)
    b1 = b_profile(pooled, token, menu, CELLS, construct, pre=pre, post=post, site=site)
    before, after = worst_gap(a, b0, construct, components), \
        worst_gap(a, b1, construct, components)
    # The property is not "always smaller": a gap already inside the cap may be traded
    # for a different one of similar size, and on the reduced cell set used here that
    # trade is within sampling noise. What must hold is that matching brings the worst
    # gap inside the cap, or at least does not make it worse.
    assert abs(after[1]) <= max(cap, abs(before[1])), f"{construct}: {before} -> {after}"


def test_carrier_selection_prefers_low_excess_shapes():
    """Garnishes add but never remove, so carriers are the only control on excess."""
    target = {"Assign": 0.9}          # a set with almost no companions but assignments
    picked = build_b_cores("For", "for", 5, np.random.default_rng(5), target=target)
    excess = 0
    for core in picked:
        if core.subtype.startswith("other_role"):
            continue
        sig = shape_signature("For", core.subtype, core.shape_index, "for")
        excess += sum(1 for c in sig if target.get(c, 0.0) < 0.5)
    unguided = build_b_cores("For", "for", 5, np.random.default_rng(5))
    un_excess = sum(
        sum(1 for c in shape_signature("For", k.subtype, k.shape_index, "for")
            if target.get(c, 0.0) < 0.5)
        for k in unguided if not k.subtype.startswith("other_role"))
    assert excess <= un_excess


def test_sibling_b_cores_come_from_the_a_grammars():
    cores = sibling_b_cores("For", 4, np.random.default_rng(3))
    assert cores, "For shares `for` with the comprehensions"
    assert all(c.subtype.startswith("other_role:sibling:") for c in cores)
    assert any("ListComp" in c.subtype for c in cores)


# --- draw diversity: the harness bug that made every prompt identical -------------

def test_draws_are_diverse_within_a_set():
    """One seed per construct must spawn a per-prompt stream, not one shared draw.

    A constant seed collapses menu values and garnish choices to a single realisation
    across the whole set — it looks like a working generator and silently removes the
    variation averaging is supposed to cancel. Asserted here because the failure is
    invisible in every other test.
    """
    from profilebattery.generate.assemble import assemble

    menu = StubMenu()
    core = build_a_cores("For", 1, np.random.default_rng(11))[0]
    cell = Cell("finance", "module", 0, 0)
    texts = {assemble(core, menu, cell, seed=s).text for s in range(12)}
    assert len(texts) >= 6, "draws within one cell are not diverse"

    across = {assemble(core, menu, c, seed=abs(hash((core.core_id, c.key()))) % 2**31).text
              for c in CELLS}
    assert len(across) == len(CELLS), "one text per cell expected, got collisions"


def test_garnish_placement_is_visible_at_the_right_sites():
    """A garnish after the carrier is invisible at the token site, visible at the last."""
    from profilebattery.generate.assemble import assemble_b
    from profilebattery.generate.audit import present, scope_for

    menu, cell = StubMenu(), Cell("finance", "module", 0, 0)
    core = [k for k in build_b_cores("For", "for", 5, np.random.default_rng(5))
            if not k.subtype.startswith("other_role")][0]
    post = assemble_b(core, "for", menu, cell, seed=3, post_probs={"guard_rebind": 1.0})
    pre = assemble_b(core, "for", menu, cell, seed=3, pre_probs={"guard_rebind": 1.0})
    assert post.valid and pre.valid
    s, b = scope_for(post, "token")
    assert "If" not in present(post.text, "For", s, b)
    s, b = scope_for(post, "last")
    assert "If" in present(post.text, "For", s, b)
    s, b = scope_for(pre, "token")
    assert "If" in present(pre.text, "For", s, b)


# --- corpus-weighted axes (§3, decided 20 Sept) -----------------------------------

def test_the_axis_rule_is_explicit_and_only_body_is_weighted():
    from profilebattery.generate.weights import AXIS_KIND, WEIGHTED_AXES
    assert WEIGHTED_AXES == ("body",)
    for axis in ("operand", "suite", "variant", "element", "source", "host", "access"):
        assert AXIS_KIND[axis] == "cancels", axis


def test_allocation_honours_the_floor_and_keeps_every_level():
    from profilebattery.generate.weights import FLOOR, allocate
    levels = ("a", "b", "c", "d")
    got = allocate(levels, 8, {"a": 0.9, "b": 0.05, "c": 0.03, "d": 0.02})
    assert len(got) == 8
    assert set(got) == set(levels), "no level may vanish"
    assert got.count("a") >= 4
    with pytest.raises(ValueError):
        allocate(levels, 3, {l: 0.25 for l in levels})   # cannot cover 4 levels with 3


def test_weights_are_realised_by_allocation_so_cores_stay_exchangeable():
    """The condition on the bootstrap is met by construction rather than by a change.

    Allocating cores to levels leaves every core standing for 1/n of its set, so the core
    bootstrap needs no weighting — which is the one place unequal core weights could have
    silently corrupted an error bar. The cost is resolution: 1/n.
    """
    import collections
    from profilebattery.generate.core import spread
    lv = {"body": ("p", "q", "r", "s"), "operand": tuple("abcdefgh"),
          "suite": ("s1", "s2")}
    w = {"body": {"p": 0.55, "q": 0.20, "r": 0.15, "s": 0.10}}
    out = spread(lv, 24, np.random.default_rng(3), weights=w)
    counts = collections.Counter(c["body"] for c in out)
    assert counts["p"] > counts["q"] > 1
    assert len({c["operand"] for c in out}) == 8, "coverage axes stay covered"
    assert len({tuple(c.values()) for c in out}) == 24


def test_a_weighted_level_cannot_exceed_its_distinct_partners():
    """With eight unweighted combinations, no level may be allocated more than eight."""
    import collections
    from profilebattery.generate.core import spread
    lv = {"body": ("p", "q"), "operand": tuple("abcdefgh")}
    out = spread(lv, 14, np.random.default_rng(1),
                 weights={"body": {"p": 0.95, "q": 0.05}})
    counts = collections.Counter(c["body"] for c in out)
    assert max(counts.values()) <= 8
    assert len({tuple(c.values()) for c in out}) == 14


def test_fit_drives_companion_heavy_levels_down():
    from profilebattery.generate.weights import FLOOR, fit_body_weights
    sig = {"mutate": {"BinOp": 1.0}, "transform_append": {"BinOp": 1.0},
           "accumulate": {"AugAssign": 1.0}, "delegate": {"Call": 1.0}}
    w = fit_body_weights(sig, {"BinOp": 0.13, "Call": 0.99})
    assert w["delegate"] > 0.5
    assert w["mutate"] + w["transform_append"] < 0.2
    assert all(v >= FLOOR - 1e-9 for v in w.values())
    assert abs(sum(w.values()) - 1.0) < 1e-3   # weights are rounded for the log
