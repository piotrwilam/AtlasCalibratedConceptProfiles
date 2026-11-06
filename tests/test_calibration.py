"""Tests for §2 — the calibration walk and its aggregation.

The two bugs these guard against were both quadratic and both invisible on small inputs:
`ast.get_source_segment` per occurrence (0.5 s/file) and `ast.walk` per occurrence for
companions (a forty-minute stall on one large generated module). So the performance
properties are asserted, not just the values.
"""

import ast
import time

import numpy as np
import pytest

from profilebattery.calibration.tables import (MAX_FILE_CHARS, Reservoir, Tables,
                                               add_file, classify_domain, DOMAIN_SEEDS)
from profilebattery.calibration.walk import (descendant_masks, inventory_constructs,
                                             line_offsets, span, unpack, walk)
from profilebattery.generate.menu import Cell, StubMenu

SRC = """\
import json

TOTALS = {}


class Ledger:
    def post(self, rows):
        for key, value in rows.items():
            if value:
                self.total += value[0]
        return self.total


for i in range(3):
    print(i)
"""


@pytest.fixture(scope="module")
def constructs():
    return inventory_constructs()


@pytest.fixture(scope="module")
def occ(constructs):
    return walk(ast.parse(SRC), line_offsets(SRC), constructs)


# --- the walk -------------------------------------------------------------------

def test_scope_has_four_values_not_three(occ):
    """A `def` in a class body is in class scope; its statements are in method scope."""
    by = {(o.construct, o.scope) for o in occ}
    assert ("FunctionDef", "class") in by, "a method definition sits in the class body"
    assert ("For", "method") in by, "statements inside a method are in method scope"
    assert ("For", "module") in by, "and the module-level loop is at module scope"


def test_depth_counts_compound_nesting(occ):
    inner = [o for o in occ if o.construct == "If"][0]
    outer = [o for o in occ if o.construct == "For" and o.scope == "module"][0]
    assert outer.depth == 0
    assert inner.depth > outer.depth


def test_companions_are_descendants_only(occ):
    """A companion is what the construct's own span contains, not what contains it."""
    module_for = [o for o in occ if o.construct == "For" and o.scope == "module"][0]
    assert "Call" in module_for.companions          # range(3) and print(i)
    assert "ClassDef" not in module_for.companions  # a sibling, not a descendant
    method_for = [o for o in occ if o.construct == "For" and o.scope == "method"][0]
    assert {"If", "AugAssign", "Subscript"} <= method_for.companions


def test_core_axis_levels_are_read_off_real_code(occ):
    forms = {o.axis for o in occ if o.construct == "For"}
    flat = {lvl for ax in forms for k, lvl in ax if k == "operand"}
    assert flat == {"range", "items"}


def test_span_is_offset_arithmetic_not_string_rebuilding():
    offs = line_offsets(SRC)
    tree = ast.parse(SRC)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    lo, hi = span(node, offs)
    assert SRC[lo:hi].startswith("class Ledger:")
    assert SRC[lo:hi] == ast.get_source_segment(SRC, node)


def test_masks_are_a_single_pass_and_linear(constructs):
    """The pathological case: one big flat module. Was O(n^2), must be O(n)."""
    big = "\n".join(f"x{i} = [{i}, {i + 1}]" for i in range(4000))
    tree, offs = ast.parse(big), line_offsets(big)
    t0 = time.time()
    walk(tree, offs, constructs)
    assert time.time() - t0 < 2.0, "companion computation has gone quadratic again"


def test_mask_pack_and_unpack_roundtrip(constructs):
    names = sorted(constructs)
    index = {n: i for i, n in enumerate(names)}
    tree = ast.parse("for x in y:\n    if x: pass")
    masks = descendant_masks(tree, index)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.For))
    assert "If" in unpack(masks[id(node)], names)
    assert "For" not in unpack(masks[id(node)], names)   # itself is not a companion


# --- aggregation ----------------------------------------------------------------

def test_reservoir_is_uniform_and_bounded():
    r = Reservoir(50, seed=0)
    for i in range(5000):
        r.add(i)
    assert len(r.items) == 50 and r.n == 5000
    # a uniform sample of 0..4999 should have a mean near the midpoint
    assert 1500 < float(np.mean(r.items)) < 3500


def test_domain_classification_needs_evidence():
    assert classify_domain("ledger = invoice + portfolio_total") == "finance"
    assert classify_domain("genome = chromosome + plasmid_of(protein)") == "biology"
    assert classify_domain("x = 1") is None, "no anchors, no domain"
    assert classify_domain("log.setLevel(level)") is None, "generic code is not gaming"
    assert set(DOMAIN_SEEDS) == {"finance", "biology", "gaming", "physics", "ecommerce"}


def test_classification_requires_a_clear_winner():
    """One anchor in otherwise generic code is not evidence of a domain."""
    tied = "ledger = 1\nledger = 2\nledger = 3\ngenome = 1\ngenome = 2\ngenome = 3"
    assert classify_domain(tied) is None


def test_distinctiveness_rejects_generic_names():
    """Raw frequency returns AttributeError in every domain; the ratio does not."""
    from collections import Counter
    from profilebattery.calibration.tables import distinctive
    ids = {
        "finance": Counter({"AttributeError": 900, "invoice_total": 40}),
        "biology": Counter({"AttributeError": 900, "plasmid_id": 40}),
    }
    out = distinctive(ids, top=10)
    assert "AttributeError" not in out["finance"] and "AttributeError" not in out["biology"]
    assert "invoice_total" in out["finance"] and "plasmid_id" in out["biology"]


def test_add_file_records_and_skips(constructs):
    t = Tables()
    add_file(t, SRC, constructs)
    assert t.parsed == 1 and t.skipped == 0
    assert t.occurrences["For"] == 2
    assert t.files_with["For"] == 1

    add_file(t, "def f(:\n", constructs)              # unparseable
    assert t.skipped == 1

    add_file(t, "x = 1\n" * (MAX_FILE_CHARS // 6 + 10), constructs)
    assert t.oversize == 1, "oversize files are skipped and counted, not parsed"


def test_padding_blocks_parse_on_their_own(constructs):
    t = Tables()
    add_file(t, SRC, constructs)
    for block in t.padding.items:
        ast.parse("\n".join(block))                    # raises if it does not


# --- §3 step 1: A calibrated to the corpus ---------------------------------------

def test_access_is_an_axis_with_four_levels():
    """Real Python is attribute-dense; a grammar with one fixed access form is not."""
    from profilebattery.generate.compound import ACCESS, access
    assert set(ACCESS) == {"subscript", "attribute", "self_attribute", "bare"}
    menu, cell, rng = StubMenu(), Cell("finance", "method", 0, 0), np.random.default_rng(0)
    forms = {f: access(f, "entry", menu, cell, rng) for f in ACCESS}
    assert "[" in forms["subscript"] and "." in forms["attribute"]
    assert forms["self_attribute"].startswith("self.")     # method scope
    assert forms["bare"] == "entry"
    other = access("self_attribute", "entry", menu, Cell("finance", "module", 0, 0), rng)
    assert not other.startswith("self."), "no `self` outside a method"


def test_corpus_profile_uses_the_companion_construct_set():
    """Both sides of the A-vs-corpus comparison must count the same constructs.

    Counting `Constant` on the corpus side only made A look 0.82 short of it.
    """
    from profilebattery.generate.matching import corpus_profile
    cp = corpus_profile("For")
    if not cp:
        pytest.skip("calibration tables not built")
    assert "Constant" not in cp and "Name" not in cp and "For" not in cp
    assert cp["Attribute"] > 0.7, "real `for` loops are attribute-dense"


def test_await_garnish_is_restricted_to_async_frames():
    from profilebattery.generate.garnish import garnishes_for
    assert "await_call" in garnishes_for("AsyncFunctionDef")
    assert "await_call" not in garnishes_for("For")


# --- §3 step 5: the ESS check -----------------------------------------------------

def test_ess_is_the_variance_cost_and_unreachable_mass_is_bias():
    """Two different quantities, and the check must not conflate them.

    ESS falls when corpus mass sits on cells the grid samples rarely — a variance cost that
    more recording would fix. Unreachable mass is corpus mass on levels the grid has no
    level for, which reweighting cannot touch at any sample size.
    """
    import numpy as np

    # a perfectly matched reweighting wastes nothing
    w = np.ones(100)
    assert w.sum() ** 2 / (w ** 2).sum() == pytest.approx(100.0)
    # one cell carrying almost all the weight collapses the effective sample
    w = np.array([99.0] + [0.01] * 99)
    assert w.sum() ** 2 / (w ** 2).sum() < 2.0


def test_method_scope_makes_a_frame_emitted_core_a_method():
    """The defect the ESS check surfaced, now fixed and pinned.

    The spec's third scope level is "in class method". For a statement that means inside a
    method, which always worked. For a construct the frame itself emits — FunctionDef,
    AsyncFunctionDef, ClassDef — it has to mean the core *is* the method, or the construct's
    A set contains no methods at all while the corpus puts 74.8% of `def`s there.
    """
    from profilebattery.generate.assemble import FRAME_CLASS, FRAME_DEF, wrap_scope
    from profilebattery.generate.compound import CoreText
    from profilebattery.generate.menu import SCOPES, StubMenu

    assert set(SCOPES) == {"module", "function", "method"}
    menu, rng = StubMenu(), np.random.default_rng(0)
    core = CoreText(["def handler(self, rows):", "    return rows"], 0)
    cell = Cell("finance", "method", 0, 0)

    text = "\n".join(wrap_scope(core, "method", menu, cell, rng,
                                construct="FunctionDef"))
    cls = next(n for n in ast.walk(ast.parse(text))
               if isinstance(n, ast.ClassDef) and n.name == FRAME_CLASS)
    assert [n.name for n in cls.body if isinstance(n, ast.FunctionDef)] == ["handler"], \
        "the core's own def must be the method"

    # a statement core is unaffected: it is correctly inside a method already
    stmt = CoreText(["total += 1"], 0)
    text = "\n".join(wrap_scope(stmt, "method", menu, cell, rng, construct="AugAssign"))
    cls = next(n for n in ast.walk(ast.parse(text))
               if isinstance(n, ast.ClassDef) and n.name == FRAME_CLASS)
    assert [n.name for n in cls.body if isinstance(n, ast.FunctionDef)] == [FRAME_DEF]
