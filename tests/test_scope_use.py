"""Item 95: C2's scope class, which the battery did not have.

A scope position is a use of a name the construct binds, standing later in the enclosing block.
Measured on the pilot battery over 720 prompts per construct: `Assign` **7**, `AugAssign` **4**,
and `AnnAssign`, `ImportFrom`, `NamedExpr`, `FunctionDef` **zero**. So the class was absent for
all six constructs that can have one and C2's scope reading did not exist.

Three separate causes, and these tests pin all three:

  * the generator never emitted a downstream use of the *bound* name -- the `before_use`
    neighbourhood appended a statement with a freshly drawn name instead;
  * `scope_positions` could not report one for `NamedExpr` at all, because a walrus is not a
    statement and `_blocks` is keyed by statement;
  * `_scope_bindings` counted `rows` as bound by `rows[0] = 3`, which mutates and binds nothing.
"""

import ast
import collections

import numpy as np
import pytest

from profilebattery.generate import simple
from profilebattery.generate.align import line_start_offsets
from profilebattery.generate.calibrated_menu import default_menu
from profilebattery.generate.expressions import OPERANDS
from profilebattery.generate.families import build_a_cores
from profilebattery.generate.menu import Cell
from profilebattery.generate.positions import (SCOPE_BINDERS, _bound_in_target,
                                               _enclosing_statement, scope_positions)
from profilebattery.generate.scopeuse import (CORPUS_USES, bound_names, trailing_levels,
                                              uses_for)

CELL = Cell("gaming", "module", 0, 0)


@pytest.fixture(scope="module")
def menu():
    return default_menu()


# --- the binding rule -------------------------------------------------------------------

@pytest.mark.parametrize("src,expected", [
    ("a_head = 1", ["a_head"]),
    ("a, b = x, y", ["a", "b"]),
    ("*rest, = xs", ["rest"]),
    ("a_head = a_tail = 1", ["a_head", "a_tail"]),
    ("tally += 1", ["tally"]),
    ("n: int = 3", ["n"]),
    # the two that were wrong: a subscript or attribute target binds nothing
    ("rows[0] = 3", []),
    ("rows[0] += 1", []),
    ("head.field = 3", []),
])
def test_only_real_bindings_count(src, expected):
    node = ast.parse(src).body[0]
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    got = [n for t in targets for n in _bound_in_target(t)]
    assert got == expected


# --- the finder had to reach a non-statement node ---------------------------------------

def test_namedexpr_scope_use_is_found():
    """A walrus is not a statement, so `_blocks` had no entry and the finder returned []."""
    src = "total = (w_seen := 5)\nemit_all(w_seen)\n"
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.NamedExpr))
    got = scope_positions(node, tree, line_start_offsets(src))
    assert [p.bound_name for p in got] == ["w_seen"]


def test_enclosing_statement_picks_the_innermost():
    src = "if (w := f()) > 0:\n    pass\n"
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.NamedExpr))
    assert _enclosing_statement(node, tree) is tree.body[0]


def test_a_statement_is_its_own_enclosing_statement():
    tree = ast.parse("a = 1\n")
    assert _enclosing_statement(tree.body[0], tree) is tree.body[0]


# --- the generator emits a use of the name the construct binds --------------------------

def test_bound_names_reads_the_construct_not_the_enclosing_statement():
    """Without the construct argument this returned the `Assign`'s target, not the walrus's."""
    lines = ["batch = [(w_seen := x) + w_seen, 48]"]
    assert bound_names(lines) == ["batch"]
    assert bound_names(lines, "NamedExpr") == ["w_seen"]


@pytest.mark.parametrize("construct", sorted(CORPUS_USES))
def test_every_construct_the_corpus_asks_for_gets_a_level(construct):
    if construct in simple.SPECS:
        assert any(k.startswith("scope_use")
                   for k in simple.neighbourhoods_for(construct))
    else:
        assert len(trailing_levels(construct)) > 1


def test_constructs_with_no_corpus_uses_get_no_level():
    """`AnnAssign`'s corpus median is 0 downstream uses at every cell, so none are invented."""
    assert uses_for("AnnAssign") == 0
    assert not any(k.startswith("scope_use")
                   for k in simple.neighbourhoods_for("AnnAssign"))
    assert trailing_levels("ClassDef") == ("none",)


def test_non_binders_keep_their_axes_unchanged():
    for c in ("Pass", "Return", "Delete", "Raise"):
        assert not any(k.startswith("scope_use") for k in simple.neighbourhoods_for(c))
    for c in ("BinOp", "Call", "Subscript"):
        assert trailing_levels(c) == ("none",)
        assert "trailing" not in build_a_cores(c, 8, np.random.default_rng(3))[0].axes


# --- the declaration must not drift from the renderer -----------------------------------

SIMPLE_WITH_LEVEL = sorted(c for c in simple.SPECS
                           if any(k.startswith("scope_use")
                                  for k in simple.neighbourhoods_for(c)))


def test_the_set_of_constructs_with_a_level_is_the_corpus_set():
    assert SIMPLE_WITH_LEVEL == ["Assign", "AugAssign", "Import", "ImportFrom"]


@pytest.mark.parametrize("construct", SIMPLE_WITH_LEVEL)
def test_the_non_binding_declaration_matches_what_the_renderer_emits(construct, menu):
    """The allocator trusts this table; if it drifts, the scope-use level silently dies.

    Every construct that gets a level is checked, not only the ones with an entry — so a variant
    that stops binding is caught even where the table is currently empty, as it is for the two
    import constructs.
    """
    spec = simple.SPECS[construct]
    declared = simple.NON_BINDING_VARIANTS.get(construct, frozenset())
    for variant in spec.variants:
        lines = spec.render(variant, OPERANDS[0], menu, CELL, np.random.default_rng(5))
        binds = bool(bound_names(lines, construct))
        assert binds is (variant not in declared), (
            f"{construct}/{variant}: declared non-binding={variant in declared}, "
            f"renders {'a binding' if binds else 'no binding'}")


def test_the_scope_use_level_never_lands_on_a_non_binding_variant():
    """The collision that put all sixteen `AugAssign` scope-use cores on `subscript_target`."""
    for construct, non_binding in simple.NON_BINDING_VARIANTS.items():
        if not any(k.startswith("scope_use")
                   for k in simple.neighbourhoods_for(construct)):
            continue
        cores = simple.build_cores(construct, 24, np.random.default_rng(11))
        bad = [c.axes for c in cores
               if c.axes["neighbourhood"].startswith("scope_use")
               and c.axes["variant"] in non_binding]
        assert not bad, bad


# --- end to end -------------------------------------------------------------------------

@pytest.mark.parametrize("construct", ["Assign", "AugAssign", "ImportFrom",
                                       "NamedExpr", "FunctionDef"])
def test_a_scope_use_core_actually_yields_a_scope_position(construct, menu):
    found = 0
    for core in build_a_cores(construct, 24, np.random.default_rng(5)):
        level = str(core.axes.get("trailing") or core.axes.get("neighbourhood") or "")
        if not level.startswith("scope_use"):
            continue
        text = "\n".join(core.render(menu, CELL, np.random.default_rng(9)).lines)
        tree = ast.parse(text)                       # must also still be valid Python
        node = next((n for n in ast.walk(tree)
                     if type(n).__name__ == construct), None)
        assert node is not None
        found += len(scope_positions(node, tree, line_start_offsets(text)))
    assert found > 0, f"{construct}: no scope position from any scope-use core"


def test_a_scope_use_adds_no_binding_of_its_own(menu):
    """An assignment would have been the obvious shape and binds a further name."""
    core = next(c for c in build_a_cores("Assign", 24, np.random.default_rng(5))
                if c.axes["neighbourhood"].startswith("scope_use"))
    text = "\n".join(core.render(menu, CELL, np.random.default_rng(9)).lines)
    tree = ast.parse(text)
    appended = tree.body[-1]
    assert isinstance(appended, ast.Expr) and isinstance(appended.value, ast.Call)
