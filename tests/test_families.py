"""Tests for families 2-5, and the properties shared across all five.

The invariants: every (core x cell) assembles into parseable code with a resolvable
anchor; the anchor lands on the core's construct and not on the grid's scope frame or
on the padding; cores are structurally distinct and cover their axes; and no axis level
places a construct where its own grammar forbids it.
"""

import ast
import itertools

import numpy as np
import pytest

from profilebattery.generate import builtins_family as bf
from profilebattery.generate import comprehensions as comp
from profilebattery.generate import compound, expressions, simple
from profilebattery.generate.assemble import FRAME_CLASS, FRAME_DEF, assemble
from profilebattery.generate.audit import present
from profilebattery.generate.hosts import HOSTS, hosts_for
from profilebattery.generate.menu import DOMAINS, SCOPES, Cell, StubMenu

N_CORES = 8
CELLS = [Cell(d, s, p, 0) for d, s, p in itertools.product(DOMAINS, SCOPES, (0, 1, 2))]
FAMILIES = {"compound": compound, "simple": simple, "expressions": expressions,
            "comprehensions": comp}
ALL_NODE_CONSTRUCTS = [(fam, c) for fam, mod in FAMILIES.items() for c in mod.SPECS]


@pytest.fixture(scope="module")
def menu():
    return StubMenu()


@pytest.mark.parametrize("family,construct", ALL_NODE_CONSTRUCTS)
def test_every_core_and_cell_assembles(family, construct, menu):
    mod = FAMILIES[family]
    for core in mod.build_cores(construct, N_CORES, np.random.default_rng(11)):
        for cell in CELLS:
            p = assemble(core, menu, cell, seed=7)
            assert p.valid, f"{core.core_id} {cell.key()}: {p.reason}\n{p.text}"
            assert p.anchor_lineno >= p.body_start, "anchor landed in the padding"
            ast.parse(p.text)


@pytest.mark.parametrize("family,construct", ALL_NODE_CONSTRUCTS)
def test_cores_are_distinct_and_cover_their_axes(family, construct):
    cores = FAMILIES[family].build_cores(construct, N_CORES, np.random.default_rng(11))
    assert len({tuple(c.axes.values()) for c in cores}) == N_CORES
    for axis in cores[0].axes:
        assert len({c.axes[axis] for c in cores}) >= 2, f"{construct}/{axis} is constant"


@pytest.mark.parametrize("family,construct", ALL_NODE_CONSTRUCTS)
def test_anchor_is_the_surface_token_inside_the_outermost_target(family, construct, menu):
    """The anchor is the construct's own surface token, not the node's start.

    An `Assign` node begins at its target and a `Call` node at its function, so anchoring on
    `col_offset` reads the identifier before the `=` or the `(`. The anchor must therefore
    sit *within* the outermost target node's span and carry that construct's surface token.
    """
    from profilebattery.generate.align import char_offset, line_start_offsets, node_span

    cls = getattr(ast, construct)
    for core in FAMILIES[family].build_cores(construct, 4, np.random.default_rng(3)):
        p = assemble(core, menu, Cell("physics", "method", 1, 0), seed=13)
        assert p.valid, p.reason
        starts = line_start_offsets(p.text)
        off = char_offset(p.text, p.anchor_lineno, p.anchor_col)
        spanning = [n for n in ast.walk(ast.parse(p.text))
                    if isinstance(n, cls)
                    and getattr(n, "name", None) not in (FRAME_DEF, FRAME_CLASS)
                    and node_span(n, starts)[0] <= off < node_span(n, starts)[1]]
        assert spanning, f"{core.core_id}: anchor is outside every {construct} span"
        token = p.flags.get("surface_token")
        if p.flags.get("surface_token_found") and token:
            assert p.text[off:off + len(token)] == token, \
                f"{core.core_id}: anchor does not carry {token!r}"


def test_no_host_creates_its_own_target():
    """The data-driven exclusion that keeps the anchor off the host scaffolding."""
    for construct in expressions.SPECS:
        for host in hosts_for(construct):
            assert construct not in HOSTS[host][1], f"{host} creates {construct}"


def test_host_rotation_keeps_compound_companions_at_one_eighth(menu):
    """Eight hosts over eight cores puts each host companion at ~0.125."""
    for construct in ("BinOp", "Compare", "Subscript"):
        n = hits = 0
        for core in expressions.build_cores(construct, N_CORES,
                                            np.random.default_rng(11)):
            for cell in CELLS:
                p = assemble(core, menu, cell, seed=7)
                n += 1
                hits += "If" in present(p.text, construct, p.body_start)
        assert hits / n <= 0.15, f"{construct}: If at {hits / n:.3f}"


def test_scope_frame_and_padding_are_not_counted_as_companions(menu):
    """Both are shared across A, B and C, so both cancel and must be excluded."""
    core = expressions.build_cores("BinOp", 1, np.random.default_rng(0))[0]
    p = assemble(core, menu, Cell("gaming", "method", 2, 0), seed=5)
    found = present(p.text, "BinOp", p.body_start)
    assert "ClassDef" not in found and "FunctionDef" not in found
    assert "Import" not in found, "padding leaked into the companion count"


# --- family 5: builtins ---------------------------------------------------------

def test_builtin_inventory_is_mechanical():
    inv = bf.inventory()
    assert 120 < len(inv) < 200
    assert inv["len"] == "function" and inv["int"] == "type"
    assert inv["ValueError"] == "exception" and inv["None"] == "keyword_constant"
    assert not any(n.startswith("_") for n in inv)
    assert not set(inv) & set(bf.SITE_ADDITIONS)


@pytest.mark.parametrize("name", ["len", "range", "zip", "isinstance", "getattr",
                                  "int", "dict", "ValueError", "None", "Ellipsis"])
def test_builtin_cores_assemble_and_anchor(name, menu):
    kind = bf.inventory()[name]
    for core in bf.build_cores(name, N_CORES, np.random.default_rng(11), kind):
        assert core.target_kind in ("name", "constant")
        for cell in CELLS[:9]:
            p = assemble(core, menu, cell, seed=7)
            assert p.valid, f"{core.core_id} {cell.key()}: {p.reason}\n{p.text}"
            assert p.anchor_lineno >= p.body_start


def test_arity_uses_signature_then_hints():
    assert bf.arity("len") == 1
    assert bf.arity("zip") == 2          # no introspectable signature -> hint table
    assert bf.arity("isinstance") == 2
    assert bf.arity("dict") == 0
