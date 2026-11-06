"""Family grammar 3 of 5: constructs that occupy operand position.

An expression construct has no suite of its own, so its axes are different from the
compound family's: `host` (where the expression sits — shared, see hosts.py),
`variant` (the construct's own internal form: which operator, how many operands,
which subscript shape) and `operand` (what its operands are made of). Rotating the
host across cores is what keeps any single host companion at ~1/8, inside the cap.

Three constructs cannot use the generic hosts, because their grammar restricts where
they may appear: Starred (call argument, display, or assignment target), Slice (only
inside a subscript, which is therefore a *component*, not a companion) and NamedExpr
(needs parentheses in most positions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import scopeuse as _scopeuse
from . import weights as _weights
from .core import Core, spread
from .hosts import hosts_for, place
from .menu import Cell, Menu

OPERANDS = ("names", "ints", "floats", "calls", "subscripts", "attributes", "mixed")


def operand(form: str, menu: Menu, cell: Cell, rng: np.random.Generator,
            which: int = 0) -> str:
    n = menu.name(cell.domain, "singular", rng)
    p = menu.name(cell.domain, "plural", rng)
    f = menu.field(cell.domain, rng)
    return {
        "names": f"{n}_head" if which == 0 else f"{n}_tail",
        "ints": menu.int_literal(rng),
        "floats": menu.float_literal(rng),
        "calls": f"len({p})" if which == 0 else f"sum({p})",
        "subscripts": f"{p}[{menu.int_literal(rng)}]" if which == 0 else f"{p}[0]",
        "attributes": f"{n}_head.{f}" if which == 0 else f"{n}_tail.{f}",
        "mixed": f"{n}_head" if which == 0 else menu.int_literal(rng),
    }[form]


def _binop(v, op_form, menu, cell, rng):
    a, b = operand(op_form, menu, cell, rng, 0), operand(op_form, menu, cell, rng, 1)
    return f"{a} {v} {b}"


def _compare(v, op_form, menu, cell, rng):
    a, b = operand(op_form, menu, cell, rng, 0), operand(op_form, menu, cell, rng, 1)
    if v == "chained":
        c = operand(op_form, menu, cell, rng, 1)
        return f"{a} < {b} < {c}"
    if v in ("in", "not in"):
        return f"{a} {v} {menu.name(cell.domain, 'plural', rng)}"
    return f"{a} {v} {b}"


def _boolop(v, op_form, menu, cell, rng):
    a, b = operand(op_form, menu, cell, rng, 0), operand(op_form, menu, cell, rng, 1)
    if v == "three":
        return f"{a} and {b} and {operand(op_form, menu, cell, rng, 1)}"
    return f"{a} {v} {b}"


def _unaryop(v, op_form, menu, cell, rng):
    a = operand(op_form, menu, cell, rng, 0)
    return f"{v} {a}" if v == "not" else f"{v}{a}"


def _subscript(v, op_form, menu, cell, rng):
    p = menu.name(cell.domain, "plural", rng)
    idx = {"int_index": menu.int_literal(rng),
           "name_index": operand(op_form, menu, cell, rng, 0),
           "negative": "-1",
           "string_key": repr(menu.field(cell.domain, rng)),
           "nested": f"{p}_ref[{menu.int_literal(rng)}]",
           "tuple_index": f"{menu.int_literal(rng)}, {menu.int_literal(rng)}"}[v]
    return f"{p}[{idx}]"


def _attribute(v, op_form, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    f = menu.field(cell.domain, rng)
    return {"single": f"{n}_head.{f}",
            "chained": f"{n}_head.{f}.value",
            "on_call": f"sorted({menu.name(cell.domain, 'plural', rng)}).__class__",
            "on_subscript": f"{menu.name(cell.domain, 'plural', rng)}[0].{f}"}[v]


def _call(v, op_form, menu, cell, rng):
    vb, p = menu.verb(cell.domain, rng), menu.name(cell.domain, "plural", rng)
    a = operand(op_form, menu, cell, rng, 0)
    return {"no_args": f"{vb}_all()",
            "positional": f"{vb}_all({a})",
            "keyword": f"{vb}_all({a}, strict=True)",
            "method": f"{p}.copy()",
            "nested": f"{vb}_all({vb}_one({a}))",
            "star_args": f"{vb}_all(*{p})"}[v]


def _ifexp(v, op_form, menu, cell, rng):
    a, b = operand(op_form, menu, cell, rng, 0), operand(op_form, menu, cell, rng, 1)
    n = menu.name(cell.domain, "singular", rng)
    if v == "nested":
        return f"{a} if {n}_ok else ({b} if {n}_alt else {a})"
    return f"{a} if {n}_ok else {b}"


def _namedexpr(v, op_form, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    a = operand(op_form, menu, cell, rng, 0)
    return {"simple": f"({n}_seen := {a})",
            "in_compare": f"({n}_seen := {a}) > 0",
            "reused": f"({n}_seen := {a}) + {n}_seen"}[v]


def _lambda(v, op_form, menu, cell, rng):
    f = menu.field(cell.domain, rng)
    return {"no_args": "lambda: 0",
            "one_arg": f"lambda row: row[{f!r}]",
            "default": f"lambda row, scale={menu.int_literal(rng)}: scale",
            "two_args": f"lambda row, other: row[{f!r}] - other"}[v]


def _dict(v, op_form, menu, cell, rng):
    f, g = menu.field(cell.domain, rng), menu.field(cell.domain, rng)
    a = operand(op_form, menu, cell, rng, 0)
    return {"empty": "{}",
            "one_pair": f"{{{f!r}: {a}}}",
            "two_pairs": f"{{{f!r}: {a}, {g!r}: {menu.int_literal(rng)}}}",
            "unpack": f"{{**{menu.name(cell.domain, 'plural', rng)}_base}}",
            "nested": f"{{{f!r}: {{{g!r}: {a}}}}}"}[v]


def _set(v, op_form, menu, cell, rng):
    a = operand(op_form, menu, cell, rng, 0)
    return {"one": f"{{{a}}}",
            "two": f"{{{a}, {menu.int_literal(rng)}}}",
            "two_same": f"{{{a}, {a}}}",
            "unpack": f"{{*{menu.name(cell.domain, 'plural', rng)}_base, {a}}}"}[v]


def _list(v, op_form, menu, cell, rng):
    a, b = operand(op_form, menu, cell, rng, 0), operand(op_form, menu, cell, rng, 1)
    return {"empty": "[]", "one": f"[{a}]", "two": f"[{a}, {b}]",
            "nested": f"[[{a}], [{b}]]",
            "unpack": f"[*{menu.name(cell.domain, 'plural', rng)}_base, {a}]"}[v]


def _tuple(v, op_form, menu, cell, rng):
    a, b = operand(op_form, menu, cell, rng, 0), operand(op_form, menu, cell, rng, 1)
    return {"two": f"({a}, {b})", "three": f"({a}, {b}, {a})",
            "nested": f"(({a}, {b}), {a})",
            "singleton": f"({a},)"}[v]


def _starred(v, op_form, menu, cell, rng):
    p = menu.name(cell.domain, "plural", rng)
    return {"in_call": f"*{p}", "in_list": f"*{p}", "in_set": f"*{p}"}[v]


def _slice(v, op_form, menu, cell, rng):
    p = menu.name(cell.domain, "plural", rng)
    lo, hi = menu.int_literal(rng), menu.int_literal(rng)
    inner = {"lower": f"{lo}:", "upper": f":{hi}", "both": f"{lo}:{hi}",
             "step": f"{lo}:{hi}:2", "negative": "-2:"}[v]
    return f"{p}[{inner}]"


@dataclass(frozen=True)
class ExprSpec:
    construct: str
    variants: tuple[str, ...]
    expr: Callable
    hosts: tuple[str, ...] | None = None
    components: tuple[str, ...] = ()


SPECS: dict[str, ExprSpec] = {
    "BinOp": ExprSpec("BinOp", ("+", "-", "*", "/", "//", "%", "**"), _binop),
    "Compare": ExprSpec("Compare", ("==", "!=", "<", ">=", "in", "not in", "is",
                                    "chained"), _compare),
    "BoolOp": ExprSpec("BoolOp", ("and", "or", "three"), _boolop),
    "UnaryOp": ExprSpec("UnaryOp", ("not", "-", "~"), _unaryop),
    "Subscript": ExprSpec("Subscript", ("int_index", "name_index", "negative",
                                        "string_key", "nested", "tuple_index"),
                          _subscript),
    "Attribute": ExprSpec("Attribute", ("single", "chained", "on_call",
                                        "on_subscript"), _attribute),
    "Call": ExprSpec("Call", ("no_args", "positional", "keyword", "method", "nested",
                              "star_args"), _call),
    "IfExp": ExprSpec("IfExp", ("simple", "nested"), _ifexp),
    "NamedExpr": ExprSpec("NamedExpr", ("simple", "in_compare", "reused"), _namedexpr),
    "Lambda": ExprSpec("Lambda", ("no_args", "one_arg", "default", "two_args"), _lambda),
    "Dict": ExprSpec("Dict", ("empty", "one_pair", "two_pairs", "unpack", "nested"),
                     _dict),
    "Set": ExprSpec("Set", ("one", "two", "two_same", "unpack"), _set),
    "List": ExprSpec("List", ("empty", "one", "two", "nested", "unpack"), _list),
    "Tuple": ExprSpec("Tuple", ("two", "three", "nested", "singleton"), _tuple),
    # restricted positions
    "Starred": ExprSpec("Starred", ("in_call", "in_list", "in_set"), _starred,
                        hosts=("call_arg", "list_element")),
    "Slice": ExprSpec("Slice", ("lower", "upper", "both", "step", "negative"), _slice,
                      components=("Subscript",)),
}

FRAME_CLASS = "operand position"


def build_cores(construct: str, n: int, rng: np.random.Generator) -> list[Core]:
    spec = SPECS[construct]
    from ..inventory.tokens import NODE_TOKENS
    hosts = spec.hosts or hosts_for(construct, NODE_TOKENS.get(construct))
    levels = {"host": hosts, "variant": spec.variants, "operand": OPERANDS}
    # item 95: only `NamedExpr` in this family binds a name whose uses follow it. The axis is
    # *omitted* rather than given a single level, so every other construct's cores are
    # unchanged -- a one-level axis is not an axis, and `spread` and the core audit both say so.
    _trailing = _scopeuse.trailing_levels(construct)
    if len(_trailing) > 1:
        levels["trailing"] = _trailing
    combos = spread(levels, n, rng,
                    weights=_weights.for_construct(construct))
    cores = []
    for i, axes in enumerate(combos):
        def render(menu: Menu, cell: Cell, r: np.random.Generator, _a=axes):
            from .compound import CoreText
            expr = spec.expr(_a["variant"], _a["operand"], menu, cell, r)
            lines = place(_a["host"], expr, menu, cell, r)
            return CoreText(_scopeuse.apply_trailing(_a.get("trailing", "none"), lines, menu, cell, r,
                                                     construct=construct), 0)
        cores.append(Core(core_id=f"{construct}/c{i}", construct=construct,
                          family="expression", axes=axes, frame_class=FRAME_CLASS,
                          companions=spec.components, render=render))
    return cores
