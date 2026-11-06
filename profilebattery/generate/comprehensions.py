"""Family grammar 4 of 5: comprehensions.

A comprehension is an expression, so it reuses the shared host axis. Its own axes are
what a comprehension is made of: the `source` it iterates, the `condition` filtering
it, and the `element` it builds. Note that a comprehension's `for` is *not* a For
node — it is a `comprehension` helper, excluded by predicate R — so comprehension
cores do not contaminate any loop construct's A set.
"""

from __future__ import annotations

import numpy as np

from . import weights as _weights
from .core import Core, spread
from .hosts import hosts_for, place
from .menu import Cell, Menu

SOURCES = ("name", "range", "call", "items", "zip", "nested")
CONDITIONS = ("none", "single_if", "two_if", "ternary_element")
ELEMENTS = ("name", "subscript", "call", "tuple", "binop")

SPECS = {"ListComp": ("[", "]"), "SetComp": ("{", "}"), "GeneratorExp": ("(", ")"),
         "DictComp": ("{", "}")}
FRAME_CLASS = "comprehension"


def _source(level, menu, cell, rng):
    p = menu.name(cell.domain, "plural", rng)
    return {"name": p, "range": f"range({menu.int_literal(rng)})",
            "call": f"sorted({p})", "items": f"{p}.items()",
            "zip": f"zip({p}, {p}_ref)",
            "nested": f"[q for q in {p}]"}[level]


def _element(level, var, menu, cell, rng):
    f = menu.field(cell.domain, rng)
    return {"name": var, "subscript": f"{var}[{f!r}]",
            "call": f"len({var})", "tuple": f"({var}, {menu.int_literal(rng)})",
            "binop": f"{var} * {menu.int_literal(rng)}"}[level]


def _expr(construct, source, condition, element, menu, cell, rng):
    open_b, close_b = SPECS[construct]
    var = menu.name(cell.domain, "singular", rng)
    target = f"{var}_key, {var}" if source in ("items", "zip") else var
    src = _source(source, menu, cell, rng)
    el = _element(element, var, menu, cell, rng)
    if condition == "ternary_element":
        # the conditional wraps the *value*; for DictComp the `key:` prefix goes on
        # afterwards, or the colon lands inside the parentheses and does not parse
        el = f"({el} if {var} else {menu.int_literal(rng)})"
    if construct == "DictComp":
        key = var if source not in ("items", "zip") else f"{var}_key"
        el = f"{key}: {el}"
    if condition == "ternary_element":
        cond = ""
    elif condition == "single_if":
        cond = f" if {var}"
    elif condition == "two_if":
        cond = f" if {var} if {var} != {menu.int_literal(rng)}"
    else:
        cond = ""
    return f"{open_b}{el} for {target} in {src}{cond}{close_b}"


def build_cores(construct: str, n: int, rng: np.random.Generator) -> list[Core]:
    levels = {"host": hosts_for(construct), "source": SOURCES,
              "condition": CONDITIONS, "element": ELEMENTS}
    combos = spread(levels, n, rng,
                    weights=_weights.for_construct(construct))
    cores = []
    for i, axes in enumerate(combos):
        def render(menu: Menu, cell: Cell, r: np.random.Generator, _a=axes):
            from .compound import CoreText
            expr = _expr(construct, _a["source"], _a["condition"], _a["element"],
                         menu, cell, r)
            return CoreText(place(_a["host"], expr, menu, cell, r), 0)
        cores.append(Core(core_id=f"{construct}/c{i}", construct=construct,
                          family="comprehension", axes=axes,
                          frame_class=FRAME_CLASS, companions=(), render=render))
    return cores
