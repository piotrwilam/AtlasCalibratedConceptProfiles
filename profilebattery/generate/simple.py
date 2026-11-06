"""Family grammar 2 of 5: simple statements (no suite of their own).

Axes: `variant` (the statement's own internal form), `operand` (shared with the
expression family) and `neighbourhood` (what sits immediately around it — a simple
statement's analogue of the compound family's suite shape).

Six constructs cannot stand alone: Return and Global need a function, Break and
Continue need a loop, Nonlocal needs a *nested* function, and for all of them that
enclosing frame is a **component, not a companion** — it is grammatically obligatory,
so the co-occurrence cap must exempt it, and the concept profile for these constructs
is unavoidably "construct together with its obligatory frame".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import scopeuse as _scopeuse
from . import weights as _weights
from .core import Core, spread
from .expressions import OPERANDS, operand
from .menu import Cell, Menu

IND = "    "
#: neighbourhood -> node types its own filler creates. Same data-driven exclusion as
#: the host axis: a neighbourhood may never create the construct being measured, or the
#: anchor becomes ambiguous between the filler and the core.
NEIGHBOURHOOD_CREATES: dict[str, tuple[str, ...]] = {
    "alone": (),
    "after_assign": ("Assign", "Subscript"),
    "before_use": ("Call",),
    "between_comments": (),
    # item 95: the two levels that give C2 its scope class. `before_use` already appended a
    # statement after the core, but with a freshly drawn name, so it created a `Call` and no
    # scope position; these append a use of the name the statement actually binds.
    "scope_use": _scopeuse.CREATES,
    "scope_use_pair": _scopeuse.CREATES,
}
NEIGHBOURHOODS = tuple(NEIGHBOURHOOD_CREATES)

#: the scope-use levels are offered only to constructs that bind a name whose uses follow them.
#: Offering them to `Pass` or `Return` would find no bound name, append nothing, and silently
#: over-represent `alone` for every construct that cannot have a scope class at all.
SCOPE_USE_LEVELS = ("scope_use", "scope_use_pair")


def neighbourhoods_for(construct: str) -> tuple[str, ...]:
    from .positions import SCOPE_BINDERS

    out = []
    for k, creates in NEIGHBOURHOOD_CREATES.items():
        if construct in creates:
            continue
        if k in SCOPE_USE_LEVELS:
            # Gated on the *corpus* median, not on whether the grammar permits a use. A name
            # `AnnAssign` binds has a corpus median of 0 downstream uses at every cell, so
            # generating them would invent a position the corpus does not support -- and C2's
            # scope class for `AnnAssign` should stay *not applicable*, which is a finding.
            if construct not in SCOPE_BINDERS:
                continue
            if _scopeuse.uses_for(construct) < (2 if k == "scope_use_pair" else 1):
                continue
        out.append(k)
    return tuple(out)


def _neighbour(kind, lines, menu, cell, rng, construct=None):
    n, p = menu.name(cell.domain, "singular", rng), menu.name(cell.domain, "plural", rng)
    if kind == "alone":
        return lines, 0
    if kind == "after_assign":
        return [f"{n}_head = {p}[0]"] + lines, 1
    if kind == "before_use":
        return lines + [f"{menu.verb(cell.domain, rng)}_all({p})"], 0
    if kind == "between_comments":
        return [f"# {menu.text(cell.domain, rng)}"] + lines + ["# checked"], 1
    if kind in SCOPE_USE_LEVELS:
        uses = 2 if kind == "scope_use_pair" else 1
        # [] when the statement binds nothing re-emittable -- `rows[0] = 3` binds no name --
        # and then the prompt simply has no scope position, which the finder reports as the
        # category *not applicable* rather than as a count of zero
        return lines + _scopeuse.use_lines(lines, uses, menu, cell, rng,
                                           construct=construct), 0
    raise KeyError(kind)


# --- frames (components, not companions) ----------------------------------------

def _frame(kind, lines, menu, cell, rng):
    p = menu.name(cell.domain, "plural", rng)
    n = menu.name(cell.domain, "singular", rng)
    if kind is None:
        return lines, 0
    if kind == "function":
        return [f"def {menu.verb(cell.domain, rng)}_local({p}):"] + \
            [IND + ln for ln in lines], 1
    if kind == "loop":
        return [f"for {n}_head in {p}:"] + [IND + ln for ln in lines], 1
    if kind == "nested_function":
        return ([f"def {menu.verb(cell.domain, rng)}_outer({p}):",
                 IND + f"{n}_tally = 0",
                 IND + f"def {menu.verb(cell.domain, rng)}_inner():"] +
                [IND * 2 + ln for ln in lines], 3)
    raise KeyError(kind)


# --- statements ------------------------------------------------------------------

def _assign(v, op, menu, cell, rng):
    n, p = menu.name(cell.domain, "singular", rng), menu.name(cell.domain, "plural", rng)
    f = menu.field(cell.domain, rng)
    rhs = operand(op, menu, cell, rng, 0)
    return {"single_name": [f"{n}_head = {rhs}"],
            "tuple_unpack": [f"{n}_head, {n}_tail = {rhs}, {rhs}"],
            "subscript_target": [f"{p}[0] = {rhs}"],
            "attribute_target": [f"{n}_head.{f} = {rhs}"],
            "chained": [f"{n}_head = {n}_tail = {rhs}"],
            "starred_target": [f"{n}_head, *{n}_rest = {p}"]}[v]


def _augassign(v, op, menu, cell, rng):
    n, p = menu.name(cell.domain, "singular", rng), menu.name(cell.domain, "plural", rng)
    f = menu.field(cell.domain, rng)
    rhs = operand(op, menu, cell, rng, 0)
    return {"add": [f"{n}_total += {rhs}"], "sub": [f"{n}_total -= {rhs}"],
            "mul": [f"{n}_total *= {rhs}"], "floordiv": [f"{n}_total //= {rhs}"],
            "or_": [f"{n}_flags |= {rhs}"],
            "subscript_target": [f"{p}[0] += {rhs}"],
            "attribute_target": [f"{n}_head.{f} += {rhs}"]}[v]


def _annassign(v, op, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    f = menu.field(cell.domain, rng)
    rhs = operand(op, menu, cell, rng, 0)
    return {"name_builtin": [f"{n}_total: int = {rhs}"],
            "name_generic": [f"{n}_rows: list[int] = []"],
            "name_string": [f'{n}_head: "dict" = {{}}'],
            "attribute": [f"{n}_ratio: float = {rhs}"],
            "no_value": [f"{n}_pending: int"]}[v]


def _return(v, op, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    rhs = operand(op, menu, cell, rng, 0)
    return {"bare": ["return"], "value": [f"return {rhs}"],
            "tuple": [f"return {rhs}, {n}_tally"],
            "conditional": [f"return {rhs} if {n}_ok else {n}_tally"],
            "call": [f"return len({menu.name(cell.domain, 'plural', rng)})"]}[v]


def _raise(v, op, menu, cell, rng):
    msg = menu.text(cell.domain, rng)
    return {"class_only": ["raise ValueError"],
            "with_args": [f'raise ValueError("{msg}")'],
            "from_cause": [f'raise RuntimeError("{msg}") from None'],
            "subclass": [f'raise KeyError("{msg}")']}[v]


def _assert(v, op, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    rhs = operand(op, menu, cell, rng, 0)
    return {"simple": [f"assert {n}_ok"],
            "with_message": [f'assert {n}_ok, "{menu.text(cell.domain, rng)}"'],
            "compare": [f"assert {rhs} > 0"],
            "call": [f"assert len({menu.name(cell.domain, 'plural', rng)})"]}[v]


def _import(v, op, menu, cell, rng):
    return {"single": ["import json"], "multiple": ["import json, csv"],
            "aliased": ["import json as js"], "dotted": ["import os.path"],
            "dotted_aliased": ["import os.path as osp"]}[v]


def _importfrom(v, op, menu, cell, rng):
    return {"single": ["from json import loads"],
            "multiple": ["from json import loads, dumps"],
            "aliased": ["from json import loads as parse"],
            "relative": ["from . import helpers"],
            "parenthesised": ["from json import (loads, dumps)"]}[v]


def _global(v, op, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    return {"single": [f"global {n}_total", f"{n}_total = 0"],
            "multiple": [f"global {n}_total, {n}_count",
                         f"{n}_total = {n}_count = 0"]}[v]


def _nonlocal(v, op, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    return {"single": [f"nonlocal {n}_tally", f"{n}_tally += 1"],
            "multiple": [f"nonlocal {n}_tally", f"{n}_tally = 1"]}[v]


def _delete(v, op, menu, cell, rng):
    n, p = menu.name(cell.domain, "singular", rng), menu.name(cell.domain, "plural", rng)
    f = menu.field(cell.domain, rng)
    return {"name": [f"{n}_head = 0", f"del {n}_head"],
            "subscript": [f"del {p}[0]"],
            "attribute": [f"del {n}_head.{f}"],
            "multiple": [f"{n}_head = {n}_tail = 0", f"del {n}_head, {n}_tail"]}[v]


def _pass(v, op, menu, cell, rng):
    return {"alone": ["pass"], "after_comment": ["# nothing to do", "pass"],
            "with_docstring": ['"""placeholder."""', "pass"],
            "before_use": ["pass"]}[v]


def _break(v, op, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    return {"simple": ["break"],
            "guarded": [f"if {n}_head:", IND + "break"]}[v]


def _continue(v, op, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    return {"simple": ["continue"],
            "guarded": [f"if {n}_head:", IND + "continue"]}[v]


#: Variants whose target **binds no name**: a subscript or attribute target mutates an existing
#: object, so `rows[0] += 1` and `head.field = 3` introduce nothing a later statement could use.
#: Declared here and checked against the renderer by `tests/test_scope_use.py`, so the two cannot
#: drift. A scope-use neighbourhood paired with one of these produces no position at all, and the
#: allocator has to know that or it concentrates the level on exactly those cores -- which is what
#: it did, putting all sixteen of the smoke build's `AugAssign` scope-use cores on
#: `subscript_target` and leaving the construct with zero scope positions.
#: Only constructs that are offered a scope-use level appear here. `Delete` and `AnnAssign` were
#: listed first and both entries were dead: `Delete` binds nothing under any variant and is not a
#: scope binder at all, and `AnnAssign`'s corpus median is 0 so it gets no level. The drift guard
#: in `tests/test_scope_use.py` caught both, which is the argument for having written it.
NON_BINDING_VARIANTS: dict[str, frozenset[str]] = {
    "Assign": frozenset({"subscript_target", "attribute_target"}),
    "AugAssign": frozenset({"subscript_target", "attribute_target"}),
}


def _invalid_combo(construct: str):
    """Reject a scope-use neighbourhood on a variant that binds nothing."""
    non_binding = NON_BINDING_VARIANTS.get(construct, frozenset())
    if not non_binding:
        return None

    def invalid(combo: dict) -> bool:
        return (combo.get("neighbourhood") in SCOPE_USE_LEVELS
                and combo.get("variant") in non_binding)
    return invalid


@dataclass(frozen=True)
class StmtSpec:
    construct: str
    variants: tuple[str, ...]
    render: Callable
    frame: str | None = None
    components: tuple[str, ...] = ()


SPECS: dict[str, StmtSpec] = {
    "Assign": StmtSpec("Assign", ("single_name", "tuple_unpack", "subscript_target",
                                  "attribute_target", "chained", "starred_target"),
                       _assign),
    "AugAssign": StmtSpec("AugAssign", ("add", "sub", "mul", "floordiv", "or_",
                                        "subscript_target", "attribute_target"),
                          _augassign),
    "AnnAssign": StmtSpec("AnnAssign", ("name_builtin", "name_generic", "name_string",
                                        "attribute", "no_value"), _annassign),
    "Return": StmtSpec("Return", ("bare", "value", "tuple", "conditional", "call"),
                       _return, frame="function", components=("FunctionDef",)),
    "Raise": StmtSpec("Raise", ("class_only", "with_args", "from_cause", "subclass"),
                      _raise),
    "Assert": StmtSpec("Assert", ("simple", "with_message", "compare", "call"), _assert),
    "Import": StmtSpec("Import", ("single", "multiple", "aliased", "dotted",
                                  "dotted_aliased"), _import),
    "ImportFrom": StmtSpec("ImportFrom", ("single", "multiple", "aliased", "relative",
                                          "parenthesised"), _importfrom),
    "Global": StmtSpec("Global", ("single", "multiple"), _global, frame="function",
                       components=("FunctionDef", "Assign")),
    "Nonlocal": StmtSpec("Nonlocal", ("single", "multiple"), _nonlocal,
                         frame="nested_function", components=("FunctionDef",)),
    "Delete": StmtSpec("Delete", ("name", "subscript", "attribute", "multiple"),
                       _delete),
    "Pass": StmtSpec("Pass", ("alone", "after_comment", "with_docstring",
                              "before_use"), _pass),
    "Break": StmtSpec("Break", ("simple", "guarded"), _break, frame="loop",
                      components=("For",)),
    "Continue": StmtSpec("Continue", ("simple", "guarded"), _continue, frame="loop",
                         components=("For",)),
}

FRAME_CLASS = "simple statement"


def build_cores(construct: str, n: int, rng: np.random.Generator) -> list[Core]:
    spec = SPECS[construct]
    levels = {"variant": spec.variants, "operand": OPERANDS,
              "neighbourhood": neighbourhoods_for(construct)}
    combos = spread(levels, n, rng,
                    weights=_weights.for_construct(construct),
                    invalid=_invalid_combo(construct))
    cores = []
    for i, axes in enumerate(combos):
        def render(menu: Menu, cell: Cell, r: np.random.Generator, _a=axes):
            from .compound import CoreText
            stmt = spec.render(_a["variant"], _a["operand"], menu, cell, r)
            stmt, _ = _neighbour(_a["neighbourhood"], stmt, menu, cell, r, construct)
            lines, _ = _frame(spec.frame, stmt, menu, cell, r)
            return CoreText(lines, 0)
        cores.append(Core(core_id=f"{construct}/c{i}", construct=construct,
                          family="simple", axes=axes,
                          frame_class=FRAME_CLASS if spec.frame is None
                          else f"requires {spec.frame}",
                          companions=spec.components, render=render))
    return cores
