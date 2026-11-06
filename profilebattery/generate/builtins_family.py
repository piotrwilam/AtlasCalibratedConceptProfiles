"""Family grammar 5 of 5: builtins, by call shape.

The inventory has two kinds of member and they need different machinery. Families 1-4
target **AST node types**: the anchor is a node, and verification asks whether a node
of that class sits at the anchored position. A builtin is a **name**: the anchor is the
`Name` node, and verification asks whether that identifier is referenced in a *use*
position. Three builtins are neither — `True`, `False` and `None` are keywords in
Python 3 and parse as `Constant`, so they get a third verification path.

The construct list is read mechanically from the `builtins` module at the pinned
version (handout S1), so ~130 constructs cost no spec rows: each name is classified
(function / type / exception / constant) and the call shapes follow from the class.
Arity comes from `inspect.signature` where the C implementation exposes one, and from a
small hint table where it does not.
"""

from __future__ import annotations

import builtins as _builtins
import inspect

import numpy as np

from . import weights as _weights
from .core import Core, spread
from .expressions import OPERANDS, operand
from .hosts import hosts_for, place
from .menu import Cell, Menu

#: added by `site`, not part of the language (handout S1)
SITE_ADDITIONS = ("exit", "quit", "copyright", "credits", "license", "help")

#: arity for builtins whose C signature is not introspectable
ARITY_HINTS = {
    "range": 1, "zip": 2, "isinstance": 2, "issubclass": 2, "getattr": 2, "setattr": 3,
    "delattr": 2, "hasattr": 2, "pow": 2, "divmod": 2, "map": 2, "filter": 2,
    "print": 1, "open": 1, "next": 1, "iter": 1, "compile": 3, "slice": 2, "super": 0,
    "input": 0, "globals": 0, "locals": 0, "vars": 0, "breakpoint": 0, "object": 0,
    "dict": 0, "exec": 1, "eval": 1, "anext": 1, "aiter": 1,
}

CALL_SHAPES = ("bare_call", "one_arg", "full_arity", "kwarg", "as_reference",
               "in_comprehension", "chained_result", "star_args")
EXC_SHAPES = ("raise_it", "isinstance_check", "as_reference", "tuple_member")
CONST_SHAPES = ("compare", "assign", "argument", "default_arg")


def inventory() -> dict[str, str]:
    """Public builtins -> kind. Mechanical, per handout S1."""
    out = {}
    for name in dir(_builtins):
        if name.startswith("_") or name in SITE_ADDITIONS:
            continue
        obj = getattr(_builtins, name)
        if obj is True or obj is False or obj is None:
            out[name] = "keyword_constant"
        elif isinstance(obj, type) and issubclass(obj, BaseException):
            out[name] = "exception"
        elif isinstance(obj, type):
            out[name] = "type"
        elif callable(obj):
            out[name] = "function"
        else:
            out[name] = "constant"
    return out


def arity(name: str) -> int:
    obj = getattr(_builtins, name)
    try:
        sig = inspect.signature(obj)
        n = sum(1 for prm in sig.parameters.values()
                if prm.default is inspect.Parameter.empty
                and prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD))
        return min(n, 3)
    except (ValueError, TypeError):
        return ARITY_HINTS.get(name, 1)


def _args(n, op, menu, cell, rng):
    return ", ".join(operand(op, menu, cell, rng, i % 2) for i in range(n))


def _callable_expr(name, shape, op, menu, cell, rng):
    p = menu.name(cell.domain, "plural", rng)
    a1 = operand(op, menu, cell, rng, 0)
    n = arity(name)
    return {
        "bare_call": f"{name}()" if n == 0 else f"{name}({_args(n, op, menu, cell, rng)})",
        "one_arg": f"{name}({a1})" if n <= 1 else f"{name}({_args(n, op, menu, cell, rng)})",
        "full_arity": f"{name}({_args(max(n, 1), op, menu, cell, rng)})",
        "kwarg": f"{menu.verb(cell.domain, rng)}_one(kind={name})",
        "as_reference": f"{name}",
        "in_comprehension": f"[{name}(row) for row in {p}]" if n >= 1
        else f"[{name}() for row in {p}]",
        "chained_result": f"{name}({a1}).__class__" if n >= 1 else f"{name}().__class__",
        "star_args": f"{name}(*{p})",
    }[shape]


def _exception_expr(name, shape, op, menu, cell, rng):
    a1 = operand(op, menu, cell, rng, 0)
    return {"raise_it": f"{name}",          # placed by the host, e.g. assign/return
            "isinstance_check": f"isinstance({a1}, {name})",
            "as_reference": f"{name}",
            "tuple_member": f"({name}, ValueError)" if name != "ValueError"
            else f"({name}, KeyError)"}[shape]


def _constant_expr(name, shape, op, menu, cell, rng):
    a1 = operand(op, menu, cell, rng, 0)
    return {"compare": f"{a1} is {name}", "assign": f"{name}",
            "argument": f"{menu.verb(cell.domain, rng)}_one({name})",
            "default_arg": f"{name}"}[shape]


#: Shapes that place the builtin's name **immediately after a non-space character**, so the read
#: site sees a different token: `(` in `tuple_member` and `argument`, `=` in `kwarg`, `[` in
#: `in_comprehension`. The same rule as `hosts.HOST_PRECEDES` one level in, and the same reason:
#: `▁ValueError` and the `Value`+`Error` pair the tokeniser produces after `(` are not the same
#: read site, and bare `ValueError` is not in the vocabulary at all (A19/A19a). Excluded rather
#: than space-padded, because `(  ValueError, KeyError)` is not code the corpus contains.
SHAPES_BREAKING_THE_TOKEN = frozenset({"tuple_member", "argument", "kwarg", "in_comprehension"})


def shapes_for(kind: str, name: str | None = None) -> tuple[str, ...]:
    """The shapes for a builtin kind, dropping those that would change its read token.

    `name` is optional only so the existing callers and tests keep working; when it is given
    and identifier-shaped -- which every builtin name is -- the breaking shapes are removed.
    """
    base = (EXC_SHAPES if kind == "exception"
            else CONST_SHAPES if kind in ("constant", "keyword_constant")
            else CALL_SHAPES)
    if not name or not name.strip().isidentifier():
        return base
    kept = tuple(s for s in base if s not in SHAPES_BREAKING_THE_TOKEN)
    return kept if len(kept) >= 2 else base


FRAME_CLASS = "builtin reference"


def build_cores(name: str, n: int, rng: np.random.Generator,
                kind: str | None = None) -> list[Core]:
    kind = kind or inventory()[name]
    shapes = shapes_for(kind, name)
    # the builtin's own name is its surface token, so the host must not change the piece the
    # read site sees: `f(len(xs))` gives `len` where every other host gives `▁len`, and the two
    # are unrelated embeddings (A19/A19a)
    levels = {"host": hosts_for("Call" if kind in ("function", "type") else "Assign", name),
              "shape": shapes, "operand": OPERANDS}
    combos = spread(levels, n, rng,
                    weights=_weights.for_construct(name))
    maker = {"exception": _exception_expr, "constant": _constant_expr,
             "keyword_constant": _constant_expr}.get(kind, _callable_expr)
    target_kind = "constant" if kind == "keyword_constant" else "name"
    cores = []
    for i, axes in enumerate(combos):
        def render(menu: Menu, cell: Cell, r: np.random.Generator, _a=axes):
            from .compound import CoreText
            expr = maker(name, _a["shape"], _a["operand"], menu, cell, r)
            return CoreText(place(_a["host"], expr, menu, cell, r), 0)
        cores.append(Core(core_id=f"{name}/c{i}", construct=name, family="builtin",
                          axes=dict(axes, kind=kind), frame_class=FRAME_CLASS,
                          companions=("Call",) if kind in ("function", "type") else (),
                          render=render, target_kind=target_kind))
    return cores
