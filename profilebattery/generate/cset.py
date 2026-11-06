"""The C set — the yardstick, generated rather than harvested.

Decision 2: corpus-typical code sampled **without regard to construct content**, one shared
pool, no construct exclusion of any kind. Signal-freeness is not claimed; C buys
*typicality*, and construct specificity comes from A - B and from the sibling contrasts.

Generated, not harvested, and that is the substantive choice. Harvested corpus code would
differ from A and B in formatting, identifier style and comment density — all of which the
formatter normalises within the battery — so the yardstick would carry that difference into
every profile. Generating C from the same menu and the same formatter makes the yardstick
differ from the battery in exactly one respect: no construct is anchored in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import simple
from .assemble import PAD_LINES_PER_LEVEL, Prompt, fmt, wrap_scope
from .compound import BODIES, CoreText, _ind
from .menu import Cell, Menu

#: statement kinds C draws from — the battery's own material, with no construct targeted
STATEMENT_KINDS = ("assign", "augassign", "call", "attribute_read", "subscript_read",
                   "compare", "guard", "loop", "import", "return_like", "docstring")
MIN_STATEMENTS, MAX_STATEMENTS = 3, 9


@dataclass
class CPrompt:
    text: str
    cell: Cell
    seed: int
    n_statements: int
    formatted: bool
    valid: bool
    reason: str = ""


def _statement(kind: str, menu: Menu, cell: Cell, rng: np.random.Generator) -> list[str]:
    n = menu.name(cell.domain, "singular", rng)
    p = menu.name(cell.domain, "plural", rng)
    f = menu.field(cell.domain, rng)
    v = menu.verb(cell.domain, rng)
    if kind == "assign":
        return [f"{n}_value = {menu.int_literal(rng)}"]
    if kind == "augassign":
        return [f"{menu.name(cell.domain, 'total', rng)} += {menu.int_literal(rng)}"]
    if kind == "call":
        return [f"{v}_all({p})"]
    if kind == "attribute_read":
        return [f"{n}_flag = {n}_head.{f}"]
    if kind == "subscript_read":
        return [f"{n}_first = {p}[0]"]
    if kind == "compare":
        return [f"{n}_ok = {menu.name(cell.domain, 'total', rng)} > "
                f"{menu.int_literal(rng)}"]
    if kind == "guard":
        return [f"if {n}_ok:", _ind([f"{v}_one({p})"])[0]]
    if kind == "loop":
        return [f"for {n}_item in {p}:", _ind([f"{v}_one({n}_item)"])[0]]
    if kind == "import":
        return ["import json"]
    if kind == "return_like":
        return [f"{n}_result = {v}_all({p}) or None"]
    if kind == "docstring":
        return [f'"""{menu.text(cell.domain, rng)}."""']
    raise KeyError(kind)


def build_one(menu: Menu, cell: Cell, seed: int,
              kinds: tuple[str, ...] = STATEMENT_KINDS) -> CPrompt:
    rng = np.random.default_rng(seed)
    k = int(rng.integers(MIN_STATEMENTS, MAX_STATEMENTS + 1))
    lines: list[str] = []
    for _ in range(k):
        kind = kinds[int(rng.integers(len(kinds)))]
        if kind == "docstring" and lines:
            kind = "assign"          # a docstring is only a docstring first
        lines.extend(_statement(kind, menu, cell, rng))
    body = wrap_scope(CoreText(lines, 0), cell.scope, menu, cell, rng)
    if cell.padding:
        pad = menu.padding(cell.domain, cell.padding * PAD_LINES_PER_LEVEL, rng)
        pad_src, pad_ok = fmt("\n".join(pad) + "\n")
        src, ok = fmt("\n".join(pad + [""] + body) + "\n")
        ok = ok and pad_ok and src.startswith(pad_src.rstrip("\n"))
    else:
        src, ok = fmt("\n".join(body) + "\n")
    p = CPrompt(text=src, cell=cell, seed=seed, n_statements=k, formatted=ok,
                valid=False)
    if not ok:
        p.reason = "formatter failed or padding boundary moved"
        return p
    import ast
    try:
        ast.parse(src)
    except SyntaxError as e:
        p.reason = f"parse: {e.msg}"
        return p
    p.valid = True
    return p
