"""One AST walk per file, yielding a record per construct occurrence.

Scope, nesting depth, span extent, companions and — where the grammar's own axes are
detectable in real code — the core-axis level. That last one is what turns "core weights
balanced for this run" into "corpus-weighted reweighting later is arithmetic": the
weights need the corpus rate of each axis level, and a `for` over `range(...)` versus
over `.items()` is as visible in the corpus as it is in a core.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

#: the AST constructs of the inventory, as predicate R leaves them
def inventory_constructs() -> frozenset[str]:
    from ..inventory.asdl import load
    from ..inventory.build import ASDL
    from ..inventory.predicate_r import classify
    kept, _ = classify(load(str(ASDL)))
    return frozenset(kept)


def companion_constructs() -> frozenset[str]:
    """The construct set for *companion* counting, on both the corpus and battery sides.

    `Name` and `Constant` are inventory members but appear in essentially every span, so
    as companions they carry no information — and counting them on one side only made A
    look 0.82 short of the corpus on `Constant` when the two tables simply disagreed
    about what to count. The companion set is the inventory less those two.
    """
    from ..generate.audit import COUNTED
    return frozenset(COUNTED)


COMPOUND = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.For, ast.AsyncFor,
            ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try, ast.Match)


@dataclass
class Occurrence:
    construct: str
    scope: str
    depth: int
    lines: int
    chars: int
    companions: frozenset[str]
    axis: tuple[tuple[str, str], ...]


def _iter_form(node: ast.For) -> str:
    """The operand axis of the compound family, read off real code."""
    it = node.iter
    if isinstance(it, ast.Call):
        fn = it.func
        name = fn.id if isinstance(fn, ast.Name) else (
            fn.attr if isinstance(fn, ast.Attribute) else "")
        if name in ("range", "enumerate", "zip", "sorted", "reversed"):
            return name if name in ("range", "enumerate", "zip") else "call_result"
        if name == "items":
            return "items"
        return "call_result"
    if isinstance(it, ast.Attribute):
        return "attribute"
    if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
        return "list_literal"
    if isinstance(it, ast.Dict):
        return "dict_literal"
    if isinstance(it, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
        return "comprehension"
    if isinstance(it, ast.Name):
        return "name"
    if isinstance(it, ast.Subscript):
        return "subscript"
    return "other"


def _suite_shape(node) -> str:
    body = getattr(node, "body", []) or []
    orelse = getattr(node, "orelse", []) or []
    if orelse:
        return "else"
    if any(isinstance(s, COMPOUND) for s in body):
        return "companion"
    return "simple" if len(body) <= 1 else "two_stmt"


def axes_of(node, construct: str) -> tuple[tuple[str, str], ...]:
    out = []
    if isinstance(node, (ast.For, ast.AsyncFor)):
        out.append(("operand", _iter_form(node)))
    if isinstance(node, COMPOUND):
        out.append(("suite", _suite_shape(node)))
    if isinstance(node, ast.Call):
        out.append(("call_shape",
                    "no_args" if not node.args and not node.keywords else
                    ("keyword" if node.keywords else "positional")))
    return tuple(out)


def line_offsets(source: str) -> list[int]:
    """Character offset of each line start — computed once per file.

    `ast.get_source_segment` re-splits the whole source on every call, which made the
    first calibration pass quadratic: ~450 occurrences per file, each rebuilding a 10 KB
    string. Offsets turn the span extent into arithmetic.
    """
    offs, pos = [0], 0
    for line in source.splitlines(keepends=True):
        pos += len(line)
        offs.append(pos)
    return offs


def span(node, offs: list[int]) -> tuple[int, int]:
    lo_line = getattr(node, "lineno", 1) - 1
    hi_line = (getattr(node, "end_lineno", None) or getattr(node, "lineno", 1)) - 1
    lo = offs[min(lo_line, len(offs) - 1)] + getattr(node, "col_offset", 0)
    hi = offs[min(hi_line, len(offs) - 1)] + (getattr(node, "end_col_offset", 0) or 0)
    return lo, max(hi, lo)


def descendant_masks(tree: ast.AST, index: dict[str, int]) -> dict[int, int]:
    """id(node) -> bitmask of inventory constructs in its subtree, in one O(n) pass.

    The first version computed companions with `ast.walk(child)` per occurrence, which is
    O(n) per occurrence and therefore O(n²) per file. It was invisible on ordinary files
    and stalled the run for forty minutes on one large generated module. With at most 63
    constructs the set fits in an integer, so the union is a bitwise OR and the whole
    file costs one post-order pass.
    """
    order: list[ast.AST] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(ast.iter_child_nodes(node))
    masks: dict[int, int] = {}
    for node in reversed(order):                      # children before parents
        m = 0
        for child in ast.iter_child_nodes(node):
            bit = index.get(type(child).__name__)
            m |= masks.get(id(child), 0) | (1 << bit if bit is not None else 0)
        masks[id(node)] = m
    return masks


def unpack(mask: int, names: list[str]) -> frozenset[str]:
    return frozenset(names[i] for i in range(len(names)) if mask >> i & 1)


def walk(tree: ast.AST, offs: list[int],
         constructs: frozenset[str]) -> list[Occurrence]:
    """Occurrences with scope, depth, span and companions, in one pass."""
    names = sorted(constructs)
    index = {n: i for i, n in enumerate(names)}
    masks = descendant_masks(tree, index)
    # Scope has four values, not the grid's three: a `def` sitting in a class body is
    # in "class" scope, and the statements inside it are in "method" scope. Collapsing
    # the two would put every method definition at module level.
    out: list[Occurrence] = []
    # (node, scope, compound depth, enclosing class flag)
    stack = [(tree, "module", 0, False)]
    while stack:
        node, scope, depth, in_class = stack.pop()
        for child in ast.iter_child_nodes(node):
            c_scope, c_depth, c_in_class = scope, depth, in_class
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                c_scope = "method" if in_class else "function"
            elif isinstance(child, ast.ClassDef):
                c_scope, c_in_class = "class", True
            if isinstance(child, COMPOUND):
                c_depth = depth + 1
            name = type(child).__name__
            if name in constructs:
                out.append(Occurrence(
                    construct=name, scope=scope, depth=depth,
                    lines=(getattr(child, "end_lineno", child.lineno) or child.lineno)
                    - child.lineno + 1,
                    chars=span(child, offs)[1] - span(child, offs)[0],
                    companions=unpack(masks.get(id(child), 0), names),
                    axis=axes_of(child, name)))
            stack.append((child, c_scope, c_depth, c_in_class))
    return out



