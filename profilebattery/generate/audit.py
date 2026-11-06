"""Co-occurrence audit: which inventory constructs a set's prompts drag along.

The handout asks for "no non-component construct in more than a stated share of any
A set". Measured on the generated text by parsing, not on the declared companion
tuple, because a hand-written declaration is exactly the thing that goes stale.

Two granularities are reported, because they answer different questions:

* **core share** — in how many of a construct's cores the companion appears. This is
  the design knob: a companion in 7 of 8 cores is baked into the measurement.
* **prompt share** — in what fraction of the set's prompts. This is what the profile
  actually averages over.

And the number that matters most is neither: it is the **A-minus-B difference**. A
companion present at the same rate in A and in B cancels in the concept profile
pi(A) - pi(B); one present in A alone does not. So a high but *matched* share is
harmless, and a low but *asymmetric* one is not. The cap belongs on the difference.
"""

from __future__ import annotations

import ast
import collections
from dataclasses import dataclass

import numpy as np

from .menu import Cell, Menu

#: Inventory node types worth counting. Excludes what predicate R drops (helpers,
#: context markers, containers) and the three that carry no construct information
#: on their own (Name, Constant, Expr).
COUNTED = (
    "For", "AsyncFor", "While", "If", "With", "AsyncWith", "Try", "TryStar",
    "FunctionDef", "AsyncFunctionDef", "ClassDef", "Match", "Return", "Raise",
    "Assert", "Import", "ImportFrom", "Global", "Nonlocal", "Delete", "Pass",
    "Break", "Continue", "Assign", "AugAssign", "AnnAssign", "Call", "Subscript",
    "Attribute", "Compare", "BoolOp", "BinOp", "UnaryOp", "ListComp", "SetComp",
    "DictComp", "GeneratorExp", "Lambda", "IfExp", "Dict", "Set", "List", "Tuple",
    "Starred", "Slice", "NamedExpr", "JoinedStr", "FormattedValue", "Await", "Yield",
)

#: Constructs that are obligatory parts of the target's own syntax, so not companions.
COMPONENTS: dict[str, tuple[str, ...]] = {
    "Match": ("MatchAs", "MatchValue", "MatchSequence", "MatchMapping"),
}


@dataclass
class Share:
    companion: str
    a_cores: int
    n_cores: int
    a_prompts: float
    b_prompts: float

    @property
    def core_share(self) -> float:
        return self.a_cores / self.n_cores

    @property
    def diff(self) -> float:
        return self.a_prompts - self.b_prompts


def present(src: str, exclude: str, from_line: int = 1,
            before: tuple[int, int] | None = None) -> set[str]:
    """Inventory constructs the *core* contributes, excluding shared scaffolding.

    Two things must not be counted, and counting them was over-reporting badly:

    * **the grid's scope frame.** Scope is a grid dimension, so two thirds of every A
      set sits inside a `def` and one third inside a `class` — by design, and
      identically in A, B and C, so it cancels in every contrast. The frame's own
      def/class are named with the reserved prefix and skipped here.
    * **the padding.** The padding pool is corpus-typical code and contains imports,
      calls and assignments of its own. It is drawn from one shared pool for A, B and C,
      so it cancels too; `from_line` scopes the count to the core region.

    `before` scopes the count causally. The model is a causal transformer, so the vector
    read at the construct token cannot contain anything that appears after it: a
    subscript in the loop body and a guard under the header contribute nothing at the
    construct-token site and everything at the last-token site. So the companion share
    that matters is per site — the prefix up to the anchor for the token and span-end
    sites, the whole core region for the last-token site.

    What remains is what the core itself drags — the number the design knob controls.
    """
    from .assemble import FRAME_CLASS as _FC, FRAME_DEF as _FD

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        name = type(node).__name__
        if name not in COUNTED or name == exclude:
            continue
        if getattr(node, "lineno", 0) < from_line:
            continue
        if before is not None and \
                (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)) >= before:
            continue
        if name in ("FunctionDef", "AsyncFunctionDef", "ClassDef") and \
                getattr(node, "name", None) in (_FD, _FC):
            continue
        out.add(name)
    return out


#: the three recorded sites, and how each scopes the companion count
SITES = ("token", "span_end", "last")


def scope_for(prompt, site: str) -> tuple[int, tuple[int, int] | None]:
    """(from_line, before) for one prompt at one site."""
    if site == "token":
        return prompt.body_start, (prompt.anchor_lineno, prompt.anchor_col)
    if site == "span_end":
        end = prompt.span_end_lineno or prompt.anchor_lineno
        return prompt.body_start, (end + 1, 0)
    return prompt.body_start, None


def audit(construct: str, a_prompts: list[tuple[str, int]],
          b_prompts: list[tuple[str, int]], per_core: list[set[str]]) -> list[Share]:
    """Companion shares for one construct's A set, against its B set.

    Prompts are (text, body_start) pairs so the count can skip the padding.
    """
    n_cores = len(per_core)
    a_count = collections.Counter()
    for src, start in a_prompts:
        a_count.update(present(src, construct, start))
    b_count = collections.Counter()
    for src, start in b_prompts:
        b_count.update(present(src, construct, start))
    core_count = collections.Counter()
    for s in per_core:
        core_count.update(s)

    comps = set(COMPONENTS.get(construct, ()))
    out = []
    for companion in sorted(set(a_count) | set(b_count)):
        if companion in comps:
            continue
        out.append(Share(companion, core_count[companion], n_cores,
                         a_count[companion] / max(len(a_prompts), 1),
                         b_count[companion] / max(len(b_prompts), 1)))
    return sorted(out, key=lambda s: -abs(s.diff))
