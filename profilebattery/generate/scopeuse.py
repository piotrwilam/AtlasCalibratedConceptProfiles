r"""Downstream uses of a construct-bound name — item 95, C2's scope class.

C2 reads behaviour loss at three classes of position: **obliged**, **governed** and **scope**.
The scope class is a use of a name the construct binds, standing *later in the enclosing block*
— `total = len(xs)` and then, two statements on, `report(total)`. Decision 7 forbids pooling it
with the governed class, so it is read on its own or not at all.

**It was not there.** Measured on the pilot battery (1 Oct), over 720 prompts per construct:
`Assign` had **7** scope positions, `AugAssign` **4**, and `AnnAssign`, `ImportFrom`,
`NamedExpr` and `FunctionDef` **none**. So the class was absent for all six constructs that can
have one, and C2's scope reading did not exist. The earlier figure of "45-90 positions per full
set" for `Assign` was wrong: the garnish carrier that was supposed to supply them fires on only
1-8% of prompts, and when it does it draws its own fresh name rather than the bound one.

**The fix is a core-template change, which is why it had to happen before generation.** The
simple family already had a `before_use` neighbourhood that appends a statement after the core —
but it appends `verb_all(plural_name)` with a freshly drawn name, so it creates a `Call` and no
scope position at all. These levels append a use of *the name the statement actually binds*.

**The bound name is recovered by parsing the rendered lines with `positions._scope_bindings`** —
the same function the position finder uses — rather than by threading a name out of fourteen
render functions. That is deliberate: the generator can then only ever reference a name the
finder will recognise, so the emitted count and the found count cannot disagree. If the
statement binds nothing the level degrades to appending nothing, and the prompt is one the
finder will mark `not applicable`, which is a category and not a count of zero.

**How many uses.** From the proxy corpus (item 83, `BINDING_USES_PROXY.json`), the median number
of uses following the statement, conditional on the room available: `Assign` 1 at one following
statement and **2** from two on; `AugAssign`, `Import` and `ImportFrom` **1** throughout;
`AnnAssign` and `ClassDef` 0. So two levels are enough to cover the corpus: one use, and two.
"""

from __future__ import annotations

import ast

#: the node types a scope-use line creates, for the same data-driven exclusion the
#: neighbourhood and host axes use: a level may never create the construct being measured
CREATES: tuple[str, ...] = ("Call",)

#: construct -> how many downstream uses the corpus median asks for, by room available.
#: Read off `validation/positions/BINDING_USES_PROXY.json`; constructs whose corpus median is
#: zero are absent, and for those the class stays *not applicable* rather than being invented.
CORPUS_USES = {"Assign": 2, "AugAssign": 1, "Import": 1, "ImportFrom": 1,
               "FunctionDef": 1, "AsyncFunctionDef": 1, "NamedExpr": 1}


def bound_names(lines: list[str], construct: str | None = None) -> list[str]:
    """Names the rendered core binds, by the finder's own rule.

    Parsed rather than passed in, so the generator and `positions.scope_positions` cannot
    disagree about what counts as bound. Indentation is stripped because a core's lines may
    already sit inside a frame when this is called.

    **`construct` is not optional in practice, and leaving it out was a bug.** The finder reads
    the bindings of *the construct being measured*, which it reaches with `find_node`; without
    the argument this read the enclosing statement's bindings instead. For a `NamedExpr` placed
    as an assignment's value -- `batch = [(w_seen := x) + w_seen, 48]` -- that returned `batch`,
    the `Assign`'s target, so the generated use was a scope use of the **Assign** and the
    `NamedExpr` set still had none.
    """
    from .positions import _scope_bindings

    src = "\n".join(lines)
    for candidate in (src, "\n".join(l.strip() for l in lines)):
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        out: list[str] = []
        if construct is None:
            for node in tree.body:
                out += _scope_bindings(node)
        else:
            for node in ast.walk(tree):
                if type(node).__name__ == construct:
                    out += _scope_bindings(node)
                    break
        # a name the construct binds *and* that is a plain identifier we can re-emit
        return [n for n in out if n.isidentifier()]
    return []


def use_lines(lines: list[str], n_uses: int, menu, cell, rng,
              indent: str = "", construct: str | None = None) -> list[str]:
    """`n_uses` statements loading a name the rendered statement binds; [] if it binds none.

    Each is an expression statement — a call — so it creates a `Call` and nothing else. An
    assignment would have been the obvious shape and is wrong twice over: it would create an
    `Assign`, which is one of the constructs being measured, and it would bind a further name
    whose own uses the finder would then count.
    """
    names = bound_names(lines, construct)
    if not names or n_uses <= 0:
        return []
    out = []
    for i in range(n_uses):
        name = names[i % len(names)]
        verb = menu.verb(cell.domain, rng)
        out.append(f"{indent}{verb}_all({name})" if i % 2 == 0
                   else f"{indent}{verb}_one({name}, strict=True)")
    return out


def uses_for(construct: str) -> int:
    """The corpus-median number of downstream uses for this construct, 0 where it has none."""
    return CORPUS_USES.get(construct, 0)


#: the levels of the shared `trailing` axis, for families that have no `neighbourhood` axis.
#: The simple family already had one and the scope-use levels went in there; the expression and
#: compound families get this instead, so no existing axis changes for a construct that cannot
#: have a scope class.
TRAILING_NONE = "none"
TRAILING_ONE = "scope_use"
TRAILING_PAIR = "scope_use_pair"


def trailing_levels(construct: str) -> tuple[str, ...]:
    """`(none,)` for a construct with no scope class, plus the levels the corpus asks for."""
    from .positions import SCOPE_BINDERS

    if construct not in SCOPE_BINDERS:
        return (TRAILING_NONE,)
    n = uses_for(construct)
    out = [TRAILING_NONE]
    if n >= 1:
        out.append(TRAILING_ONE)
    if n >= 2:
        out.append(TRAILING_PAIR)
    return tuple(out)


def apply_trailing(level: str, lines: list[str], menu, cell, rng,
                   indent: str = "", construct: str | None = None) -> list[str]:
    """`lines` with the trailing scope uses the level asks for appended."""
    if level == TRAILING_NONE:
        return lines
    uses = 2 if level == TRAILING_PAIR else 1
    return lines + use_lines(lines, uses, menu, cell, rng, indent, construct)
