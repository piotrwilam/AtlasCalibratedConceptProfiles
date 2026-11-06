r"""The two-use sets (D1, G5) -- which turn out to need no new construct sets at all.

D1 asks for "the two uses of `for` (statement, comprehension) and of `if` (statement,
conditional expression) as separate A sets, each with a B set containing no use of the other
kind, with their own core ids", and for statement uses to be placed "only at indented positions,
so that both uses read the same token piece". Measured on the battery as it stands, the first
two requirements are already met and the third is met by *reading* rather than by generating.

**Separate A sets with their own core ids: already so.** `For` and `ListComp` /
`GeneratorExp` / `DictComp` / `SetComp` are separate constructs with separate cores -- `For/c0`
against `ListComp/c0` -- and so are `If` and `IfExp`. Nothing needs adding.

**A B set free of the other use: already so, at read time.** Each B set carries the other uses
explicitly, as `other_role:sibling:*` groups -- `For_B` holds 41 `GeneratorExp`, 37 `ListComp`,
36 `AsyncFor` and 35 `DictComp` prompts of 720; `If_B` holds 108 `IfExp` of 720 (exact). The
kernel drops them by name, and `Kernel.contrast` already does so by default (`drop=":sibling:"`),
so the sibling-free B *is* the B with no use of the other kind.

**The same token piece: by restricting the scope, not by changing the battery.** This is the one
real confound and it is measured. A comprehension's `for` and an `IfExp`'s `if` are always
mid-expression, so they are always preceded by a space and always read `▁for` / `▁if` -- 720 of
720 at every scope. A *statement* `for` or `if` reads `for` / `if` at **module** scope, where it
begins its line, and `▁for` / `▁if` at function and method scope, where it is indented: 240 of
720 against 480 of 720 (exact, measured on the pilot battery). So a third of the statement
prompts read a different token piece from every comprehension prompt, and a difference between
the two uses could be the piece rather than the use for that third.

The remedy is the per-scope accumulators of D1, which exist for exactly this reason: read the
two-use comparison on the **indented scopes only**, where both uses read `▁for`. That costs two
thirds of the prompts per set rather than a change to the battery, and it leaves the main
reading of `For` -- which needs its module-scope stratum, the cell weights being
corpus-calibrated -- untouched. Restricting the battery instead would have damaged the reading
that matters most to protect one that does not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwoUsePair:
    token: str
    statement: str
    expression: tuple[str, ...]
    scopes: tuple[str, ...]
    note: str = ""


#: The pairs D1 names, with the scopes at which both members read the same token piece.
PAIRS: tuple[TwoUsePair, ...] = (
    TwoUsePair(
        "for", "For", ("ListComp", "GeneratorExp", "DictComp", "SetComp"),
        ("function", "method"),
        "the rarer use is `GeneratorExp` at 0.104 per thousand tokens against `ListComp`'s "
        "0.376 and the statement's much higher rate, so it is the one clause 3 predicts a "
        "split from"),
    TwoUsePair(
        "if", "If", ("IfExp",), ("function", "method"),
        "`If` and `IfExp` bind no name, so neither has a dependent class and P8 cannot be read "
        "for this pair -- recorded in C13 rather than discovered when the matrix is read"),
)

#: scopes at which a statement use is indented, so its anchor piece carries the space marker
INDENTED_SCOPES = ("function", "method")

#: what the read costs: a statement set keeps this share of its prompts
SHARE_KEPT = 480 / 720


def pair_of(construct: str) -> TwoUsePair | None:
    for p in PAIRS:
        if construct == p.statement or construct in p.expression:
            return p
    return None


def scope_keyed_constructs() -> tuple[str, ...]:
    """Sets needing per-scope accumulators: the two-use members and their controls.

    The controls are `GeneratorExp` -- already a member -- and `While`, which shares no token
    with anything and so measures what a scope restriction does to a construct that has no
    two-use question at all.
    """
    out: set[str] = {"While"}
    for p in PAIRS:
        out.add(p.statement)
        out.update(p.expression)
    return tuple(sorted(out))
