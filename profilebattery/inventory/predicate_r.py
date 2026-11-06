"""Predicate R: the exclusion rule, as a checkable predicate rather than judgement.

A constructor is excluded iff it has no concrete syntax of its own (report S6). That
covers five disjoint cases, and every exclusion carries the case that produced it, so
the kept and excluded lists can be re-derived from CPython's files by anyone.
"""

from __future__ import annotations

from .asdl import AsdlType

#: types whose constructors are context markers — no surface syntax at all
CONTEXT_TYPES = ("expr_context",)

#: operator marker classes, folded into the node that carries the operator's syntax
OPERATOR_TYPES = {"boolop": "BoolOp", "operator": "BinOp", "unaryop": "UnaryOp",
                  "cmpop": "Compare"}

#: the module containers: a whole-file wrapper, not a construct in a file
CONTAINER_TYPES = ("mod",)

#: sum constructors that are helpers: they never correspond to their own token sequence
HELPER_CONSTRUCTORS = ("ExceptHandler", "TypeIgnore")

#: pure wrappers: the node unparses to exactly its child's token sequence, so it has no
#: concrete syntax of its own and R excludes it by the same rule as the helpers.
#: `x.method()` as a statement is Expr(Call(...)) and the Expr contributes nothing;
#: `case 1:` is MatchValue(Constant(1)) where `case` and `:` belong to match_case (a
#: product type, already excluded) and `1` to Constant. MatchSingleton is the same case
#: with a raw constant rather than a node as its child, so its check compares against
#: repr(value) instead of unparse(child). Verified construct by construct in
#: tests/test_inventory.py rather than asserted. Excluding these three is what takes the
#: raw 66 constructors to the 63 both documents record.
WRAPPER_CONSTRUCTORS = ("Expr", "MatchValue", "MatchSingleton")


def classify(types: dict[str, AsdlType]) -> tuple[dict[str, str], dict[str, str]]:
    """(kept construct -> owning ASDL type, excluded name -> reason)."""
    kept: dict[str, str] = {}
    excluded: dict[str, str] = {}

    for tname, t in types.items():
        if t.kind == "product":
            # a product type has no constructor of its own: `arguments`, `arg`,
            # `keyword`, `alias`, `withitem`, `comprehension`, `match_case`
            excluded[tname] = "product type: no constructor of its own"
            continue
        if tname in CONTEXT_TYPES:
            for c in t.constructors:
                excluded[c] = f"context marker ({tname})"
            continue
        if tname in OPERATOR_TYPES:
            for c in t.constructors:
                excluded[c] = f"operator marker, folded into {OPERATOR_TYPES[tname]}"
            continue
        if tname in CONTAINER_TYPES:
            for c in t.constructors:
                excluded[c] = f"module container ({tname})"
            continue
        for c in t.constructors:
            if c in HELPER_CONSTRUCTORS:
                excluded[c] = "helper: no token sequence of its own"
            elif c in WRAPPER_CONSTRUCTORS:
                excluded[c] = "pure wrapper: unparses to its child's token sequence"
            else:
                kept[c] = tname
    return kept, excluded


def report(types: dict[str, AsdlType]) -> str:
    kept, excluded = classify(types)
    by_type: dict[str, list[str]] = {}
    for c, t in kept.items():
        by_type.setdefault(t, []).append(c)
    lines = [f"kept {len(kept)} constructs, excluded {len(excluded)} names", ""]
    for t, cs in by_type.items():
        lines.append(f"  {t:12} {len(cs):3}  {', '.join(cs)}")
    lines += ["", "excluded:"]
    for name, why in sorted(excluded.items()):
        lines.append(f"  {name:18} {why}")
    return "\n".join(lines)
