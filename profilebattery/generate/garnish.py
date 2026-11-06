"""Companion garnishes: the material that makes a B set match its A set.

The companion audit showed the concept profile carrying large A-minus-B asymmetries —
Subscript at +0.69, Call at +0.66, Compare at +0.62 — because A's bodies and B's shapes
were designed independently. A companion present at equal rates in A and B cancels in
pi(A) - pi(B); one present in A alone does not.

The fix is to match B to A rather than sterilise A: sterilising A would meet the cap by
damaging the fidelity of the declared distribution, which is what corpus-rate rotation
exists to protect. So a B prompt is `garnish(carrier(token))` — the carrier holds the
token in its subtype context, and the garnish carries the companions A has.

Each garnish declares the node types it creates, so one that would create the measured
construct is excluded — the same data-driven rule as hosts and neighbourhoods.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .menu import Cell, Menu

IND = "    "


def _g(fn, *creates):
    return (fn, creates)


GARNISHES: dict[str, tuple[Callable, tuple[str, ...]]] = {
    "subscript": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_pick = "
                                     f"{m.name(c.domain,'plural',r)}[0]"],
                    "Subscript", "Assign"),
    "compare": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_ok = "
                                   f"{b or m.name(c.domain,'total',r)} > "
                                   f"{m.int_literal(r)}"],
                  "Compare", "Assign"),
    "binop": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_sum = "
                                 f"{m.int_literal(r)} + {m.int_literal(r)}"],
                "BinOp", "Assign"),
    "attribute": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_val = "
                                     f"{m.name(c.domain,'singular',r)}_head."
                                     f"{m.field(c.domain,r)}"],
                    "Attribute", "Assign"),
    "call": _g(lambda m, c, r, b=None: [f"{m.verb(c.domain,r)}_all("
                               f"{m.name(c.domain,'plural',r)})"], "Call"),
    "guard": _g(lambda m, c, r, b=None: [f"if {m.name(c.domain,'singular',r)}_ok:",
                                 IND + "pass"], "If", "Pass"),
    "loop": _g(lambda m, c, r, b=None: [f"for {m.name(c.domain,'singular',r)}_item in "
                               f"{m.name(c.domain,'plural',r)}:", IND + "pass"],
               "For", "Pass"),
    "augassign": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'total',r)} += "
                                     f"{m.int_literal(r)}"], "AugAssign"),
    "boolop": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_both = "
                                  f"{b or (m.name(c.domain,'singular',r) + '_ok')} and "
                                  f"{m.name(c.domain,'singular',r)}_alt"],
                 "BoolOp", "Assign"),
    "unaryop": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_bad = not "
                                   f"{b or (m.name(c.domain,'singular',r) + '_ok')}"],
                  "UnaryOp", "Assign"),
    "list": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'plural',r)}_batch = "
                               f"[{m.int_literal(r)}, {m.int_literal(r)}]"],
               "List", "Assign"),
    "dict": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_spec = "
                               f"{{{m.field(c.domain,r)!r}: {m.int_literal(r)}}}"],
               "Dict", "Assign"),
    "tuple": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_pair = "
                                f"({m.int_literal(r)}, {m.int_literal(r)})"],
                "Tuple", "Assign"),
    "assign": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_seed = "
                                 f"{m.int_literal(r)}"], "Assign"),
    "pass": _g(lambda m, c, r, b=None: ["pass"], "Pass"),
    "listcomp": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'plural',r)}_kept = "
                                   f"[q for q in {m.name(c.domain,'plural',r)}]"],
                   "ListComp", "Assign"),
    "genexp": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'total',r)} = "
                                f"sum(q for q in {m.name(c.domain,'plural',r)})"],
                 "GeneratorExp", "Call", "Assign"),
    "fstring": _g(lambda m, c, r, b=None: [f'{m.name(c.domain,"singular",r)}_label = '
                                  f'f"{{{b or m.name(c.domain,"total",r)}}}"'],
                  "JoinedStr", "FormattedValue", "Assign"),
    "starred": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'plural',r)}_merged = "
                                  f"[*{m.name(c.domain,'plural',r)}]"],
                  "Starred", "List", "Assign"),
    "slice": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'plural',r)}_head = "
                               f"{m.name(c.domain,'plural',r)}[0:2]"],
                "Slice", "Subscript", "Assign"),
    "ifexp": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_pick = "
                                f"{m.int_literal(r)} if "
                                f"{m.name(c.domain,'singular',r)}_ok else "
                                f"{m.int_literal(r)}"], "IfExp", "Assign"),
    "lambda": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_key = "
                                 f"lambda q: q"], "Lambda", "Assign"),
    "set": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'plural',r)}_seen = "
                              f"{{{m.int_literal(r)}, {m.int_literal(r)}}}"],
              "Set", "Assign"),
    "namedexpr": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'total',r)} = "
                                    f"({m.name(c.domain,'singular',r)}_seen := "
                                    f"{m.int_literal(r)})"], "NamedExpr", "Assign"),
    "import": _g(lambda m, c, r, b=None: ["import json"], "Import"),
    "setcomp": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'plural',r)}_uniq = "
                                  f"{{q for q in {m.name(c.domain,'plural',r)}}}"],
                  "SetComp", "Assign"),
    "dictcomp": _g(lambda m, c, r, b=None: [f"{m.name(c.domain,'singular',r)}_map = "
                                   f"{{q: q for q in "
                                   f"{m.name(c.domain,'plural',r)}}}"],
                   "DictComp", "Assign"),
    # Assign-free variants. Every garnish above binds its result, which makes it
    # unusable when the measured construct *is* Assign — and Assign is the one companion
    # no garnish can avoid otherwise. Passing the value to a call instead of binding it
    # is the natural Assign-free form, and it reads like real logging code.
    "attribute_call": _g(lambda m, c, r, b=None: [f"{m.verb(c.domain,r)}_all("
                                         f"{m.name(c.domain,'singular',r)}_head."
                                         f"{m.field(c.domain,r)})"],
                         "Attribute", "Call"),
    "subscript_call": _g(lambda m, c, r, b=None: [f"{m.verb(c.domain,r)}_all("
                                         f"{m.name(c.domain,'plural',r)}[0])"],
                         "Subscript", "Call"),
    "compare_call": _g(lambda m, c, r, b=None: [f"{m.verb(c.domain,r)}_all("
                                       f"{b or m.name(c.domain,'total',r)} > "
                                       f"{m.int_literal(r)})"], "Compare", "Call"),
    # a literal becomes the binding: a BinOp of a Name and a literal creates the same node
    # types as one of two literals, so the fit is untouched -- and this garnish carries the
    # highest post probability of any that `Assign` is fitted with (0.07), which is why it is
    # worth taking. Measured consequence in the smoke build: the scope class goes from zero.
    "binop_call": _g(lambda m, c, r, b=None: [f"{m.verb(c.domain,r)}_all("
                                     f"{b or m.int_literal(r)} + {m.int_literal(r)})"],
                     "BinOp", "Call"),
    # Pass-free block variants: `guard` and `loop` both need a body, and `pass` as the
    # filler pushes Pass up wherever a block is added. A statement body avoids it.
    "guard_assign": _g(lambda m, c, r, b=None: [f"if {m.name(c.domain,'singular',r)}_ok:",
                                        IND + f"{m.name(c.domain,'total',r)} += 1"],
                       "If", "AugAssign"),
    "loop_assign": _g(lambda m, c, r, b=None: [f"for {m.name(c.domain,'singular',r)}_item in "
                                      f"{m.name(c.domain,'plural',r)}:",
                                       IND + f"{m.name(c.domain,'total',r)} += 1"],
                      "For", "AugAssign"),
    "guard_call": _g(lambda m, c, r, b=None: [f"if {m.name(c.domain,'singular',r)}_ok:",
                                      IND + f"{m.verb(c.domain,r)}_all("
                                      f"{m.name(c.domain,'plural',r)})"],
                     "If", "Call"),
    # Break and Continue need *a* loop, not the measured one: a `while` supplies them
    # for a `For` target, which is otherwise unreachable.
    "while_break": _g(lambda m, c, r, b=None: [f"while {m.name(c.domain,'singular',r)}_ok:",
                                       IND + "break"], "While", "Break"),
    "while_continue": _g(lambda m, c, r, b=None: [f"while {m.name(c.domain,'singular',r)}_ok:",
                                          IND + f"{m.name(c.domain,'total',r)} += 1",
                                          IND + "continue"],
                         "While", "Continue", "AugAssign"),
    # assert-borne variants: no Assign and no Call, for targets where both are scarce
    "attribute_assert": _g(lambda m, c, r, b=None: [f"assert {m.name(c.domain,'singular',r)}"
                                            f"_head.{m.field(c.domain,r)}"],
                           "Attribute", "Assert"),
    "subscript_assert": _g(lambda m, c, r, b=None: [f"assert {m.name(c.domain,'plural',r)}[0]"],
                           "Subscript", "Assert"),
    # Every other If supplier drags a companion the target may not have (Pass, Call,
    # AugAssign), which the overshoot guard then clips. Rebinding a name in the body
    # leaves only Assign, which is exempt from the cap, so these are the unconstrained
    # suppliers of If and For.
    "guard_rebind": _g(lambda m, c, r, b=None: [f"if {m.name(c.domain,'singular',r)}_ok:",
                                        IND + f"{m.name(c.domain,'singular',r)}_ok = "
                                        f"False"], "If", "Assign"),
    "loop_rebind": _g(lambda m, c, r, b=None: [f"for {m.name(c.domain,'singular',r)}_item in "
                                      f"{m.name(c.domain,'plural',r)}:",
                                       IND + f"{m.name(c.domain,'singular',r)}_last = "
                                       f"{m.name(c.domain,'singular',r)}_item"],
                      "For", "Assign"),
    "while_rebind": _g(lambda m, c, r, b=None: [f"while {m.name(c.domain,'singular',r)}_ok:",
                                        IND + f"{m.name(c.domain,'singular',r)}_ok = "
                                        f"False"], "While", "Assign"),
    "break_rebind": _g(lambda m, c, r, b=None: [f"while {m.name(c.domain,'singular',r)}_ok:",
                                        IND + f"{m.name(c.domain,'singular',r)}_ok = "
                                        f"False", IND + "break"],
                       "While", "Break", "Assign"),
    # Return cannot stand outside a function, so its supplier brings one. Needed by
    # ClassDef, whose A set carries Return through its method bodies.
    # `await` needs an async frame, so this garnish is usable only for constructs that
    # already supply one — AsyncFunctionDef, AsyncFor, AsyncWith. The corpus puts Await in
    # 0.84 of `async def` spans; without a supplier that gap is unreachable.
    "await_call": _g(lambda m, c, r, b=None: [f"await {m.verb(c.domain,r)}_all("
                                     f"{m.name(c.domain,'plural',r)})"],
                     "Await", "Call"),
    "def_return": _g(lambda m, c, r, b=None: [f"def {m.verb(c.domain,r)}_helper("
                                     f"{m.name(c.domain,'plural',r)}):",
                                      IND + f"return {m.name(c.domain,'plural',r)}"],
                     "FunctionDef", "Return"),
}

#: companion -> every garnish that supplies it. A multimap, not a single choice: which
#: supplier is usable depends on the target construct and on how much headroom each
#: one's side-effects leave, so the fitter picks per construct rather than globally.
SUPPLIERS: dict[str, tuple[str, ...]] = {}
for _name, (_fn, _creates) in GARNISHES.items():
    for _c in _creates:
        SUPPLIERS[_c] = SUPPLIERS.get(_c, ()) + (_name,)


#: garnishes that only parse inside an async function
ASYNC_ONLY = ("await_call",)
#: constructs whose cores supply an async frame
ASYNC_FRAMED = ("AsyncFunctionDef", "AsyncFor", "AsyncWith")


def garnishes_for(construct: str) -> tuple[str, ...]:
    """Garnishes usable for a construct: never one that creates it, and `await` only
    where the construct's own core supplies the `async def` it needs."""
    out = []
    for n, (_, creates) in GARNISHES.items():
        if construct in creates:
            continue
        if n in ASYNC_ONLY and construct not in ASYNC_FRAMED:
            continue
        out.append(n)
    return tuple(out)


#: garnishes whose read operand is a scalar, so a bound name substitutes there coherently
READS_A_SCALAR = ("compare", "unaryop", "boolop", "fstring",
                  # the composites the A-side fit actually places *after* the carrier, which
                  # is where a scope use has to be. The simple four above are fitted mostly
                  # `pre`, so on their own they almost never carry a scope use: `Assign`'s
                  # post probabilities total about 0.14 and only `compare_call` of those was
                  # in this list, giving an expected 0.2 scope positions per 24-prompt set --
                  # which is what the smoke build measured as zero.
                  "compare_call", "binop_call")


def render(names: list[str], menu: Menu, cell: Cell,
           rng: np.random.Generator, binding: str | None = None) -> list[str]:
    """Render the chosen garnishes, optionally reading `binding` where one reads a scalar.

    `binding` is supplied on the A side only, and only for the constructs whose dependent
    class is the *scope* class (C2): an assignment, an import, a class or a walrus binds a name
    that later tokens in the block copy, and without a later statement that uses it the class
    has no positions at all -- measured at zero on the pilot battery against a corpus median of
    one to two (`validation/positions/BINDING_USES.json`).

    Substituting the name costs nothing on the companion axis, which is why it is done this way
    rather than by adding a statement: the garnish creates exactly the node types it declares,
    whether it reads a fresh name or a bound one, so A's companion marginals and therefore
    |A - B| are unchanged. Only the identifier differs, and an identifier is not a companion.
    Length is unchanged for the same reason. B renders the same garnish with no binding, so the
    pair stays matched.
    """
    out: list[str] = []
    for n in names:
        fn = GARNISHES[n][0]
        b = binding if (binding and n in READS_A_SCALAR) else None
        out.extend(fn(menu, cell, rng, b))
    return out
