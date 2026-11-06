r"""Diagnostic positions: the two classes C2 reads, found from the formatted prompt.

Decision 7 (30 Sept) fixes two classes of diagnostic position per construct.

*Obliged* -- the grammar forces the next token: `in` after `for x`, the `:` closing a header.
*Dependent* -- **a position is dependent if changing the construct's binding would change the
correct token there.** Piotr fixed the wording on 30 Sept as *binding* rather than *header*,
which extends the class from five constructs to ten, and split it in two, because the two halves
answer P1 differently and are never pooled.

*Governed* uses: the binding and the use are both inside the construct, which governs the use --
`for x in xs: ... x`, a `with ... as f` body, a function's parameters, a comprehension's own
target, an `except ... as e` body.

*Scope* uses: the construct binds a name and later tokens in the same block copy it -- `total =
0` then `total += x`, `from os import path` then `path.join`, `def f()` then `f()`. Here the
later use is the construct's *function* rather than something it governs, so it measures name
tracking of a different kind. For the assignment family, which has no body at all, this is the
only dependent reading there is -- which is why the wording change is worth its extra work.

Positions cannot be declared by the generator's templates, because the assembler runs `black`
after it builds a prompt and reformatting moves every line. They are therefore recovered from
the *formatted* text by AST, exactly as the construct's own anchor already is (`align.py`), and
the dependent rule turns out to be general rather than per-construct: `for` binds its target,
`with` binds its `as` name, `def` binds its parameters, a comprehension binds its own target,
an `except` clause binds its name. So one rule -- *names bound in the header, used in the body*
-- covers every construct that has dependent positions at all, and the ones that have none are
identified rather than guessed at.

The obliged class does need a small table, because which literal token the grammar forces is a
fact about each construct. The table says what to look for; where to look for it is general --
between the spans of the node's own children, which is the technique `align.surface_offset`
already uses to find a construct's own keyword.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .align import line_start_offsets, node_span, token_at, token_spans


@dataclass
class Position:
    kind: str                     # "obliged" | "dependent"
    role: str                     # "in", "header_colon", "body_use", "param_use", ...
    char: int                     # character offset in the formatted prompt
    bound_name: str | None = None
    token: int | None = None      # filled by locate()
    distance: int | None = None   # token distance from the construct token
    matched: bool = True          # False when the position could not be resolved

    #: the counterpart position in the matched control set, filled by the control builder (G4)
    control_index: int | None = None

    #: obliged positions only: True where the grammar forces the token and nothing else is
    #: legal, False where it only expects it. None for the dependent classes (E10)
    forced: bool | None = None
    #: the floor the loss is read against: the conditional entropy of the battery's own
    #: continuations in this *slot*, in nats, filled per set by `floors.py` (E10)
    floor_nats: float | None = None
    #: the token index of the slot's start -- the position where the obliged literal would sit
    #: whether or not it is present. The floor is the entropy over what occupies this index
    #: across the set, not over the piece at `token`: that one is the literal by construction
    #: and its entropy is zero, which is what the first implementation measured by mistake.
    #: For `For`'s `in` the slot is the token after the target, which holds `in` when the
    #: target is simple and `,` when it is a tuple -- and the battery emits both.
    slot: int | None = None

    #: for a token read by more than one construct (amendment A10's five piece-matched groups:
    #: `for`, `with`, `def`, `if`, `yield`), which construct's use this mark belongs to. None
    #: where the token is read by one construct only, so a reading can tell the two apart
    #: instead of pooling a shared token's positions (item 10b).
    use: str | None = None

    def as_dict(self) -> dict:
        """The form the battery stores. `class` rather than `kind`, to match C2's wording."""
        return {"class": ("obliged" if self.kind == "obliged"
                          else "scope" if self.role == "scope_use" else "governed"),
                "role": self.role, "token_index": self.token,
                "distance": self.distance, "bound_name": self.bound_name,
                "control_index": self.control_index, "matched": self.matched,
                "forced": self.forced, "floor_nats": self.floor_nats,
                "slot": self.slot, "use": self.use}


#: the three position classes of C2
CLASSES = ("obliged", "governed", "scope")

#: constructs whose header binds a name whose uses the construct *governs* -- the binding and
#: the use both inside it. Declared rather than inferred from `_bindings`' dispatch, so that a
#: battery's emitted marks can be checked against the grammar instead of against themselves.
GOVERNED_BINDERS = frozenset({
    "For", "AsyncFor", "With", "AsyncWith", "FunctionDef", "AsyncFunctionDef",
    "ListComp", "SetComp", "GeneratorExp", "DictComp", "ExceptHandler",
})

#: constructs that bind a name whose uses follow them in the enclosing block
SCOPE_BINDERS = frozenset({
    "Assign", "AugAssign", "AnnAssign", "NamedExpr", "Import", "ImportFrom",
    "FunctionDef", "AsyncFunctionDef", "ClassDef",
})


def declared_classes(construct: str) -> frozenset[str]:
    """Which of C2's three classes this construct's grammar can have at all.

    A class the grammar cannot give is *not applicable* (C2) -- a category, distinct from a
    count of zero, which is a fact about the prompt. Emitting both is what lets a later reading
    tell "this construct has no dependent positions" from "this prompt happened not to get one".
    """
    out = set()
    if OBLIGED_LITERALS.get(construct):
        out.add("obliged")
    if construct in GOVERNED_BINDERS:
        out.add("governed")
    if construct in SCOPE_BINDERS:
        out.add("scope")
    return frozenset(out)


#: the literal tokens each construct's grammar forces, in the order they appear in the header.
#: A construct absent from this table has no obliged position beyond its own surface token.
OBLIGED_LITERALS = {
    "For": ("in", ":"), "AsyncFor": ("in", ":"),
    "While": (":",), "If": (":",), "Try": (":",), "With": (":",), "AsyncWith": (":",),
    "FunctionDef": ("(", ":"), "AsyncFunctionDef": ("(", ":"), "ClassDef": (":",),
    "ListComp": ("for", "in"), "SetComp": ("for", "in"), "GeneratorExp": ("for", "in"),
    "DictComp": ("for", "in"),
    "IfExp": ("if", "else"),
    "Call": ("(", ")"), "Dict": ("}",), "Set": ("}",), "List": ("]",), "Subscript": ("]",),
    "Assign": ("=",), "AnnAssign": (":",), "AugAssign": ("=",),
    "Lambda": (":",), "Assert": (),
    "ExceptHandler": (":",),
}

#: Which obliged roles the grammar *forces* -- and the test is narrower than it first looks
#: (Cowork, 1 Oct): a position is forced only when **no expression precedes it**, because wherever
#: one does the model cannot know the expression is complete. After `if x` the condition may
#: continue (`if x and y`, `if x[0]`); after `range(n)` so may the iterable (`.`, `[`, `+`). So a
#: header's closing `:` is **expected**, not forced, and its floor is the battery's own entropy
#: rather than zero -- which the measurement already said and an earlier draft of this file
#: explained away as a tokenisation artefact: `If`'s loss there is 0.912 nats at 4.19B and the
#: battery's entropy at that slot 0.42 to 1.47 across sets, and neither of those is merging.
#: A closing bracket is expected for the same reason, an expression preceding it.
#:
#: What is left forced is a short list, and it is now complete (item 96b, 1 Oct): the `(` after
#: a `def` name, where only `(` is legal after the identifier, and the `:` where **nothing at all
#: precedes it in the header** -- `try:`, a bare `except:`, and the `else:` and `finally:` of a
#: compound statement. Those colons get their own role, `bare_colon`, because the distinction is
#: not about the character but about whether an expression sits in front of it: `except E:` is
#: `header_colon` and expected, `except:` is `bare_colon` and forced. `else:` and `finally:` have
#: no AST node of their own, so they are found by `bare_colon_positions` scanning the construct's
#: own lines; `try:` and bare `except:` fall out of the header rule, having no child before the
#: colon.
FORCED_ROLES = frozenset({"open_paren", "bare_colon"})

#: the role name each literal is reported under
ROLE_OF = {"in": "in", ":": "header_colon", "(": "open_paren", ")": "close_paren",
           "]": "close_bracket", "}": "close_brace", "for": "comp_for",
           "if": "ternary_if", "else": "ternary_else", "=": "assign_eq"}

#: Constructs whose own `:` has **nothing before it** and is therefore forced. Declared, not
#: inferred, because neither the gap text nor the child spans can tell the cases apart: `class C:`
#: and `def f():` have no child span in front of the colon either, yet both are *expected* --
#: after the class name `(` is legal, after the parameter list `->` is. The honest list is short:
#: `try:`, and `except:` when it carries no exception type (`except E:` is expected). `else:` and
#: `finally:` belong here too and are found separately, having no AST node of their own.
BARE_COLON_CONSTRUCTS = frozenset({"Try"})


def _colon_is_bare(node, kind: str) -> bool:
    """Whether this construct's `:` has no expression in front of it."""
    if kind in BARE_COLON_CONSTRUCTS:
        return True
    return kind == "ExceptHandler" and getattr(node, "type", None) is None


#: **The slot table** (item 96b, E10's remaining piece). A slot is where the grammar *branches*,
#: and the branch point is role-specific -- which is why one rule for every role was wrong.
#:
#:   `literal`                the literal's own token: the branch is *at* the literal, because the
#:                            alternatives are what else could stand in that position. Right for
#:                            every role but one. `_preceding_end` and `elt_end`, both tried
#:                            first, are wrong for a reason worth recording: Python's AST does not
#:                            record parentheses, so a parenthesised element's span ends *inside*
#:                            its own closing `)` and the slot landed on that paren -- which is
#:                            where `comp_for`'s 1.414 nats came from.
#:   `target_first_component` the one exception: after the **first component** of a `for` target,
#:                            not after the whole target. `in`'s alternative appears *earlier*
#:                            than `in` does, which is why it needs its own rule. In
#:                            `for x, y in ...` the whole target is the tuple `x, y`, so a slot at
#:                            the literal has only `in` in it and no entropy -- which is what the
#:                            first implementation measured. The branch is after `x`, where `,` and
#:                            `in` are both legal, and the battery emits both: 540 `in` to 178 `,`
#:                            over the pilot's 720 prompts, a floor of **0.578 nats**, which is
#:                            E10's manually checked 0.554 reproduced by the code.
#:   `None`                   a forced role: one legal continuation, entropy zero by construction,
#:                            and no slot to measure.
SLOT_RULE = {
    "in": "target_first_component",
    "header_colon": "literal", "assign_eq": "literal", "comp_for": "literal",
    "ternary_if": "literal", "ternary_else": "literal",
    "close_paren": "literal", "close_bracket": "literal", "close_brace": "literal",
    "open_paren": None, "bare_colon": None,
}

#: The alternatives the grammar admits at each role's slot, as the *surface strings* that may
#: occupy it. Declared, because it is a fact about Python and not about any battery: the floor
#: is only estimable where every piece observed at the slot is one of these. Where the tokeniser
#: merges the literal with the end of the expression in front of it -- `):\n`, `]:\n`, `"]:\n`
#: for a header's colon -- the slot is not one syntactic position across prompts but several, and
#: pooling them measures the variety of the preceding expressions rather than the branch. That is
#: not a floor, and `floors.py` reports it as unavailable rather than writing the number.
SLOT_ALTERNATIVES = {
    "in": (" in", ","),
    "header_colon": (":",),
    # `assign_eq` covers `AugAssign` too, whose literal `=` arrives inside the augmented
    # operator: ` +=`, ` -=`, ` |=`. Those begin at the slot and carry nothing in front of them,
    # so they are the branch -- the generator's choice of operator -- and not bundling.
    "assign_eq": (" =", " +=", " -=", " *=", " /=", " //=", " %=", " **=", " &=", " |=",
                  " ^=", " >>=", " <<=", " @=", ",", ".", "[", ":"),
    "comp_for": (" for", "]", ")", "}", ","),
    "ternary_if": (" if",), "ternary_else": (" else",),
    "close_paren": (")", ","), "close_bracket": ("]", ","), "close_brace": ("}", ","),
}

#: the five token pieces read by more than one construct, piece-matched (amendment A10). A mark
#: on one of these carries `use`, naming which construct's use of the shared token it is, so a
#: reading can separate them instead of pooling the token's positions (item 10b).
TWO_USE_TOKENS = {
    "for": ("For", "AsyncFor", "ListComp", "SetComp", "GeneratorExp", "DictComp"),
    "with": ("With", "AsyncWith"),
    "def": ("FunctionDef", "AsyncFunctionDef"),
    "if": ("If", "IfExp"),
    "yield": ("Yield", "YieldFrom"),
}
#: construct -> the shared token it reads, inverted from the table above
TWO_USE_OF = {c: tok for tok, cs in TWO_USE_TOKENS.items() for c in cs}

#: literals that close a construct: the grammar forces them at the end of its span, not in a gap
CLOSERS = (")", "]", "}")


def _bindings(node) -> list[tuple[str, ast.AST]]:
    """Names the construct binds whose uses it *governs* -- binding and use both inside it."""
    out: list[tuple[str, ast.AST]] = []
    if isinstance(node, (ast.For, ast.AsyncFor)):
        out += [(n.id, node) for n in ast.walk(node.target) if isinstance(n, ast.Name)]
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                out += [(n.id, node) for n in ast.walk(item.optional_vars)
                        if isinstance(n, ast.Name)]
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        for arg in (list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
                    + [x for x in (a.vararg, a.kwarg) if x]):
            out.append((arg.arg, node))
    elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        for gen in node.generators:
            out += [(n.id, node) for n in ast.walk(gen.target) if isinstance(n, ast.Name)]
    elif isinstance(node, ast.ExceptHandler) and node.name:
        out.append((node.name, node))
    return out


def _bound_in_target(target) -> list[str]:
    """Names an assignment target actually *binds*, which is narrower than the names in it.

    `ast.walk` over the target was wrong, and wrong in a way that mattered once the generator
    started emitting these uses on purpose (item 95): `rows[0] = 3` and `head.field = 3` contain
    the name `rows` / `head`, but they **mutate** an existing object and bind nothing. A later
    use of `rows` is therefore not a use of a name this statement bound, and counting it as one
    would have put the scope class's loss at positions whose name the construct never introduced.

    What does bind: a plain `Name`, and the `Name`s inside a `Tuple`, `List` or `Starred` target,
    which is how `a, b = ...` and `*rest, = ...` work. The walk stops at a `Subscript` or
    `Attribute` rather than descending into it.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for e in target.elts for n in _bound_in_target(e)]
    if isinstance(target, ast.Starred):
        return _bound_in_target(target.value)
    return []


def _scope_bindings(node) -> list[str]:
    """Names the construct binds whose uses follow it in the enclosing block."""
    out: list[str] = []
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            out += _bound_in_target(t)
    elif isinstance(node, ast.NamedExpr):
        out += _bound_in_target(node.target)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for a in node.names:
            name = a.asname or a.name.split(".")[0]
            if name != "*":
                out.append(name)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.append(node.name)          # the def's own name, used by later calls
    return out


def _blocks(tree: ast.AST) -> dict[int, tuple[list, int]]:
    """For every statement, the block it sits in and its index there.

    A scope use is a use *after* the binding in its own block, nested contents included. That is
    the conservative reading: Python keeps the name bound after the enclosing block ends too,
    but a use further out is separated from the binding by the block boundary itself, and
    charging it to the construct would measure the reader's memory rather than the binding.
    """
    out = {}
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if isinstance(block, list):
                for i, stmt in enumerate(block):
                    out[id(stmt)] = (block, i)
    return out


def _enclosing_statement(node, tree):
    """The innermost statement containing `node`, or `node` itself if it is one.

    A scope use is a use that follows the binding *in its own block*, and a block holds
    statements -- so `_blocks` is keyed by statement. `NamedExpr` is an expression, so it was
    never in that map and `scope_positions` returned nothing for it no matter what the prompts
    contained: `total = (w := 5)` followed by `emit_all(w)` reported zero scope positions. That
    is why `NamedExpr` sat at exactly 0 in the pilot battery, and no change to the generator
    could have moved it (item 95).

    The statement that *contains* the walrus is the one whose block the use has to follow, so
    that is what this returns. Innermost by span, because an outer statement contains it too.
    """
    if isinstance(node, ast.stmt):
        return node
    best, best_width = None, None
    for parent in ast.walk(tree):
        if not isinstance(parent, ast.stmt):
            continue
        if not any(c is node for c in ast.walk(parent)):
            continue
        width = (getattr(parent, "end_lineno", parent.lineno) - parent.lineno,
                 getattr(parent, "end_col_offset", 0))
        if best_width is None or width < best_width:
            best, best_width = parent, width
    return best


def scope_positions(node, tree, line_starts: list[int]) -> list[Position]:
    """Uses of a construct-bound name that follow it in its own block."""
    names = set(_scope_bindings(node))
    if not names:
        return []
    blocks = _blocks(tree)
    stmt = _enclosing_statement(node, tree)
    where = blocks.get(id(stmt)) if stmt is not None else None
    if where is None:
        return []
    block, i = where
    out = []
    for stmt in block[i + 1:]:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name) and n.id in names and isinstance(n.ctx, ast.Load):
                lo, _ = node_span(n, line_starts)
                out.append(Position("dependent", "scope_use", lo, bound_name=n.id))
    return sorted(out, key=lambda p: p.char)


def _body_region(node) -> list[ast.AST]:
    """The part of the construct a header change would propagate into."""
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return [node.elt] + [c for g in node.generators for c in g.ifs]
    if isinstance(node, ast.DictComp):
        return [node.key, node.value] + [c for g in node.generators for c in g.ifs]
    if isinstance(node, ast.NamedExpr):
        return [node.value]
    body = getattr(node, "body", None)
    if body is None:
        return []
    # IfExp and Lambda have a single expression as their body, not a list of statements
    return list(body) if isinstance(body, list) else [body]


def dependent_positions(node, line_starts: list[int]) -> list[Position]:
    """Governed uses: names the construct binds, used inside the region it governs."""
    out = []
    bound = {name for name, _ in _bindings(node)}
    if not bound:
        return out
    for region in _body_region(node):
        for n in ast.walk(region):
            if isinstance(n, ast.Name) and n.id in bound and isinstance(n.ctx, ast.Load):
                lo, _ = node_span(n, line_starts)
                out.append(Position("dependent", "governed_use", lo, bound_name=n.id))
    return sorted(out, key=lambda p: p.char)


def _child_spans(node, line_starts) -> list[tuple[int, int]]:
    return [node_span(c, line_starts) for c in ast.iter_child_nodes(node)
            if hasattr(c, "lineno")]


def _target_first_component(node, line_starts) -> int | None:
    """The character just after the first component of a `for` target.

    `for x, y in ...` has a tuple target whose span covers `x, y`, so the gap after it holds
    only `in`. The grammar branches one component earlier -- after `x`, where `,` and `in` are
    both legal -- and that is the slot whose entropy is the floor.
    """
    tgt = getattr(node, "target", None)
    if tgt is None:
        gens = getattr(node, "generators", None)
        if not gens:
            return None
        tgt = gens[0].target
    first = tgt.elts[0] if isinstance(tgt, (ast.Tuple, ast.List)) and tgt.elts else tgt
    if not hasattr(first, "lineno"):
        return None
    return node_span(first, line_starts)[1]


def _last_element_end(node, line_starts) -> int | None:
    """The character just after the last element inside a display, call or subscript.

    Where the closer and `,` are both legal: `[a]` against `[a, b]`, `f(a)` against `f(a, b)`.
    An empty display has no element, and then the closer really is forced -- returning None
    leaves it without a slot, which is the honest answer rather than a floor of zero.
    """
    kids = [c for c in ast.iter_child_nodes(node) if hasattr(c, "lineno")]
    if not kids:
        return None
    return max(node_span(c, line_starts)[1] for c in kids)


def _preceding_end(node, line_starts, found: int, lo: int, fallback: int) -> int:
    """The character just after the last thing that precedes `found` inside this construct.

    `gap_start` -- the start of the gap between child spans -- was the slot for every expected
    role, and it is wrong wherever the thing in front of the literal has no `lineno`. The
    parameter list of a `def` is an `ast.arguments`, which does not, so `_child_spans` drops it
    and the gap runs from the construct's first character: `def f(a, b):` put the slot at `d`.
    Walking to the deepest spans instead finds the last parameter and puts the slot on the `)`,
    where the set also emits `,`. With nothing in front at all -- `class C:` -- the slot is the
    literal's own position, which is right: across the set that character holds `(` wherever the
    class has a base.
    """
    best = -1
    for c in ast.walk(node):
        if c is node or not hasattr(c, "lineno"):
            continue
        s0, s1 = node_span(c, line_starts)
        if lo <= s0 and s1 <= found:
            best = max(best, s1)
    return best if best >= 0 else fallback


def _slot_for(role: str, node, line_starts, found: int, lo: int,
              gap_start: int) -> int | None:
    """The slot character for one role, by SLOT_RULE. None for a forced role."""
    rule = SLOT_RULE.get(role, "gap_start")
    if rule is None:
        return None
    if rule == "target_first_component":
        return _target_first_component(node, line_starts)
    return found


def bare_colon_positions(node, text: str, line_starts: list[int]) -> list[Position]:
    """The `else:`, `elif`-free `else:` and `finally:` colons, which have no AST node.

    `ast` gives `orelse` and `finalbody` as statement lists, not as clauses, so the `else:` and
    `finally:` keywords are nowhere in the tree. They are forced positions -- nothing precedes
    the colon -- and leaving them out would drop the only forced colons most constructs have.
    Found by scanning backwards from the first statement of the clause to its keyword, so an
    `else` inside a nested construct is never picked up: the scan starts inside this node's own
    clause and stops at the keyword that opens it.
    """
    out = []
    for attr, kw in (("orelse", "else"), ("finalbody", "finally")):
        body = getattr(node, attr, None) or []
        # `IfExp` also has an `orelse`, but it is an *expression* and has no `else:` of its own
        if not body or not isinstance(body, list):
            continue
        if attr == "orelse" and isinstance(node, ast.If) and isinstance(body[0], ast.If):
            continue                                   # `elif`, which has no colon of its own
        first = node_span(body[0], line_starts)[0]
        seg = text[:first]
        k = seg.rfind(kw + ":")
        if k < 0:
            continue
        colon = k + len(kw)
        out.append(Position("obliged", "bare_colon", colon, forced=True, floor_nats=0.0))
    return out


def obliged_positions(node, text: str, line_starts: list[int]) -> list[Position]:
    """The literal tokens the grammar forces, searched for outside the children's spans.

    The header ends at the first body statement, so the closing `:` is the last one before it;
    a literal such as `in` sits between two child spans. Searching the *gaps* rather than the
    whole span is what keeps a `:` inside a dict display or a slice from being mistaken for the
    header's own.
    """
    kind = type(node).__name__
    literals = OBLIGED_LITERALS.get(kind)
    if not literals:
        return []
    lo, hi = node_span(node, line_starts)
    if isinstance(node, ast.IfExp):
        # `body if test else orelse`: the node does not begin with its header either, so the
        # literals are searched across the whole span between the three children's spans
        header_hi = hi
        spans = sorted(_child_spans(node, line_starts))
    elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        # a comprehension's element expression precedes its generators, so the header is not a
        # prefix of the node: search from the first generator's target to the end of the node
        gens = node.generators
        lo = min(node_span(g.target, line_starts)[0] for g in gens)
        lo = max(lo - 4, node_span(node, line_starts)[0])     # room for the `for ` itself
        header_hi = hi
        spans = sorted(node_span(g.target, line_starts) for g in gens)
    else:
        body = _body_region(node)
        header_hi = min([node_span(b, line_starts)[0] for b in body], default=hi)
        spans = sorted(s for s in _child_spans(node, line_starts) if s[0] < header_hi)
    gaps, cur = [], lo
    for s0, s1 in spans:
        if s0 > cur:
            gaps.append((cur, min(s0, header_hi)))
        cur = max(cur, s1)
    if cur < header_hi:
        gaps.append((cur, header_hi))
    out = []
    slot_of = {}
    for g0, g1 in gaps:
        slot_of.setdefault("gap", g0)
    for lit in literals:
        role = ROLE_OF.get(lit, lit)
        if lit in CLOSERS:
            # The closer is the last character of the node's span. It is **expected, not
            # forced** (item 10b): an expression precedes it, so `[a]` and `[a, b]` are both
            # legal and the slot after the last element carries real entropy. Before 96b the
            # closers were emitted with no slot and therefore with no floor at all.
            if hi - 1 >= lo and text[hi - 1] == lit:
                out.append(Position("obliged", role, hi - 1,
                                    forced=False, floor_nats=None,
                                    slot=_slot_for(role, node, line_starts, hi - 1, lo, hi - 1),
                                    use=kind if kind in TWO_USE_OF else None))
            continue
        found = -1
        for g0, g1 in gaps:
            seg = text[g0:g1]
            k = seg.rfind(lit) if lit == ":" else seg.find(lit)
            if k >= 0:
                cand = g0 + k
                # a word literal must stand alone, or `in` matches inside an identifier
                if lit.isalpha():
                    before = text[cand - 1] if cand else " "
                    after = text[cand + len(lit)] if cand + len(lit) < len(text) else " "
                    if before.isalnum() or before == "_" or after.isalnum() or after == "_":
                        continue
                found = cand
                if lit != ":":
                    break
        if found >= 0:
            gap_start = next((g0 for g0, g1 in gaps if g0 <= found < g1), found)
            if lit == ":" and _colon_is_bare(node, kind):
                # nothing precedes the colon -- `try:`, a bare `except:` -- so it is forced,
                # and it is a different role from a header's own `:` (item 96b)
                role = "bare_colon"
            slot_char = _slot_for(role, node, line_starts, found, lo, found)
            forced = role in FORCED_ROLES
            out.append(Position("obliged", role, found,
                                forced=forced,
                                floor_nats=0.0 if forced else None,
                                slot=slot_char,
                                use=kind if kind in TWO_USE_OF else None))
    out += bare_colon_positions(node, text, line_starts)
    return sorted(out, key=lambda p: p.char)


def find_node(tree: ast.AST, kind: str, tok_lo: int, tok_hi: int, text: str,
              line_starts: list[int]):
    """The node of `kind` that the construct's anchor *token* points at.

    The anchor token contains the construct's first character but need not start at it: an
    indented `for` tokenises with its leading whitespace, so the token begins one or more
    characters before the node does (`align.py` records that as `exact=False`). Requiring the
    node's span to contain the token's start therefore misses exactly the indented cases --
    measured at 129 of 200 `For` prompts before this was fixed. The test is overlap between
    the node's span and the token's, preferring the smallest node that starts nearest the
    token, which is the same tie-break `align.surface_offset` uses.
    """
    best, best_key = None, None
    for n in ast.walk(tree):
        if type(n).__name__ != kind:
            continue
        lo, hi = node_span(n, line_starts)
        if hi <= tok_lo or lo >= tok_hi:                 # no overlap
            continue
        key = (abs(lo - tok_lo), hi - lo)
        if best_key is None or key < best_key:
            best, best_key = n, key
    return best


def locate(positions: list[Position], pieces: list[str], construct_token: int
           ) -> list[Position]:
    """Character offsets to token indices, with the distance from the construct token."""
    spans = token_spans(pieces)
    for p in positions:
        p.token = token_at(p.char, spans)
        p.distance = p.token - construct_token
        if p.slot is not None:
            p.slot = token_at(p.slot, spans)
    return positions


def positions_for(text: str, construct: str, pieces: list[str],
                  construct_token: int) -> list[Position]:
    """Both classes for one prompt, as token indices. Empty if the prompt does not parse."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    ls = line_start_offsets(text)
    spans = token_spans(pieces)
    tok_lo, tok_hi = spans[construct_token]
    node = find_node(tree, construct, tok_lo, tok_hi, text, ls)
    if node is None:
        return None                      # the node could not be located: a different failure
    out = (obliged_positions(node, text, ls) + dependent_positions(node, ls)
           + scope_positions(node, tree, ls))
    out = locate(out, pieces, construct_token)
    # `token_spans` reconstructs spans from piece lengths and that accounting can fail to reach
    # the end of the text -- 8 of the pilot battery's 21,600 prompts, all with non-ASCII padding.
    # Every token index in such a prompt is suspect, so the marks are kept and flagged rather
    # than dropped: a reading that needs trustworthy indices filters on `matched` (E10).
    if not spans or spans[-1][1] != len(text):
        for p in out:
            p.matched = False
    return out
