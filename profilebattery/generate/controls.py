r"""The matched control set's per-construct templates (C2's role gap, G4, Cowork's step 5).

C2 reads the dependent class against a control that tells the construct's role from copying: a
body token that copies a name the header bound is predictable without understanding the
construct at all. The B sets cannot supply it -- a B prompt carries the construct's *token* in
another role and shares no structure with its A prompt, so no position in B corresponds to a
position in A. The control is therefore a variant of each A prompt in which **only the binding
changes**.

Two variants, answering different questions.

  **(i) same name, construct removed.** The header becomes a plain assignment binding the same
  name; the body is unchanged. The name stays copyable and the construct is gone, so the gap at
  the body's use is *the construct's contribution over copying* -- the role gap.

  **(ii) different name, construct kept.** The header binds a different name; the body is
  unchanged, so its use is now a free variable and copying is impossible. The gap is *the copy
  contribution itself*.

Which variant a construct can take is a fact about its grammar, not a preference, and this
table records it construct by construct before any transform is written.

Three things make the table simpler than it first looks.

*The control is loss-only.* Nothing is read from its activations -- no profile, no accumulator,
no yardstick -- so it sits in no fit. It need only parse, keep everything outside the header
identical, and bind the name the row says. The companion cap, the read-site overlap and the
C-set licence do not apply to it, which is why a control header may use a call or a subscript
that would cost a re-fit inside A.

*The comprehensions need no control at all.* Their governed use *precedes* the binding in source
order -- measured at a median distance of -5 tokens against +17 for `For` (`COUNTS.json`) -- so
the model must predict the use before it has seen the name. Copying from the header is
impossible by construction, and there is nothing for a control to remove. This is a better
reason than the one first given, that variant (i) is not constructible for them; the
impossibility is the point, and it removes four constructs from the set.

*The assignment and import family takes (ii) only.* Variant (i) replaces a header with a plain
assignment binding the same name, and these constructs already *are* that assignment, so (i) is
a no-op. Binding a different name and leaving the later use alone is the only control that says
anything, and the free variable it creates is exactly the copying-impossible condition.

The obliged class needs no control: those positions are forced by the grammar, and there is no
copying rival to separate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlTemplate:
    construct: str
    variant: str               # "i" | "ii"
    serves: str                # "governed" | "scope"
    a_header: str              # the A prompt's header, schematically
    control_header: str        # what replaces it
    note: str = ""


#: One row per construct whose dependent class needs a copying control. A construct absent from
#: this table either has no dependent class or needs no control, and `reason_not_needed` says
#: which -- absence is never silent.
TEMPLATES: tuple[ControlTemplate, ...] = (
    # ---- governed class, variant (i): the construct is removed, the name stays ----
    ControlTemplate(
        "For", "i", "governed",
        "for {x} in {it}:", "{x} = {first_of_it}",
        "a single target subscripts or nexts the iterable: `x = xs[0]` where that is valid, "
        "`x, y = next(iter(items))` for a tuple target or a non-subscriptable iterable. The "
        "`range` operand binds nothing and has no governed class, so it takes no control."),
    ControlTemplate(
        "With", "i", "governed",
        "with {ctx} as {f}:", "{f} = {ctx}",
        "the cleanest case: the context manager expression is already the bound value's "
        "source, so removing the `with` leaves a plain assignment of it."),
    ControlTemplate(
        "FunctionDef", "ii", "governed",
        "def {name}({param}):", "def {name}({other_param}):",
        "variant (ii), not (i), and the implementation is what settled it: dedenting a "
        "function body that contains `return` is a **syntax error**, not merely implausible "
        "code, and one of the eight body templates returns. Renaming the parameter leaves the "
        "body's use a free variable, which is the same copying-impossible condition at no "
        "syntactic risk and with no dedent."),
    # ---- scope class, variant (ii): the construct stays, the name changes ----
    ControlTemplate(
        "Assign", "ii", "scope",
        "{name} = {value}", "{other_name} = {value}",
        "variant (i) is a no-op here: the construct already *is* a plain assignment. Renaming "
        "the target leaves the later use a free variable, which is the copying-impossible "
        "condition the gap needs."),
    ControlTemplate(
        "AugAssign", "ii", "scope",
        "{name} += {value}", "{other_name} += {value}",
        "as `Assign`. The augmented target must be bound earlier in any case, so the renamed "
        "one is bound too and the statement stays plausible."),
    ControlTemplate(
        "ImportFrom", "ii", "scope",
        "from {mod} import {name}", "from {mod} import {other_name}",
        "the later use becomes a free variable. A module attribute that exists is preferred "
        "over an invented one, so the import itself stays plausible."),
    ControlTemplate(
        "Import", "ii", "scope",
        "import {mod} as {name}", "import {mod} as {other_name}",
        "as `ImportFrom`."),
    # ---- the async forms: their sync counterparts' templates, stated rather than assumed ----
    ControlTemplate(
        "AsyncFor", "i", "governed",
        "async for {x} in {it}:", "{x} = {first_of_it}",
        "as `For`. Variant (i) removes the whole statement, so the control loses the `async` "
        "keyword with it; that is correct -- the construct being removed is the point, and the "
        "control is loss-only, so it need only parse."),
    ControlTemplate(
        "AsyncWith", "i", "governed",
        "async with {ctx} as {f}:", "{f} = {ctx}",
        "as `With`."),
    ControlTemplate(
        "AsyncFunctionDef", "ii", "governed",
        "async def {name}({param}):", "async def {name}({other_param}):",
        "as `FunctionDef`."),
    ControlTemplate(
        "NamedExpr", "ii", "scope",
        "({name} := {value})", "({other_name} := {value})",
        "as `Assign`; the walrus binds in the enclosing scope, so the later use is free."),
)

#: Constructs with a dependent class that need no control, and why. Recorded so that a missing
#: row is a decision rather than an omission.
NOT_NEEDED: dict[str, str] = {
    "ListComp": "the governed use precedes the binding in source order (median distance -5 "
                "tokens), so copying from the header is impossible and there is nothing for a "
                "control to remove",
    "SetComp": "as ListComp",
    "GeneratorExp": "as ListComp",
    "DictComp": "as ListComp",
    "AnnAssign": "its corpus median of scope uses is 0, so the class is not read for it",
    "ClassDef": "its scope use is not generated -- the suite statement sits inside the class "
                "body, where the class name is not yet bound",
    "ExceptHandler": "not in the phase battery's construct list as a separate set",
}

BY_CONSTRUCT: dict[str, ControlTemplate] = {t.construct: t for t in TEMPLATES}


def needs_control(construct: str) -> bool:
    return construct in BY_CONSTRUCT


def template_for(construct: str) -> ControlTemplate | None:
    return BY_CONSTRUCT.get(construct)


def reason_not_needed(construct: str) -> str | None:
    return NOT_NEEDED.get(construct)


def audit() -> dict:
    """Every construct with a dependent class is either templated or excused, never neither.

    Called by the test of the same name and by the smoke build's report, so that adding a
    construct to the battery without deciding its control fails loudly.
    """
    from .positions import GOVERNED_BINDERS, SCOPE_BINDERS

    have_class = sorted(GOVERNED_BINDERS | SCOPE_BINDERS)
    undecided = [c for c in have_class
                 if c not in BY_CONSTRUCT and c not in NOT_NEEDED]
    return {"with_dependent_class": have_class,
            "templated": sorted(BY_CONSTRUCT),
            "excused": sorted(NOT_NEEDED),
            "undecided": undecided,
            "n_control_sets": len(BY_CONSTRUCT)}


# ---------------------------------------------------------------------------
# the transform (G4, item 85)
# ---------------------------------------------------------------------------

import ast  # noqa: E402  (kept beside the transform it serves)
from dataclasses import replace  # noqa: E402,F401

#: statements that are legal only inside the construct that encloses them. Variant (i) removes
#: the construct, so a body containing one of these cannot be dedented -- `continue` outside a
#: loop and `return` outside a function are syntax errors, not implausible code. Two of the
#: eight body templates emit `continue` or `break`, and one emits `return`, so this is the
#: common case rather than an edge: the builder falls back to variant (ii) for them.
JUMPS = (ast.Return, ast.Break, ast.Continue, ast.Yield, ast.YieldFrom, ast.Await)

#: substitutes for variant (ii). They share no prefix with a generated name, which matters: a
#: substitute like `total_alt` for `total` tokenises with the original's pieces, so copying
#: would still half-work and the control would understate the copy contribution.
SUBSTITUTES = ("parcel", "tally", "facet", "hinge", "ridge", "vertex", "margin", "strand",
               "lattice", "quorum", "plinth", "sable")

IND = "    "


@dataclass(frozen=True)
class Control:
    """A control prompt, with the arithmetic needed to find A's positions inside it.

    The pair cannot be matched by re-running the position finder on the control, and the reason
    is worth stating because it is not obvious. Under variant (ii) the header binds a
    *different* name, so the body's use is no longer a use of what the construct binds -- the
    AST rule correctly reports **no** governed position there, which is the whole point of the
    variant. Under variant (i) the construct is gone, so its governed uses become scope uses of
    a plain assignment. Either way the finder gives a different answer on the control than on
    A, by design.

    So a position is tracked through the *edit* instead. Both variants are explicit text
    transforms with known shape -- one replaced span, or a replaced header plus a dedented
    body -- and `map_offset` carries an A character offset to its counterpart exactly. An
    obliged position maps to None: it lives in the header, which variant (i) removes, and it
    needs no control anyway, being forced by the grammar.
    """

    text: str
    variant: str               # the variant actually used, which may not be the table's
    bound_name: str            # the name whose use the gap is read at
    substitute: str | None     # variant (ii) only: what the binding was renamed to
    fell_back: bool = False    # True when the table said (i) and the body forbade it
    reason: str = ""
    a_text: str = ""
    #: variant (ii): (lo, hi, replacement length) of the single replaced span
    span: tuple[int, int, int] | None = None
    #: variant (i): (first header line, last header line, indent removed, last body line)
    dedent: tuple[int, int, int, int] | None = None

    def map_offset(self, offset: int) -> int | None:
        """An A character offset carried into the control, or None if it has no counterpart."""
        from .align import line_start_offsets
        if self.span is not None:
            lo, hi, new_len = self.span
            if offset < lo:
                return offset
            if offset < hi:
                return None                      # inside the renamed binding itself
            return offset + (new_len - (hi - lo))
        if self.dedent is None:
            return None
        h0, h1, ind, body_end = self.dedent
        a_starts = line_start_offsets(self.a_text)
        c_starts = line_start_offsets(self.text)
        line = max(i for i, st in enumerate(a_starts) if st <= offset)
        col = offset - a_starts[line]
        if line < h0:
            return offset
        if line <= h1:
            return None                          # in the header the variant removed
        if line > body_end:
            new_line = line - (h1 - h0)
            return c_starts[new_line] + col if new_line < len(c_starts) else None
        new_line = line - (h1 - h0)
        if new_line >= len(c_starts):
            return None
        return c_starts[new_line] + max(col - ind, 0)


def _taken_names(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
           {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}


def _fresh_name(taken: set[str], key: str) -> str:
    """A substitute sharing no prefix with the original, chosen deterministically."""
    start = sum(ord(c) for c in key) % len(SUBSTITUTES)
    for i in range(len(SUBSTITUTES)):
        cand = SUBSTITUTES[(start + i) % len(SUBSTITUTES)]
        if cand not in taken and not any(t.startswith(cand) for t in taken):
            return cand
    return SUBSTITUTES[start] + "_x"


def _body_forbids_dedent(node) -> str | None:
    """Why variant (i) is unavailable for this node, or None if it is available."""
    for stmt in getattr(node, "body", []) or []:
        for n in ast.walk(stmt):
            if isinstance(n, JUMPS):
                return f"the body contains {type(n).__name__}, illegal outside the construct"
    return None


def _lines_of(text: str) -> list[str]:
    return text.split("\n")


def _dedent(lines: list[str]) -> list[str]:
    return [ln[len(IND):] if ln.startswith(IND) else ln for ln in lines]


def _header_span(node, lines: list[str]) -> tuple[int, int]:
    """(first, last) 0-indexed line numbers of the construct's header."""
    first = node.lineno - 1
    body = getattr(node, "body", None)
    last = (body[0].lineno - 2) if body else (getattr(node, "end_lineno", node.lineno) - 1)
    return first, max(last, first)


def _replace_span(text: str, node, new_src: str) -> str:
    """Replace one node's exact character span, leaving everything else byte-identical."""
    from .align import line_start_offsets
    starts = line_start_offsets(text)
    lo = starts[node.lineno - 1] + node.col_offset
    hi = starts[node.end_lineno - 1] + node.end_col_offset
    return text[:lo] + new_src + text[hi:]


def _variant_i(text: str, node, construct: str) -> Control | None:
    """Remove the construct, keep the name: the header becomes a plain assignment."""
    lines = _lines_of(text)
    h0, h1 = _header_span(node, lines)
    indent = lines[h0][: len(lines[h0]) - len(lines[h0].lstrip())]
    if isinstance(node, (ast.For, ast.AsyncFor)):
        target = ast.unparse(node.target)
        it = ast.unparse(node.iter)
        # next(iter(...)) rather than [0]: it is valid for every iterable, including the
        # `.items()`, `zip` and `enumerate` operands, so one form covers the whole axis
        header = [f"{indent}{target} = next(iter({it}))"]
        bound = target.split(",")[0].strip()
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        item = node.items[0]
        if item.optional_vars is None:
            return None
        bound = ast.unparse(item.optional_vars)
        header = [f"{indent}{bound} = {ast.unparse(item.context_expr)}"]
    else:
        return None
    body_end = node.body[-1].end_lineno - 1
    body = _dedent(lines[h1 + 1: body_end + 1])
    out = lines[:h0] + header + body + lines[body_end + 1:]
    src = "\n".join(out)
    try:
        ast.parse(src)
    except SyntaxError:
        return None
    return Control(src, "i", bound, None, False,
                   "the construct is removed and its name bound by a plain assignment",
                   a_text=text, dedent=(h0, h1, len(IND), body_end))


def _variant_ii(text: str, node, construct: str, tree: ast.AST) -> Control | None:
    """Keep the construct, rename what it binds: the later use becomes a free variable."""
    from .positions import _bindings, _scope_bindings
    taken = _taken_names(tree)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        # rename through `as`, not by inventing a module attribute: `from os import path as
        # parcel` keeps the import valid while leaving the later `path.join` a free variable,
        # which is what the gap needs. It adds two tokens to the header, which is harmless --
        # pairs are matched by role and bound name, never by token index.
        alias = node.names[0]
        if alias.name == "*":
            return None
        bound = alias.asname or alias.name.split(".")[0]
        sub = _fresh_name(taken, bound)
        from .align import line_start_offsets
        starts = line_start_offsets(text)
        lo = starts[alias.lineno - 1] + alias.col_offset
        hi = starts[alias.end_lineno - 1] + alias.end_col_offset
        src = text[:lo] + f"{alias.name} as {sub}" + text[hi:]
        try:
            ast.parse(src)
        except SyntaxError:
            return None
        return Control(src, "ii", bound, sub, False,
                       "the import is renamed through `as`, so the module attribute stays "
                       "real and the later use is free",
                       a_text=text, span=(lo, hi, len(f"{alias.name} as {sub}")))
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = (list(node.args.posonlyargs) + list(node.args.args)
                + list(node.args.kwonlyargs))
        if not args:
            return None
        arg = args[0]
        sub = _fresh_name(taken, arg.arg)
        starts_ok = True
        from .align import line_start_offsets
        starts = line_start_offsets(text)
        lo = starts[arg.lineno - 1] + arg.col_offset
        src = text[:lo] + sub + text[lo + len(arg.arg):]
        bound = arg.arg
        span = (lo, lo + len(arg.arg), len(sub))
    else:
        names = [n for n, _ in _bindings(node)] or _scope_bindings(node)
        if not names:
            return None
        bound = names[0]
        target = None
        for cand in ast.walk(node):
            if isinstance(cand, ast.Name) and cand.id == bound and \
                    isinstance(cand.ctx, (ast.Store, ast.Load)):
                target = cand
                break
        if target is None:
            return None
        sub = _fresh_name(taken, bound)
        src = _replace_span(text, target, sub)
        from .align import line_start_offsets as _ls
        _st = _ls(text)
        _lo = _st[target.lineno - 1] + target.col_offset
        _hi = _st[target.end_lineno - 1] + target.end_col_offset
        span = (_lo, _hi, len(sub))
        starts_ok = True
    if not starts_ok:
        return None
    try:
        ast.parse(src)
    except SyntaxError:
        return None
    return Control(src, "ii", bound, sub, False,
                   "the binding is renamed, so the later use cannot be copied from it",
                   a_text=text, span=span)


def build_control(text: str, construct: str, anchor_lo: int,
                  anchor_hi: int | None = None) -> Control | None:
    """The control prompt for one A prompt, or None if the construct needs or allows none.

    Deterministic in its inputs, so a control is reproducible from the A prompt alone -- no
    menu, no generator state and no random draw enters it, which is what lets the pair be
    rebuilt from the battery rather than stored twice.

    `anchor_lo`/`anchor_hi` are the **span of the anchor token**, not a point. The distinction
    cost 15 of 24 `For` controls on the first smoke build: an indented `for` tokenises as
    `\u2581for`, whose span begins at the *space* and therefore before the construct's node,
    so a one-character window at the token's start overlaps nothing and the node is not found.
    The tests passed throughout because they pass an offset at the construct's first character,
    which is inside the node -- the failure needed real prompts to appear.
    """
    if not needs_control(construct):
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    from .positions import find_node
    from .align import line_start_offsets  # noqa: F401
    node = find_node(tree, construct, anchor_lo,
                     anchor_hi if anchor_hi is not None else anchor_lo + 1, text,
                     line_start_offsets(text))
    if node is None:
        return None
    want = BY_CONSTRUCT[construct].variant
    if want == "i":
        why = _body_forbids_dedent(node)
        if why is None:
            got = _variant_i(text, node, construct)
            if got is not None:
                return got
            why = "the dedented body did not parse"
        got = _variant_ii(text, node, construct, tree)
        if got is None:
            return None
        return Control(got.text, "ii", got.bound_name, got.substitute, True, why,
                       a_text=got.a_text, span=got.span)
    return _variant_ii(text, node, construct, tree)
