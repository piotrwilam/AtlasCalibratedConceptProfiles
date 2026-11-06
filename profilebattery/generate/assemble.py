"""Assemble a core into a prompt: scope frame, padding, formatter, verification.

Order matters. The formatter runs over the *whole* prompt — core, frame and
padding together — so there is no style seam where generated code meets corpus
padding (handout S3). Because formatting can move lines, the construct's anchor is
derived from the formatted AST rather than tracked through it: the scope frame's
own `def`/`class` are named with a reserved prefix, which is what makes the core's
occurrence identifiable when the target type is FunctionDef or ClassDef.

Filters are validity-only. No model is run here (report S6).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np

from .compound import CoreText, _ind, realised_access
from .core import Core
from .menu import Cell, Menu

FRAME_DEF = "frame_run"
FRAME_CLASS = "FrameHost"
PAD_LINES_PER_LEVEL = 3

try:
    import black

    _MODE = black.Mode(line_length=88)

    def fmt(src: str) -> tuple[str, bool]:
        try:
            return black.format_str(src, mode=_MODE), True
        except Exception:
            return src, False
except ImportError:  # pragma: no cover - formatter is a hard requirement in release
    def fmt(src: str) -> tuple[str, bool]:
        return src, False


@dataclass
class Prompt:
    """One generated prompt with the metadata handout S3 requires per prompt."""

    text: str
    construct: str
    core_id: str
    family: str
    axes: dict[str, str]
    cell: Cell
    seed: int
    frame_class: str
    companions: tuple[str, ...]
    body_start: int = 1          # first line of the core region (after padding)
    anchor_lineno: int | None = None
    anchor_col: int | None = None
    span_end_lineno: int | None = None
    formatted: bool = True
    valid: bool = False
    reason: str = ""
    flags: dict = field(default_factory=dict)


#: constructs the scope frame itself emits. For these, "in class method" has to mean the
#: core *is* the method, not that the core sits inside one — otherwise a FunctionDef core at
#: method scope is a nested function and the construct's A set contains no methods at all,
#: while the corpus puts 74.8% of `def`s there. Statements are unaffected: an AnnAssign at
#: method scope is correctly inside a method already.
FRAME_EMITS = ("FunctionDef", "AsyncFunctionDef", "ClassDef")


def wrap_scope(core: CoreText, scope: str, menu: Menu, cell: Cell,
               rng: np.random.Generator, construct: str | None = None) -> list[str]:
    if scope == "module":
        return list(core.lines)
    if scope == "function":
        return [f"def {FRAME_DEF}({menu.name(cell.domain, 'plural', rng)}):"] + \
            _ind(core.lines)
    if scope == "method":
        if construct in FRAME_EMITS:
            # directly in the class body: the core's own def becomes the method
            return [f"class {FRAME_CLASS}:", ""] + _ind(core.lines)
        return [f"class {FRAME_CLASS}:", "", f"{'    '}def {FRAME_DEF}(self):"] + \
            _ind(core.lines, 2)
    raise KeyError(scope)


def _pad_and_format(lines: list[str], menu: Menu, cell: Cell,
                    rng: np.random.Generator) -> tuple[str, int, bool]:
    """Prepend padding, format the whole prompt, and locate the core's first line.

    The padding pool is corpus-typical code, so it contains constructs of its own — a
    `set()` call in the padding collides with a `Call` target, and for ubiquitous
    constructs "absent from the prompt" is unsatisfiable once the padding is real code.
    So both the anchor search and the B absence check are scoped to the *core region*,
    and its start is found by formatting the padding on its own and checking that the
    whole prompt still begins with it. If the formatter moves that boundary the prompt
    is rejected rather than mis-anchored.
    """
    if not cell.padding:
        src, ok = fmt("\n".join(lines) + "\n")
        return src, 1, ok
    pad = menu.padding(cell.domain, cell.padding * PAD_LINES_PER_LEVEL, rng)
    pad_src, pad_ok = fmt("\n".join(pad) + "\n")
    src, ok = fmt("\n".join(pad + [""] + lines) + "\n")
    if not (pad_ok and ok and src.startswith(pad_src.rstrip("\n"))):
        return src, -1, False
    return src, len(pad_src.rstrip("\n").splitlines()) + 1, ok


#: constructs whose dependent class is the *scope* class (C2): they bind a name that later
#: tokens in the block copy. Without a following statement that uses it the class is empty,
#: which is what the pilot battery measured -- zero, against a corpus median of one to two.
SCOPE_BINDERS = ("Assign", "AnnAssign", "AugAssign", "NamedExpr", "Import", "ImportFrom",
                 "ClassDef", "FunctionDef", "AsyncFunctionDef")


def _scope_binding_of(lines: list[str], construct: str) -> str | None:
    """The name this core's construct binds, for a post-garnish to read. None if unavailable.

    Parsed from the rendered core rather than threaded out of the template, because the
    templates compose names in a dozen places and a parse is one place. Any failure returns
    None and the garnish reads a fresh name, which is the behaviour before this existed.
    """
    if construct not in SCOPE_BINDERS:
        return None
    try:
        import ast

        from .positions import _scope_bindings
        tree = ast.parse("\n".join(lines))
    except Exception:
        return None
    for node in ast.walk(tree):
        if type(node).__name__ != construct:
            continue
        names = _scope_bindings(node)
        if names:
            return names[0]
    return None


def assemble(core: Core, menu: Menu, cell: Cell, seed: int,
             pre_probs: dict[str, float] | None = None,
             post_probs: dict[str, float] | None = None) -> Prompt:
    """Assemble and verify an A prompt.

    `pre_probs` / `post_probs` rotate companions at corpus rates — the handout's own
    requirement, which the stub menu never realised. They are the same garnishes the B
    side uses, fitted here against the *corpus* profile rather than against A: the corpus
    is what A is supposed to resemble, and B then follows the corrected A.
    """
    rng = np.random.default_rng(seed)
    from . import garnish as _garnish

    text = core.render(menu, cell, rng)
    lines = list(text.lines)
    anchor = text.anchor_line
    if post_probs:
        chosen = [g for g, pr in sorted(post_probs.items()) if rng.random() < pr]
        if chosen:
            # the post-garnish sits after the construct, which is where a scope use has to be:
            # the model is causal, so a use before the binding is not a use of it
            binding = _scope_binding_of(lines, core.construct)
            lines = lines + _garnish.render(chosen, menu, cell, rng, binding)
    if pre_probs:
        chosen = [g for g, pr in sorted(pre_probs.items()) if rng.random() < pr]
        if chosen:
            pre = _garnish.render(chosen, menu, cell, rng)
            lines, anchor = pre + lines, anchor + len(pre)
    text = CoreText(lines, anchor)
    lines = wrap_scope(text, cell.scope, menu, cell, rng, construct=core.construct)
    src, body_start, ok = _pad_and_format(lines, menu, cell, rng)
    p = Prompt(text=src, construct=core.construct, core_id=core.core_id,
               family=core.family, axes=_realised_axes(core.axes, cell), cell=cell,
               seed=seed, frame_class=core.frame_class, companions=core.companions,
               formatted=ok, body_start=body_start,
               flags={"target_kind": core.target_kind})
    verify_a(p)
    return p


def _realised_axes(axes: dict[str, str], cell) -> dict[str, str]:
    """The core's axes as the prompt actually realises them.

    A grid coordinate is a request; some levels cannot be honoured in every cell, and
    the metadata must say what was produced, not what was asked for. Today that is the
    access axis alone: `self_attribute` needs `self` bound, so outside a method it
    comes out as a plain attribute. Recording the nominal level there would misweight
    two thirds of the prompts at reweighting time.
    """
    out = dict(axes)
    if "access" in out:
        out["access"] = realised_access(out["access"], cell.scope)
    return out


def _is_node_construct(construct: str) -> bool:
    """Whether the construct names an `ast` class, as opposed to a builtin identifier.

    The A path already dispatched on `target_kind`, which `builtins_family` sets to `"name"`.
    The B path did not: `verify_b` called `_target_nodes` unconditionally and that begins with
    `getattr(ast, construct)`, so the first builtin of the phase build raised
    `AttributeError: module 'ast' has no attribute 'ArithmeticError'`. For a builtin the check B
    needs is the same in substance and different in mechanism -- no *reference* to the name in
    the core region, rather than no node of the type.
    """
    return hasattr(ast, construct)


def _target_nodes(tree: ast.AST, construct: str,
                  from_line: int = 1) -> list[tuple[int, int, int, ast.AST]]:
    """Candidate anchor nodes as (depth, lineno, col, node), outermost first.

    Uniqueness is the wrong rule for expressions: a chained attribute
    (`a.b.c`), a nested call (`f(g(x))`) and a nested display (`[[a], [b]]`) are each
    genuinely several nodes of the target type, and the construct the core placed is
    the *outermost* one. So the anchor is the shallowest candidate, earliest in source
    order on ties — after the scope frame's own def/class are excluded by name.
    """
    cls = getattr(ast, construct)
    out = []
    stack = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
            if not isinstance(child, cls):
                continue
            if construct in ("FunctionDef", "AsyncFunctionDef", "ClassDef") and \
                    getattr(child, "name", None) in (FRAME_DEF, FRAME_CLASS):
                continue
            if getattr(child, "lineno", 0) < from_line:
                continue          # in the padding, not in the core
            out.append((depth + 1, getattr(child, "lineno", 0),
                        getattr(child, "col_offset", 0), child))
    return sorted(out, key=lambda t: (t[0], t[1], t[2]))


def _name_nodes(tree: ast.AST, ident: str,
                from_line: int = 1) -> list[tuple[int, int, int, ast.AST]]:
    """Name/Attribute-free references to `ident` in the core region, outermost first."""
    out = []
    stack = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
            if isinstance(child, ast.Name) and child.id == ident \
                    and getattr(child, "lineno", 0) >= from_line:
                out.append((depth + 1, child.lineno, child.col_offset, child))
    return sorted(out, key=lambda t: (t[1], t[2]))


def verify_a(p: Prompt) -> Prompt:
    """Validity filter for an A prompt: parses, and the construct is anchorable."""
    if not p.formatted or p.body_start < 1:
        p.valid, p.reason = False, "formatter failed or padding boundary moved"
        return p
    try:
        tree = ast.parse(p.text)
    except SyntaxError as e:
        p.valid, p.reason = False, f"parse: {e.msg} line {e.lineno}"
        return p
    if p.flags.get("target_kind") == "constant":
        want = {"True": True, "False": False, "None": None}[p.construct]
        hits = [nd for nd in ast.walk(tree)
                if isinstance(nd, ast.Constant) and nd.value is want
                and getattr(nd, "lineno", 0) >= p.body_start]
        if not hits:
            p.valid, p.reason = False, f"no {p.construct} constant"
            return p
        hits.sort(key=lambda nd: (nd.lineno, nd.col_offset))
        p.anchor_lineno, p.anchor_col = hits[0].lineno, hits[0].col_offset
        p.span_end_lineno = hits[0].lineno
        p.flags["nested_target_count"] = len(hits)
        p.valid, p.reason = True, ""
        return p
    if p.flags.get("target_kind") == "name":
        nodes = _name_nodes(tree, p.construct, p.body_start)
        if not nodes:
            p.valid, p.reason = False, f"no reference to {p.construct}"
            return p
        _, ln, col, _ = nodes[0]
        p.flags["nested_target_count"] = len(nodes)
        p.anchor_lineno, p.anchor_col, p.span_end_lineno = ln, col, ln
        p.valid, p.reason = True, ""
        return p
    nodes = _target_nodes(tree, p.construct, p.body_start)
    if not nodes:
        p.valid, p.reason = False, f"no anchorable {p.construct} node"
        return p
    depth, _, _, node = nodes[0]
    if len(nodes) > 1 and nodes[1][0] == depth:
        p.valid, p.reason = False, (f"{p.construct} anchor ambiguous: two nodes at "
                                    f"depth {depth}")
        return p
    p.flags["nested_target_count"] = len(nodes)
    # the construct's own surface token, not the node's start: an Assign node begins at its
    # target and a Call node at its function, so the node offset would anchor on the
    # identifier before the `=` or the `(`
    from ..inventory.tokens import tokens_of
    from .align import surface_offset, surface_tokens

    p.anchor_lineno, p.anchor_col = node.lineno, node.col_offset
    candidates = surface_tokens(node, p.construct, tokens_of(p.construct))
    off, found = -1, False
    for token in candidates:
        off, found = surface_offset(node, p.text, token)
        if found:
            p.flags["surface_token"] = token
            break
    if candidates:
        p.flags["surface_token_found"] = found
        if found:
            # `\n`-only line counting, to match the AST's lineno: str.splitlines also
            # breaks on form feed and friends, which Python source treats as whitespace
            from .align import line_start_offsets

            offs = line_start_offsets(p.text)
            ln = max(i for i, o in enumerate(offs, start=1) if o <= off)
            p.anchor_lineno, p.anchor_col = ln, off - offs[ln - 1]
    p.span_end_lineno = getattr(node, "end_lineno", None)
    p.valid, p.reason = True, ""
    return p


def verify_b(p: Prompt, token: str, tokenizer=None) -> Prompt:
    """Validity filter for a B prompt: parses, construct absent, token present.

    `tokenizer` is the model's tokeniser. The handout requires the token to be
    present *in the tokeniser output*, not merely in the text — a stricter test
    that the identifier subtype of a keyword construct will often fail, because
    BPE/SentencePiece merges the keyword into a longer piece. Failures are a
    filter outcome, not an error.
    """
    if not p.formatted:
        p.valid, p.reason = False, "formatter failed"
        return p
    try:
        tree = ast.parse(p.text)
    except SyntaxError as e:
        p.valid, p.reason = False, f"parse: {e.msg} line {e.lineno}"
        return p
    if _is_node_construct(p.construct):
        nodes = _target_nodes(tree, p.construct, p.body_start)
        if nodes:
            p.valid, p.reason = False, f"{len(nodes)} {p.construct} nodes present"
            return p
    else:
        refs = _name_nodes(tree, p.construct, p.body_start)
        if refs:
            p.valid, p.reason = False, f"{len(refs)} references to {p.construct} present"
            return p
    if token not in p.text:
        p.valid, p.reason = False, "token absent from text"
        return p
    # B's three sites, per the handout: the token, the token's span end (the token
    # itself, since a B token is a single lexeme) and the last token. Without these the
    # per-site companion audit has no anchor to scope against on the B side.
    # Anchor on an occurrence whose own *token piece* is the token, not on the first
    # substring match. `str.find` matched `for` inside `form`, `force` and `endif`, and
    # three to five prompts per B set were anchored on a token that merely contains the
    # letters -- a silent corruption, since the profile is then a profile of `force`. The
    # piece test is also the one the handout asks for ("present in the tokeniser output"),
    # applied to the *anchor* rather than to the prompt as a whole: the old code checked
    # the two independently, so a prompt could carry the token as a piece in one place and
    # be anchored on a substring in another.
    p.flags["surface_token"] = token
    cols = _piece_exact_positions(p.text, token, tokenizer)
    if cols is None:                            # no tokeniser: the old substring behaviour
        p.flags["surface_token_found"] = None
        cols = [(i, lines.find(token)) for i, lines in enumerate(p.text.splitlines())
                if lines.find(token) >= 0]
    elif not cols:
        p.flags["surface_token_found"] = False
        p.valid, p.reason = False, "token present only inside a longer piece"
        return p
    else:
        p.flags["surface_token_found"] = True
    lines_ = p.text.splitlines()
    for i in range(p.body_start - 1, len(lines_)):
        col = next((c for ln, c in cols if ln == i), -1)
        if col >= 0:
            p.anchor_lineno, p.anchor_col = i + 1, col
            # The span end is the end of the *statement* carrying the token, not the
            # token itself. The handout's "the token itself for single-token B" makes
            # the site non-comparable: A's span end sits after the construct's whole
            # body, so against a one-token B span the two sites read structurally
            # different places in the prompt, and the companion counts cannot be
            # matched even in principle. The enclosing statement is the analogue.
            p.span_end_lineno = i + 1
            for nd in ast.walk(tree):
                if not isinstance(nd, ast.stmt):
                    continue
                if nd.lineno <= i + 1 <= (nd.end_lineno or nd.lineno):
                    p.span_end_lineno = max(p.span_end_lineno,
                                            nd.end_lineno or nd.lineno)
            break
    if p.anchor_lineno is None:
        p.valid, p.reason = False, "token not locatable in the core region"
        return p
    if tokenizer is not None:
        ids = tokenizer(p.text, add_special_tokens=False)["input_ids"]
        # **Both tokenisations, and the second is the one that matters now.** `tokenizer(token)`
        # gives the *bare* id sequence, and this check silently assumed that is what appears in
        # the text. Once the B shapes were filtered to place the token after a space (A19a), the
        # text carries the **space-prefixed** sequence instead -- `▁Ar`+`ith`+`metic`+`Error`
        # rather than `Ar`+`ith`+`metic`+`Error` -- and the check rejected 300 of 320 prompts
        # for `ArithmeticError` as "absent" while passing `AttributeError`, whose space-prefixed
        # form happens to be a single id that decodes to the name. **143 of 376 B sets came out
        # short and about seventy were empty**, which is no contrast at all: the thing A17 fixed.
        bare = tokenizer(token, add_special_tokens=False)["input_ids"]
        spaced = tokenizer(" " + token, add_special_tokens=False)["input_ids"]
        pieces = [tokenizer.decode([i]).strip() for i in ids]
        p.flags["token_in_tokeniser"] = (token in pieces) or _sublist(ids, bare) \
            or _sublist(ids, spaced)
        if not p.flags["token_in_tokeniser"]:
            p.valid, p.reason = False, "token absent from tokeniser output"
            return p
    p.valid, p.reason = True, ""
    return p


def _piece_exact_positions(text: str, token: str, tokenizer):
    """(line, col) of every occurrence of `token` that is a whole token piece.

    None when no tokeniser is available, which keeps the substring path for tests that run
    without one. The space marker is stripped before comparison: SentencePiece writes an
    indented or mid-line keyword as `\u2581for` and a line-initial one as `for`, and both are
    the token.
    """
    if tokenizer is None:
        return None
    if not token.strip().isidentifier():
        # A *symbol* token is merged with its neighbours by BPE -- `[` arrives as `("[`, `:`
        # as `":` or `):\n` -- so "the piece is the token" is almost never true for one, and
        # requiring it would re-anchor 2,191 B prompts away from the piece that actually
        # contains the construct's character (measured on the pilot battery). That containing
        # piece is `align.py`'s declared policy and the right one here: the exactness flag
        # records the boundary, and no alignment assertion is made. So the piece-exact rule
        # applies to keyword tokens only, where it is both correct and what the handout asks.
        return None
    from .align import line_start_offsets, token_spans
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    pieces = [tokenizer.convert_ids_to_tokens(i) for i in ids]
    spans = token_spans(pieces)
    starts = line_start_offsets(text)
    out = []
    # A *sequence* of pieces, not a single one. The single-piece test was right for every
    # keyword -- `for`, `with`, `def` are each one piece -- and it cannot ever succeed for a
    # name the tokeniser splits: `ArithmeticError` arrives as `Ar`+`ith`+`metic`+`Error`, so no
    # piece equals the token and the prompt was rejected as "present only inside a longer
    # piece". **82 of the 117 builtin names measured are multi-piece, and 66 of them ended with
    # an empty B set** -- no contrast at all, so no concept profile. For a one-piece token this
    # loop matches on the first step and the behaviour is unchanged.
    for i in range(len(pieces)):
        acc = ""
        for j in range(i, len(pieces)):
            acc += pieces[j].replace("\u2581", " ")
            if len(acc.strip()) > len(token):
                break
            if acc.strip() != token:
                continue
            # The occurrence is located from the token itself, not from `spans[j][1]`: the
            # reconstructed span end can disagree with the text by a character wherever a
            # leading marker stands for a space that is not there, and the boundary test must
            # not depend on that. Scan to the token's first character, then verify the literal.
            lo = spans[i][0]
            while lo < len(text) and text[lo] != token[0]:
                lo += 1
            hi = lo + len(token)
            if text[lo:hi] != token:
                break
            # the run must be the whole identifier, not part of a longer one: `ArithmeticError`
            # inside `MyArithmeticErrorKind` is the same corruption the single-piece rule was
            # introduced to stop, and the sequence form has to keep stopping it
            before = text[lo - 1] if lo > 0 else " "
            after = text[hi] if hi < len(text) else " "
            if before.isalnum() or before == "_" or after.isalnum() or after == "_":
                break
            line = max(k for k, st in enumerate(starts) if st <= lo)
            out.append((line, lo - starts[line]))
            break
    return out


def _sublist(hay: list[int], needle: list[int]) -> bool:
    if not needle:
        return False
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def assemble_b(bcore, token: str, menu: Menu, cell: Cell, seed: int,
               tokenizer=None, pre_probs: dict[str, float] | None = None,
               post_probs: dict[str, float] | None = None) -> Prompt:
    """Assemble and verify a B prompt from a shared-pool shape.

    Garnishes fit the B set's companion marginals to the A set's (matching.py), and
    *where* they sit is part of the fit. The model is causal, so a garnish placed after
    the carrier is invisible at the construct-token site and visible only at the
    last-token site. `pre_probs` are placed before the carrier and count at every site;
    `post_probs` after it and count only at the last-token site — which is what lets one
    B set match A at all three sites, since A's own companions are mostly in its body
    and therefore behind its anchor.
    """
    rng = np.random.default_rng(seed)
    from . import garnish as _garnish

    pre: list[str] = []
    if pre_probs:
        chosen = [g for g, pr in sorted(pre_probs.items()) if rng.random() < pr]
        if chosen:
            pre = _garnish.render(chosen, menu, cell, rng)
    lines = bcore.render(menu, cell, rng)
    if post_probs:
        chosen = [g for g, pr in sorted(post_probs.items()) if rng.random() < pr]
        if chosen:
            lines = lines + _garnish.render(chosen, menu, cell, rng)
    lines = pre + lines
    lines = wrap_scope(CoreText(lines, 0), cell.scope, menu, cell, rng)
    src, body_start, ok = _pad_and_format(lines, menu, cell, rng)
    p = Prompt(text=src, construct=bcore.construct, core_id=bcore.core_id,
               family="bpool", axes={"subtype": bcore.subtype,
                                     "shape": str(bcore.shape_index)},
               cell=cell, seed=seed, frame_class="none", companions=(),
               formatted=ok, body_start=body_start)
    return verify_b(p, token, tokenizer)
