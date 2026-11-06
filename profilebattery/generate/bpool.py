"""The shared B pool: token-parameterised core shapes, reused across constructs.

A B prompt has the construct's surface token present and the construct absent. The
*shape* of that prompt barely depends on which construct is being measured — "the
token inside a string literal" is the same shape whether the token is `for`,
`range` or `=`. So the pool is one set of shapes parameterised by token, not one
set per construct: ~8 shapes x 8 subtypes reused across the inventory, rather than
64 shapes per construct (handout S3).

Subtypes follow the design's eight, plus the other-role subtypes of the symbol
family (`docs/symbol_roles.md`). Applicability is per construct: `identifier` needs
a token that can appear inside a name, `dict_key` and `call_arg` need the token to
survive as a string, and the other-role subtypes exist only where the table names
one. Inapplicable subtypes are reported, never silently skipped.
"""

from __future__ import annotations

import ast
import functools
import keyword
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .menu import Cell, Menu

# f-string is NOT a subtype: its literal part is the same grammatical position and
# role as a plain string (only the delimiter differs), and its replacement field is
# expression context, which is call_arg/identifier territory — and cannot host a
# keyword at all (`f"{for}"` does not parse). A subtype is a distinct *use*, marked by
# position or role; a different delimiter is a *shape*. So the f-string shapes live
# inside `string` as variants. Merging also removes the JoinedStr/FormattedValue
# asymmetry the companion audit found: f-string shapes put JoinedStr in ~20% of B and
# 0% of A, a -0.20 companion difference in every concept profile.
TEXT_SUBTYPES = ("string", "comment", "docstring")
NAME_SUBTYPES = ("identifier", "dict_key", "call_arg", "annotation")
SUBTYPES = TEXT_SUBTYPES + NAME_SUBTYPES + ("other_role",)


# ---------------------------------------------------------------------------
# shapes: (subtype, index) -> lines. `tok` is the surface token, inserted so that
# it is present in the text; whether it survives tokenisation is a filter question.
# ---------------------------------------------------------------------------

def _string_shapes(tok, menu, cell, rng):
    """Token as literal text inside a string. Includes f-string variants (see above)."""
    n = menu.name(cell.domain, "singular", rng)
    p = menu.name(cell.domain, "plural", rng)
    msg = menu.text(cell.domain, rng)
    return [
        [f'{n}_message = "{msg} {tok} now"'],
        [f'{p}_labels = ["{tok} pending", "{msg}"]'],
        [f'if {n}_status == "{tok} ready":', f'    {menu.verb(cell.domain, rng)}_all()'],
        [f'raise ValueError("unexpected {tok} in {msg}")'],
        [f'{n}_header = "|".join(["{tok}", "{msg}"])'],
        [f'print("{msg}", "{tok}", sep=" ")'],
        [f'{p}_map = {{"kind": "{tok}", "state": "{msg}"}}'],
        [f'{n}_doc = ("{msg} "', f'          "{tok} handled")'],
        # Assign-free carriers. A B prompt for `Assign` may contain no assignment at
        # all, which rules out most natural code and was forcing that set onto the
        # def-bearing docstring shapes; these keep it in statement position instead.
        [f'print("{msg}", "{tok}")'],
        [f'assert {n}_ready, "{tok} missing"'],
        [f'raise RuntimeError("{tok}: {msg}")'],
        [f'{menu.verb(cell.domain, rng)}_all("{tok}")'],
    ] + _fstring_shapes(tok, menu, cell, rng)


def _comment_shapes(tok, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    p = menu.name(cell.domain, "plural", rng)
    lit = menu.int_literal(rng)
    return [
        [f"# {tok} is handled upstream", f"{n}_limit = {lit}"],
        [f"{n}_limit = {lit}  # {tok} guard"],
        [f"# TODO: replace {tok} with a lookup", f"{p}_cache = {{}}"],
        [f"{p}_cache = {{}}", f"# {tok} left deliberately"],
        [f"# {tok}", f"{n}_limit = {lit}"],
        [f"{n}_limit = {lit}", f"# see notes on {tok} and {menu.field(cell.domain, rng)}"],
        [f"# {menu.text(cell.domain, rng)} ({tok})", f"{p}_cache = {{}}"],
        [f"{p}_seen = set()  # {tok}: skipped", f"{n}_limit = {lit}"],
        [f"# {tok} handled below", f"print({p}_ready)"],
        [f"import json  # {tok}", f"assert {p}_ready"],
        [f"# {tok} handled below", f"{n}_limit = {lit}", f"{n}_count = {lit}"],
        [f"{n}_limit = {lit}", f"# {tok}", f"{n}_ready = True"],
        [f"# checked for {tok}", f"{n}_ready = False"],
    ]


def _docstring_shapes(tok, menu, cell, rng):
    v = menu.verb(cell.domain, rng)
    p = menu.name(cell.domain, "plural", rng)
    msg = menu.text(cell.domain, rng)
    return [
        [f'"""{tok} handling for {p}."""', f"{p}_cache = {{}}"],
        [f"def {v}_{p}():", f'    """{msg}; {tok} is not used here."""', "    return None"],
        [f"class {menu.name(cell.domain, 'class', rng)}:",
         f'    """{tok} adapter."""', "    pass"],
        [f'"""', f"{msg}", f"Uses no {tok}.", f'"""', f"{p}_cache = {{}}"],
        [f"def {v}_{p}():", f"    '''{tok} free.'''", "    return None"],
        [f'"""{msg} -- {tok}."""', f"{p}_seen = set()"],
        [f"class {menu.name(cell.domain, 'class', rng)}:",
         f'    """Notes: {tok}, {menu.field(cell.domain, rng)}."""',
         f"    {menu.field(cell.domain, rng)} = None"],
        [f"def {v}_{p}():", f'    """{tok}"""', "    return None"],
        [f'"""{tok}."""', f"{p}_ready = True"],
        [f'"""{msg}, {tok}."""', f"{p}_limit = 8"],
        [f'"""{tok} notes."""', f"{p}_ready = False", f"{p}_limit = 4"],
    ]


def _fstring_shapes(tok, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    f = menu.field(cell.domain, rng)
    return [
        [f'{n}_label = f"{tok} {{{n}_{f}}}"'],
        [f'print(f"{{{n}_{f}}} {tok}")'],
        [f'{n}_label = f"{{{n}_{f}!r}} :: {tok}"'],
        [f'raise ValueError(f"{tok} {{{n}_{f}}}")'],
        [f'{n}_rows = [f"{tok} {{i}}" for i in {n}_ids]'],
        [f'{n}_label = f"{tok}" f" {{{n}_{f}}}"'],
        [f'{n}_label = f"{{{n}_{f}:>8}} {tok}"'],
        [f'{n}_msg = f"""{tok} {{{n}_{f}}}"""'],
    ]


def _identifier_shapes(tok, menu, cell, rng):
    """Token inside a name. Position matters, and it is a tokeniser fact.

    Measured on OpenCoder's tokeniser: a keyword survives as a *prefix*
    (`for_limit` -> ['for', '_limit']) but not as a suffix (`position_for`), and not
    capitalised (`ForAdapter` -> ['For', ...], a different token id). A keyword can
    also never stand as a bare name (`for = 45` does not parse). So keyword tokens get
    prefix-only shapes; name tokens (builtins) get the full positional spread.
    """
    n = menu.name(cell.domain, "singular", rng)
    lit = menu.int_literal(rng)
    prefix_only = [
        [f"{tok}_limit = {lit}"],
        [f"{tok}_cache = {{}}", f"{tok}_cache['{n}'] = {lit}"],
        [f"def {tok}_helper({n}_rows):", f"    return {n}_rows"],
        [f"{tok}_seen = set()"],
        [f"{tok}_total = {lit} + {lit}"],
        [f"{tok}_map = {{'{menu.field(cell.domain, rng)}': {lit}}}"],
        [f"{tok}_index, {tok}_count = {lit}, {lit}"],
        [f"{tok}_rows = [{lit}, {lit}]", f"{tok}_head = {tok}_rows[0]"],
    ]
    if keyword.iskeyword(tok) or not tok.isidentifier():
        return prefix_only
    return prefix_only[:4] + [
        [f"{n}_{tok} = {lit}"],
        [f"{tok} = {lit}"],
        [f"class {tok.capitalize()}Adapter:", "    pass"],
        [f"{n}_{tok}_total = {lit} + {lit}"],
    ]


def _dict_key_shapes(tok, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    lit = menu.int_literal(rng)
    return [
        [f'{n}_spec = {{"{tok}": {lit}}}'],
        [f'{n}_spec = {{"{menu.field(cell.domain, rng)}": {lit}, "{tok}": None}}'],
        [f'{n}_value = {n}_spec["{tok}"]'],
        [f'{n}_spec.setdefault("{tok}", {lit})'],
        [f'{n}_spec = dict({{"{tok}": {lit}}})'],
        [f'{n}_keys = ["{tok}"]', f'{n}_spec = {{k: {lit} for k in {n}_keys}}'],
        [f'{n}_spec = {{"{tok}": {{"depth": {lit}}}}}'],
        [f'del {n}_spec["{tok}"]'],
    ]


def _call_arg_shapes(tok, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    v = menu.verb(cell.domain, rng)
    return [
        [f'{v}_one("{tok}")'],
        [f'{v}_one({n}_rows, mode="{tok}")'],
        [f'{n}_out = {v}_one("{tok}", strict=True)'],
        [f'getattr({n}_rows, "{tok}", None)'],
        [f'{v}_one(*["{tok}"])'],
        [f'{n}_out = sorted({n}_rows, key=lambda r: r.get("{tok}"))'],
        [f'{v}_one(**{{"mode": "{tok}"}})'],
        [f'{n}_out = {v}_one("{tok}") or {v}_one("fallback")'],
    ]


def _annotation_shapes(tok, menu, cell, rng):
    n = menu.name(cell.domain, "singular", rng)
    return [
        [f'{n}_rows: "{tok}" = None'],
        [f'def helper({n}_rows: "{tok}") -> None:', "    return None"],
        [f'{n}_map: dict[str, "{tok}"] = {{}}'],
        [f'def helper() -> "{tok}":', "    return None"],
        [f'{n}_rows: list["{tok}"] = []'],
        [f'class Holder:', f'    {n}_rows: "{tok}" = None'],
        [f'def helper({n}_rows: "{tok}" = None):', "    return None"],
        [f'{n}_opt: "{tok} | None" = None'],
    ]


SHAPES: dict[str, Callable] = {
    "string": _string_shapes,
    "comment": _comment_shapes,
    "docstring": _docstring_shapes,
    "identifier": _identifier_shapes,
    "dict_key": _dict_key_shapes,
    "call_arg": _call_arg_shapes,
    "annotation": _annotation_shapes,
}

# other-role snippets come from the symbol table; one entry per (construct, role)
OTHER_ROLE: dict[str, dict[str, Callable]] = {
    "Call": {"grouping_paren": lambda m, c, r: [
        f"{m.name(c.domain, 'total', r)} = ({m.int_literal(r)} + "
        f"{m.int_literal(r)}) * {m.int_literal(r)}"]},
    "Assign": {"keyword_argument": lambda m, c, r: [
        f"{m.verb(c.domain, r)}_one({m.name(c.domain, 'plural', r)}, "
        f"limit={m.int_literal(r)})"]},
    "Attribute": {"dotted_import": lambda m, c, r: ["import os.path",
                                                    f"{m.name(c.domain, 'singular', r)}_root "
                                                    f"= os.path"]},
    "Starred": {
        "multiplication": lambda m, c, r: [
            f"{m.name(c.domain, 'total', r)} = {m.int_literal(r)} * {m.int_literal(r)}"],
        "parameter_star": lambda m, c, r: [
            f"def {m.verb(c.domain, r)}_all(*{m.name(c.domain, 'plural', r)}):",
            "    return None"]},
    "Subscript": {"list_display": lambda m, c, r: [
        f"{m.name(c.domain, 'plural', r)} = [{m.int_literal(r)}, {m.int_literal(r)}]"]},
    "Dict": {"set_display": lambda m, c, r: [
        f"{m.name(c.domain, 'plural', r)}_seen = {{{m.int_literal(r)}, "
        f"{m.int_literal(r)}}}"]},
    "Slice": {
        "dict_key_value": lambda m, c, r: [
            f"{m.name(c.domain, 'singular', r)}_spec = "
            f"{{{m.field(c.domain, r)!r}: {m.int_literal(r)}}}"],
        "suite_header": lambda m, c, r: [
            f"if {m.name(c.domain, 'singular', r)}_ok:",
            f"    {m.name(c.domain, 'total', r)} = {m.int_literal(r)}"]},
    "AnnAssign": {
        "dict_key_value": lambda m, c, r: [
            f"{m.name(c.domain, 'singular', r)}_spec = "
            f"{{{m.field(c.domain, r)!r}: {m.float_literal(r)}}}"],
        "suite_header": lambda m, c, r: [
            f"if {m.name(c.domain, 'singular', r)}_ok:",
            f"    {m.name(c.domain, 'total', r)} = {m.int_literal(r)}"]},
    "Tuple": {"arg_list_separator": lambda m, c, r: [
        f"{m.verb(c.domain, r)}_one({m.name(c.domain, 'plural', r)}, "
        f"{m.int_literal(r)})"]},
}

# constructs whose token cannot appear inside an identifier (symbols, and keywords
# whose surface form does not survive as a name fragment)
_NO_IDENTIFIER = {"Call", "Assign", "Attribute", "Starred", "Subscript", "Dict",
                  "Slice", "AnnAssign", "Tuple", "Compare", "BinOp", "AugAssign",
                  "NamedExpr"}


@dataclass(frozen=True)
class BCore:
    core_id: str
    construct: str
    subtype: str
    shape_index: int
    render: Callable[[Menu, Cell, np.random.Generator], list[str]]


def is_name_token(token: str) -> bool:
    """True when the token can stand as a name in live syntax — i.e. a builtin.

    A reserved keyword cannot: `{"for": 1}`, `f("for")` and `x: "for" = None` all put
    the token inside a *string*, which is the string subtype in disguise rather than an
    independent use. Counting them separately would double-count the same context and
    inflate subtype coverage, so for keywords those three subtypes are inapplicable.
    """
    return token.isidentifier() and not keyword.iskeyword(token) \
        and token not in ("match", "case", "type", "_")


def applicable_subtypes(construct: str, token: str) -> tuple[str, ...]:
    """Which B subtypes exist for this construct. Inapplicable ones are reported."""
    out = list(TEXT_SUBTYPES)
    name_like = is_name_token(token)
    for s in NAME_SUBTYPES:
        if s == "identifier":
            # a keyword survives only as a *fragment* of a name (`for_limit`), and
            # whether the tokeniser still emits the token is an empirical question the
            # validity filter settles — see experiments/probe_bpool.py
            if construct in _NO_IDENTIFIER or not token.isidentifier():
                continue
        elif not name_like:
            continue
        out.append(s)
    if construct in OTHER_ROLE:
        out.append("other_role")
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _shape_ok(construct: str, subtype: str, index: int, token: str) -> bool:
    """Does this shared shape hold up for *this* construct?

    The pool is shared, so a shape may itself contain the construct being measured —
    the docstring shapes host their docstring in a `def` or `class`, and one string
    shape uses an `if`. Those are correct B prompts for `For` and wrong ones for
    FunctionDef, ClassDef and If. Applicability is therefore computed per
    (construct, subtype, shape), not assumed, and the surviving count per subtype
    goes to PROVENANCE.
    """
    probe, cell, rng = _Probe(), Cell("finance", "module", 0, 0), np.random.default_rng(0)
    try:
        lines = SHAPES[subtype](token, probe, cell, rng)[index]
        src = "\n".join(lines) + "\n"
        tree = ast.parse(src)
    except (SyntaxError, IndexError, KeyError):
        return False
    cls = getattr(ast, construct, None)
    if cls is not None and any(isinstance(nd, cls) for nd in ast.walk(tree)):
        return False
    return token in src


@functools.lru_cache(maxsize=None)
def _shape_space_precedes(subtype: str, index: int, token: str) -> bool:
    """Does this shape place the token immediately after a space?

    **The other half of the read-site fix (A19a).** A B prompt carries the construct's token in
    another role, and the role is usually a string or a comment -- where the shape may put the
    token right after the opening quote (`["for pending", ...]`) as easily as after a space
    (`"... for now"`). Those are two different tokens at the read site: `▁for` is id 848 and
    `for` is 2461, and their input embeddings have cosine +0.075, about what two random tokens
    give. Measured on the phase battery before this filter, with A already stable at 100%
    `▁for`: **B was `▁for` 0.66 / `for` 0.34 for `For`, `assert` 0.60 / `▁assert` 0.40 for
    `Assert`, `len` 0.76 / `▁len` 0.24.** So A was fixed and B was not, and 104 of the 118
    constructs failing P₁ failed on the B side alone.

    Cached: it renders and tokenises, it is pure in its three arguments, and `build_b_cores`
    asks it once per shape per construct -- without the cache the test suite went from 42
    seconds to over 400.

    Checked by rendering rather than declared, because the pool is shared and a shape's
    placement of the token is a property of the shape's own text. A shape whose token is at the
    very start of the source counts as space-preceded: a line-initial token is `for`, which is
    what an *unindented* A prompt reads -- and the primary reading is the indented cells, so a
    line-initial B would be the wrong half of the pair.
    """
    probe, cell, rng = _Probe(), Cell("finance", "module", 0, 0), np.random.default_rng(0)
    try:
        src = "\n".join(SHAPES[subtype](token, probe, cell, rng)[index]) + "\n"
    except (SyntaxError, IndexError, KeyError):
        return False
    i = src.find(token)
    if i <= 0:
        return False
    return src[i - 1] == " "


def shape_signature(construct: str, subtype: str, index: int,
                    token: str) -> frozenset[str]:
    """Which inventory constructs a carrier shape contributes on its own."""
    from .audit import present

    probe, cell, rng = _Probe(), Cell("finance", "module", 0, 0), np.random.default_rng(0)
    try:
        lines = SHAPES[subtype](token, probe, cell, rng)[index]
    except (IndexError, KeyError):
        return frozenset()
    return frozenset(present("\n".join(lines) + "\n", construct))


#: construct -> the B subtypes dropped because no shape of theirs places the token after a
#: space. Read by the build for the manifest: a subtype that disappears has to be visible,
#: and `BCore` is frozen so it cannot carry the fact itself.
DROPPED_SUBTYPES: dict[str, tuple[str, ...]] = {}


def build_b_cores(construct: str, token: str, n_per_subtype: int,
                  rng: np.random.Generator,
                  target: dict[str, float] | None = None) -> list[BCore]:
    """>=n distinct shapes per applicable subtype, drawn from the shared pool.

    Shapes that do not hold up for this construct are dropped here rather than
    rejected prompt by prompt downstream, so the achieved count per subtype is known
    before any prompt is generated.
    """
    cores: list[BCore] = []
    dropped_subtypes: list[str] = []
    for sub in applicable_subtypes(construct, token):
        if sub == "other_role":
            # Not filtered for the leading space, and it costs nothing: `OTHER_ROLE` is
            # declared only for `Call`, `Assign` and `Attribute`, whose tokens are symbols and
            # so outside the filter's scope anyway (a symbol merges with its neighbours
            # whatever precedes it -- ledger L5).
            for role, fn in OTHER_ROLE[construct].items():
                cores.append(BCore(f"{construct}/B:{role}", construct, f"other_role:{role}",
                                   0, fn))
            continue
        n_shapes = len(SHAPES[sub](token, _Probe(), Cell("finance", "module", 0, 0),
                                   np.random.default_rng(0)))
        usable = [j for j in range(n_shapes) if _shape_ok(construct, sub, j, token)]
        # Keep only the shapes that place the token after a space, so B reads the same token as
        # A does (A19a). Never below two shapes for a subtype: a subtype that cannot supply one
        # is a fact for P₁ to report, not a reason to build a thinner B set -- and the
        # unfiltered list is the honest fallback because an empty subtype is worse than a mixed
        # one, the first being no contrast at all.
        spaced = [j for j in usable
                  if token.strip().isidentifier()
                  and _shape_space_precedes(sub, j, token)]
        # A subtype with no space-preceding shape is **inapplicable** for this construct, and
        # is dropped and reported -- the convention this module already follows for subtypes a
        # construct's grammar rules out. `annotation`, `call_arg` and `dict_key` put the token
        # straight after a quote in *every* shape they have (`-> "AttributeError":`,
        # `: "AttributeError" =`), so for an identifier token they cannot supply a read site
        # that matches A's. Keeping them cost `AttributeError` its P₁ pass at 0.51 against
        # 0.49, which is no contrast rather than a weak one.
        if spaced:
            usable = spaced
        elif token.strip().isidentifier() and usable:
            dropped_subtypes.append(sub)
            continue
        if target is None:
            picks = [usable[i] for i in rng.permutation(len(usable))[:n_per_subtype]]
        else:
            # Garnishes can add a companion but never remove one, so the *excess* — what
            # a carrier contributes that the A set rarely has — is controllable only
            # here. Rank carriers by that excess and keep the lowest.
            def excess(j):
                sig = shape_signature(construct, sub, j, token)
                return sum(max(0.0, 1.0 - target.get(c, 0.0)) for c in sig)
            picks = sorted(usable, key=lambda j: (excess(j), j))[:n_per_subtype]
        for j in picks:
            def render(m, c, r, _sub=sub, _j=int(j)):
                return SHAPES[_sub](token, m, c, r)[_j]
            cores.append(BCore(f"{construct}/B:{sub}/s{int(j)}", construct, sub,
                               int(j), render))
    if dropped_subtypes:
        DROPPED_SUBTYPES[construct] = tuple(dropped_subtypes)
    if not cores:
        raise ValueError(f"{construct}: every B subtype dropped -- no contrast possible")

    return cores


def shape_census(construct: str, token: str) -> dict[str, int]:
    """Usable shapes per subtype for one construct — the PROVENANCE row."""
    out = {}
    for sub in applicable_subtypes(construct, token):
        if sub == "other_role":
            out[sub] = len(OTHER_ROLE[construct])
            continue
        n = len(SHAPES[sub](token, _Probe(), Cell("finance", "module", 0, 0),
                            np.random.default_rng(0)))
        out[sub] = sum(1 for j in range(n) if _shape_ok(construct, sub, j, token))
    return out


class _Probe:
    """Minimal menu used only to count how many shapes a subtype offers."""

    def name(self, d, r, rng): return "x"
    def verb(self, d, rng): return "v"
    def field(self, d, rng): return "f"
    def int_literal(self, rng): return "1"
    def float_literal(self, rng): return "1.0"
    def text(self, d, rng): return "t"
    def padding(self, d, n, rng): return []
