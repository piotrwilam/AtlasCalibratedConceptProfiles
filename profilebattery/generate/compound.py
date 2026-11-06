"""Family grammar 1 of 6: compound statements (header + suite).

Covers For, While, If, With, Try, FunctionDef, ClassDef, Match. Cores are produced
by crossing three declared structural axes and selecting a spread that touches
every level of each:

  operand  what the header binds or tests   (per construct)
  body     what the suite is for            (shared across the family)
  suite    the suite's shape                (simple / two_stmt / companion / else)

The leverage is that `body` and `suite` are shared: one set of body generators
serves every compound construct, so eight structurally distinct cores per
construct cost one spec row rather than eight hand-written templates.

Two constraints the grammar must respect, both discovered by construction:

* **Cores are scope-independent.** The grid's scope dimension wraps a core in a
  module / function / method frame, so a core may not contain `return` (it would
  not parse at module scope). `return` appears only in FunctionDef's own bodies,
  where the frame is the construct.
* **The anchor is a line, not a uniqueness claim.** FunctionDef and ClassDef cores
  collide with the scope wrapper, which emits a `def`/`class` of its own, so "the
  unique node of this type" is not well defined. Every core therefore reports the
  index of the line carrying its construct's own token, and verification and
  anchoring both key on that line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import scopeuse as _scopeuse
from . import weights as _weights
from .core import Core, spread
from .menu import Cell, Menu

IND = "    "

#: How a body reaches into its binding. Real Python is attribute-dense: the corpus puts
#: `Attribute` in 0.77-0.98 of every compound construct's span and `Subscript` in
#: 0.17-0.51, while a grammar whose bodies all wrote `entry["field"]` produced 0.12 and
#: 0.75 — wrong in both directions at once. Access form is therefore an axis, not a fixed
#: choice, and the four levels are balanced across cores like every other axis.
ACCESS = ("subscript", "attribute", "self_attribute", "bare")


def realised_access(form: str, scope: str) -> str:
    """The access form a cell actually gets. `self` is only bound in a method, so
    `self_attribute` degrades to a plain attribute anywhere else. The *nominal* level
    is the grid coordinate; this is what the prompt contains, and it is what the
    per-prompt metadata must record — reweighting a prompt to the corpus rate of
    `self.` when the prompt says `rows.field` misdescribes the measure."""
    if form == "self_attribute" and scope != "method":
        return "attribute"
    return form


def access(form: str, binding: str, menu: Menu, cell: Cell,
           rng: np.random.Generator) -> str:
    f = menu.field(cell.domain, rng)
    return {"subscript": f"{binding}[{f!r}]",
            "attribute": f"{binding}.{f}",
            "self_attribute": f"self.{f}",
            "bare": binding}[realised_access(form, cell.scope)]


@dataclass(frozen=True)
class CoreText:
    """Rendered core: unindented lines, and which line carries the construct."""

    lines: list[str]
    anchor_line: int


def _ind(lines: list[str], n: int = 1) -> list[str]:
    return [IND * n + ln if ln.strip() else ln for ln in lines]


# ---------------------------------------------------------------------------
# bodies — shared across the family. Each returns unindented statement lines.
# `binding` is the name the header made available (loop variable, `as` target…).
# ---------------------------------------------------------------------------

def _b_accumulate(binding, menu, cell, rng, acc="subscript"):
    total = menu.name(cell.domain, "total", rng)
    return [f"{total} += {access(acc, binding, menu, cell, rng)}"]


def _b_transform_append(binding, menu, cell, rng, acc="subscript"):
    out = f"{menu.name(cell.domain, 'plural', rng)}_scaled"
    return [f"{out}.append({access(acc, binding, menu, cell, rng)} * "
            f"{menu.float_literal(rng)})"]


def _b_mutate(binding, menu, cell, rng, acc="subscript"):
    lhs = access(acc if acc != "bare" else "subscript", binding, menu, cell, rng)
    return [f"{lhs} = {lhs} // {menu.int_literal(rng)}"]


def _b_delegate(binding, menu, cell, rng, acc="bare"):
    """Hand the binding to a function -- and to its second argument.

    The second mention is the one edit on this axis that is free: a Call with two arguments
    carries the same companion multiset as a Call with one, so the body's contribution to the
    companion fit is unchanged while its name density rises from one to two. Every other
    template needs an operator, a subscript or a call to mention the binding again, and each of
    those *is* a companion -- which is why density cannot be raised template by template
    without re-fitting the mixture (see `experiments/binding_uses_corpus.py` step 1, and the
    body-density table in the 30 Sept tests).
    """
    a = access(acc, binding, menu, cell, rng)
    return [f"{menu.verb(cell.domain, rng)}_{menu.name(cell.domain, 'singular', rng)}"
            f"({a}, {binding})" if binding else
            f"{menu.verb(cell.domain, rng)}_{menu.name(cell.domain, 'singular', rng)}({a})"]


def _b_filter_continue(binding, menu, cell, rng, acc="subscript"):
    return [f"if {access(acc, binding, menu, cell, rng)} < {menu.int_literal(rng)}:",
            IND + "continue",
            f"{menu.verb(cell.domain, rng)}_ok({binding})"]


def _b_early_exit(binding, menu, cell, rng, acc="subscript"):
    return [f"if {access(acc, binding, menu, cell, rng)} > {menu.int_literal(rng)}:",
            IND + "break"]


def _b_bind(binding, menu, cell, rng, acc="bare"):
    """The lightest body: bind the access to a local.

    Added because weighting a mixture can only reach the convex hull of its levels, and
    every existing body emitted at least one capped companion — `mutate` and
    `transform_append` a `BinOp`, `accumulate` an `AugAssign`, `delegate` a `Call`. The
    corpus targets for the sparser constructs sit outside that hull, so the fit could only
    trade one companion for another. `Assign` is exempt as quasi-obligatory, so with
    `acc="bare"` this body contributes no capped companion at all and the hull extends
    toward the low-companion corner.
    """
    return [f"{menu.name(cell.domain, 'singular', rng)}_last = "
            f"{access(acc, binding, menu, cell, rng)}"]


def _b_count(binding, menu, cell, rng, acc="bare"):
    """Increment a counter: an AugAssign and nothing else, not even an access."""
    return [f"{menu.name(cell.domain, 'total', rng)} += 1"]


def _b_return_value(binding, menu, cell, rng, acc="bare"):
    """FunctionDef only — the frame is the construct, so `return` is in scope."""
    return [f"return {access(acc, binding, menu, cell, rng)} * "
            f"{menu.int_literal(rng)}"]


def _b_class_attr(binding, menu, cell, rng, acc="bare"):
    """ClassDef only — a class body takes assignments and defs, not statements."""
    return [f"{menu.field(cell.domain, rng)}_limit = {menu.int_literal(rng)}"]


def _b_class_method(binding, menu, cell, rng, acc="bare"):
    return [f"def {menu.verb(cell.domain, rng)}(self):",
            IND + f"return self.{menu.field(cell.domain, rng)}_limit"]


BODIES: dict[str, Callable] = {
    "accumulate": _b_accumulate,
    "transform_append": _b_transform_append,
    "mutate": _b_mutate,
    "delegate": _b_delegate,
    "filter_continue": _b_filter_continue,
    "early_exit": _b_early_exit,
    "bind": _b_bind,
    "count": _b_count,
    "return_value": _b_return_value,
    "class_attr": _b_class_attr,
    "class_method": _b_class_method,
}

LOOP_BODIES = ("accumulate", "transform_append", "mutate", "delegate",
               "filter_continue", "early_exit", "bind", "count")
PLAIN_BODIES = ("accumulate", "transform_append", "mutate", "delegate", "bind", "count")


# ---------------------------------------------------------------------------
# suites — how the body is placed, and what companion the core drags
# ---------------------------------------------------------------------------

def _suite(body_lines, suite, companion_lines, menu, cell, rng, binding=None):
    """Compose the body into its suite shape.

    `two_stmt`'s second statement read nothing at all -- an `Assign` of an int literal -- which
    is why the multi-statement bodies of `FunctionDef` used their parameters a median of zero
    times against a corpus median of four at three statements and eight at four or more
    (`validation/positions/BINDING_USES.json`). Reading the binding there instead of a literal
    is free on the companion axis: the statement is an `Assign` either way, and `Name` and
    `Constant` are not constructs of the inventory, so nothing the cap tracks changes. It buys
    one dependent position per multi-statement body, which closes part of that gap rather than
    all of it -- the rest needs body templates that touch the binding in every statement, as
    real functions do, and that is a change to the mixture's hull.
    """
    if suite == "simple":
        return body_lines
    if suite == "two_stmt":
        rhs = binding if binding else f"{menu.int_literal(rng)}"
        return body_lines + [f"{menu.verb(cell.domain, rng)}_count = {rhs}"]
    if suite == "companion":
        return companion_lines + _ind(body_lines)
    raise KeyError(suite)


def _companion_if(menu, cell, rng):
    return [f"if {menu.name(cell.domain, 'singular', rng)}_ok:"]


def _companion_for(menu, cell, rng):
    v = menu.name(cell.domain, "singular", rng)
    return [f"for {v}_part in {menu.name(cell.domain, 'plural', rng)}_parts:"]


# ---------------------------------------------------------------------------
# per-construct shells
# ---------------------------------------------------------------------------

def _iterable(level, menu, cell, rng):
    plural = menu.name(cell.domain, "plural", rng)
    if level == "range":
        return f"range({menu.int_literal(rng)})"
    if level == "list_literal":
        # dict elements, not ints: the shared bodies subscript their binding, and a
        # type-incoherent core parses but is off-distribution.
        f = menu.field(cell.domain, rng)
        return (f"[{{{f!r}: {menu.int_literal(rng)}}}, "
                f"{{{f!r}: {menu.int_literal(rng)}}}]")
    if level == "name":
        return plural
    if level == "items":
        return f"{plural}.items()"
    if level == "zip":
        return f"zip({plural}, {menu.name(cell.domain, 'plural', rng)}_ref)"
    if level == "enumerate":
        return f"enumerate({plural})"
    if level == "call_result":
        return f"sorted({plural})"
    if level == "comprehension":
        return f"[r for r in {plural} if r]"
    if level == "attribute":
        # 0.118 of corpus `for` loops iterate an attribute (`for x in self.rows`) and
        # 0.041 a subscript; both were missing from the grammar's operand axis
        return f"self.{plural}" if cell.scope == "method" else f"{plural}_store.{plural}"
    if level == "subscript":
        return f"{plural}_by_key[{menu.field(cell.domain, rng)!r}]"
    raise KeyError(level)


def _shell_for(operand, body, suite, menu, cell, rng, acc="subscript"):
    var = menu.name(cell.domain, "singular", rng)
    target = f"{var}_key, {var}_value" if operand in ("items", "zip") else var
    target = f"{var}_index, {var}" if operand == "enumerate" else target
    binding = f"{var}_value" if operand in ("items", "zip") else var
    if operand == "range":
        binding = None
    it = _iterable(operand, menu, cell, rng)
    head = [f"for {target} in {it}:"]
    if binding is None:
        inner = [f"{menu.name(cell.domain, 'total', rng)} += {var} * "
                 f"{menu.int_literal(rng)}"]
    else:
        inner = BODIES[body](binding, menu, cell, rng, acc)
    if suite == "else":
        lines = head + _ind(inner) + ["else:"] + _ind(
            [f"{menu.verb(cell.domain, rng)}_done = True"])
    else:
        lines = head + _ind(_suite(inner, suite, _companion_if(menu, cell, rng),
                                  menu, cell, rng, binding))
    return CoreText(lines, 0)


def _shell_while(operand, body, suite, menu, cell, rng, acc="subscript"):
    var = menu.name(cell.domain, "singular", rng)
    plural = menu.name(cell.domain, "plural", rng)
    cond = {
        "counter": f"{var}_index < {menu.int_literal(rng)}",
        "flag": f"not {var}_done",
        "queue": f"{plural}_queue",
        "compare_call": f"len({plural}_queue) > {menu.int_literal(rng)}",
    }[operand]
    head = [f"while {cond}:"]
    inner = BODIES[body](f"{var}_head", menu, cell, rng, acc) + \
        [f"{var}_index += 1" if operand == "counter" else f"{var}_done = True"]
    if suite == "else":
        lines = head + _ind(inner) + ["else:"] + _ind(
            [f"{menu.verb(cell.domain, rng)}_drained = True"])
    else:
        lines = head + _ind(_suite(inner, suite, _companion_if(menu, cell, rng),
                                   menu, cell, rng))
    return CoreText(lines, 0)


def _shell_if(operand, body, suite, menu, cell, rng, acc="subscript"):
    var = menu.name(cell.domain, "singular", rng)
    plural = menu.name(cell.domain, "plural", rng)
    f = menu.field(cell.domain, rng)
    cond = {
        "compare": f"{var}_head[{f!r}] > {menu.int_literal(rng)}",
        "membership": f"{f!r} in {var}_head",
        "call_truth": f"len({plural}) > 0",
        "attribute_truth": f"{var}_head.is_ready",
        "isinstance": f"isinstance({var}_head, dict)",
    }[operand]
    head = [f"if {cond}:"]
    inner = BODIES[body](f"{var}_head", menu, cell, rng, acc)
    if suite == "else":
        lines = head + _ind(inner) + ["else:"] + _ind(
            [f"{menu.verb(cell.domain, rng)}_skipped = True"])
    else:
        lines = head + _ind(_suite(inner, suite, _companion_for(menu, cell, rng),
                                   menu, cell, rng))
    return CoreText(lines, 0)


def _shell_with(operand, body, suite, menu, cell, rng, acc="subscript"):
    var = menu.name(cell.domain, "singular", rng)
    plural = menu.name(cell.domain, "plural", rng)
    ctx = {
        "open_file": f"open({plural + '.csv'!r}) as {var}_handle",
        "lock": f"{plural}_lock as {var}_handle",
        "context_call": f"{menu.verb(cell.domain, rng)}_scope({plural}) as {var}_handle",
        "two_items": f"open({plural + '.csv'!r}) as {var}_handle, "
                     f"{plural}_lock as {var}_guard",
    }[operand]
    head = [f"with {ctx}:"]
    binding = f"{var}_handle"
    inner = BODIES[body](binding, menu, cell, rng, acc)
    lines = head + _ind(_suite(inner, suite if suite != "else" else "two_stmt",
                               _companion_if(menu, cell, rng), menu, cell, rng,
                               binding))
    return CoreText(lines, 0)


def _shell_try(operand, body, suite, menu, cell, rng, acc="subscript"):
    var = menu.name(cell.domain, "singular", rng)
    inner = BODIES[body](f"{var}_head", menu, cell, rng, acc)
    tail = {
        "except_named": ["except KeyError as exc:",
                         IND + f"{menu.verb(cell.domain, rng)}_failed = exc"],
        "except_tuple": ["except (KeyError, ValueError):",
                         IND + f"{menu.verb(cell.domain, rng)}_failed = True"],
        "else_clause": ["except KeyError:", IND + "pass", "else:",
                        IND + f"{menu.verb(cell.domain, rng)}_ok = True"],
        "finally_clause": ["except KeyError:", IND + "pass", "finally:",
                           IND + f"{menu.verb(cell.domain, rng)}_closed = True"],
    }[operand]
    body_lines = _suite(inner, suite if suite in ("simple", "two_stmt") else "simple",
                        _companion_if(menu, cell, rng), menu, cell, rng)
    return CoreText(["try:"] + _ind(body_lines) + tail, 0)


def _shell_functiondef(operand, body, suite, menu, cell, rng, acc="subscript"):
    name = f"{menu.verb(cell.domain, rng)}_{menu.name(cell.domain, 'plural', rng)}"
    arg = menu.name(cell.domain, "singular", rng)
    # a method carries `self`; without it the def is off-distribution against a corpus in
    # which three quarters of `def`s are methods, and its own body already writes `self.`
    lead = "self, " if cell.scope == "method" else ""
    bare = "self" if cell.scope == "method" else ""
    params = {
        "no_params": bare,
        "positional": f"{lead}{arg}_rows",
        "defaults": f"{lead}{arg}_rows, limit={menu.int_literal(rng)}",
        "kwonly": f"{lead}{arg}_rows, *, strict=False",
        "annotated": f"{lead}{arg}_rows: list, limit: int = {menu.int_literal(rng)}",
        "varargs": f"{lead}*{arg}_rows",
    }[operand]
    binding = f"{arg}_rows" if operand != "no_params" else f"{arg}_default"
    head = [f"def {name}({params}):"]
    inner = ([f"{binding} = {menu.int_literal(rng)}"] if operand == "no_params" else []) \
        + BODIES[body](binding, menu, cell, rng, acc)
    lines = head + _ind(_suite(inner, suite if suite != "else" else "two_stmt",
                               _companion_if(menu, cell, rng), menu, cell, rng,
                               binding))
    return CoreText(lines, 0)


def _shell_classdef(operand, body, suite, menu, cell, rng, acc="subscript"):
    name = menu.name(cell.domain, "class", rng)
    head = {
        "bare": [f"class {name}:"],
        "base": [f"class {name}(dict):"],
        "keyword_base": [f"class {name}(dict, metaclass=type):"],
        "docstring": [f"class {name}:", IND + f'"""{menu.text(cell.domain, rng)}."""'],
    }[operand]
    inner = BODIES[body](name, menu, cell, rng, acc)
    if operand == "docstring":
        return CoreText([head[0]] + [head[1]] + _ind(inner), 0)
    lines = head + _ind(_suite(inner, suite if suite in ("simple", "two_stmt") else "simple",
                               _companion_if(menu, cell, rng), menu, cell, rng))
    return CoreText(lines, 0)


def _shell_match(operand, body, suite, menu, cell, rng, acc="subscript"):
    var = menu.name(cell.domain, "singular", rng)
    first = {
        "literal": f"case {menu.int_literal(rng)}:",
        "capture": f"case [{var}_first, *{var}_rest]:",
        "mapping": f"case {{{menu.field(cell.domain, rng)!r}: {var}_found}}:",
        "guarded": f"case {var}_any if {var}_any:",
    }[operand]
    inner = BODIES[body](f"{var}_head", menu, cell, rng, acc)
    if suite == "two_stmt":
        inner = inner + [f"{menu.verb(cell.domain, rng)}_count = "
                         f"{menu.int_literal(rng)}"]
    lines = [f"match {var}_head:"] + _ind([first]) + _ind(inner, 2)
    if suite == "three_cases":
        lines += _ind([f"case {menu.int_literal(rng)}:"]) + _ind(
            [f"{menu.verb(cell.domain, rng)}_alt = True"], 2)
    # the wildcard case closes every match; `pass` under it is the filler, and the
    # companion audit reports it at share 1.0 because it is structural, not incidental
    lines += _ind(["case _:"]) + _ind(["pass"], 2)
    return CoreText(lines, 0)


def _async_frame(lines: list[str], menu: Menu, cell: Cell,
                 rng: np.random.Generator) -> list[str]:
    """`async for` and `async with` may not sit at module level, so the core supplies
    its own `async def`. That frame is a **component**, not a companion: it is
    grammatically obligatory, exactly as a loop is for `break`.
    """
    return [f"async def {menu.verb(cell.domain, rng)}_stream("
            f"{menu.name(cell.domain, 'plural', rng)}):"] + _ind(lines)


def _prefix_async(shell: Callable) -> Callable:
    """The async compound statements are their sync shells with an `async` prefix.

    They are not a sixth family: the header, body and suite axes are identical, which is
    why the composite/atomic class can keep all fifteen of its members rather than
    carrying an "except the async ones" caveat.
    """
    def wrapped(operand, body, suite, menu, cell, rng, acc="subscript"):
        text = shell(operand, body, suite, menu, cell, rng, acc)
        lines = list(text.lines)
        lines[text.anchor_line] = "async " + lines[text.anchor_line]
        return CoreText(lines, text.anchor_line)
    return wrapped


@dataclass(frozen=True)
class CompoundSpec:
    construct: str
    operands: tuple[str, ...]
    bodies: tuple[str, ...]
    suites: tuple[str, ...]
    shell: Callable
    companions: tuple[str, ...]
    frame: str | None = None


SPECS: dict[str, CompoundSpec] = {
    "For": CompoundSpec("For",
                        ("range", "list_literal", "name", "items", "zip",
                         "enumerate", "call_result", "comprehension", "attribute",
                         "subscript"),
                        LOOP_BODIES, ("simple", "two_stmt", "companion", "else"),
                        _shell_for, ("Call", "Subscript")),
    "While": CompoundSpec("While", ("counter", "flag", "queue", "compare_call"),
                          LOOP_BODIES, ("simple", "two_stmt", "companion", "else"),
                          _shell_while, ("Compare", "AugAssign")),
    "If": CompoundSpec("If",
                       ("compare", "membership", "call_truth", "attribute_truth",
                        "isinstance"),
                       PLAIN_BODIES, ("simple", "two_stmt", "companion", "else"),
                       _shell_if, ("Compare",)),
    "With": CompoundSpec("With", ("open_file", "lock", "context_call", "two_items"),
                         PLAIN_BODIES, ("simple", "two_stmt", "companion"),
                         _shell_with, ("Call",)),
    "Try": CompoundSpec("Try",
                        ("except_named", "except_tuple", "else_clause",
                         "finally_clause"),
                        PLAIN_BODIES, ("simple", "two_stmt"),
                        _shell_try, ("ExceptHandler",)),
    "FunctionDef": CompoundSpec("FunctionDef",
                                ("no_params", "positional", "defaults", "kwonly",
                                 "annotated", "varargs"),
                                PLAIN_BODIES + ("return_value",),
                                ("simple", "two_stmt", "companion"),
                                _shell_functiondef, ()),
    "ClassDef": CompoundSpec("ClassDef",
                             ("bare", "base", "keyword_base", "docstring"),
                             ("class_attr", "class_method"), ("simple", "two_stmt"),
                             _shell_classdef, ("FunctionDef",)),
    # the three async compound statements: the same axes, an `async` prefix, and for
    # AsyncFor/AsyncWith an obligatory `async def` frame (a component, not a companion)
    "AsyncFunctionDef": CompoundSpec("AsyncFunctionDef",
                                     ("no_params", "positional", "defaults", "kwonly",
                                      "annotated", "varargs"),
                                     PLAIN_BODIES + ("return_value",),
                                     ("simple", "two_stmt", "companion"),
                                     _prefix_async(_shell_functiondef), ()),
    "AsyncFor": CompoundSpec("AsyncFor",
                             ("range", "list_literal", "name", "items", "zip",
                              "enumerate", "call_result", "comprehension", "attribute",
                              "subscript"),
                             LOOP_BODIES, ("simple", "two_stmt", "companion", "else"),
                             _prefix_async(_shell_for),
                             ("AsyncFunctionDef", "Call", "Subscript"),
                             frame="async"),
    "AsyncWith": CompoundSpec("AsyncWith",
                              ("open_file", "lock", "context_call", "two_items"),
                              PLAIN_BODIES, ("simple", "two_stmt", "companion"),
                              _prefix_async(_shell_with),
                              ("AsyncFunctionDef", "Call"), frame="async"),
    "Match": CompoundSpec("Match", ("literal", "capture", "mapping", "guarded"),
                          PLAIN_BODIES, ("simple", "two_stmt", "three_cases"),
                          _shell_match, ("MatchAs", "Pass")),
}

FRAME_CLASS = "header+suite"


def build_cores(construct: str, n: int, rng: np.random.Generator) -> list[Core]:
    """n structurally distinct cores for one compound construct (handout S3)."""
    spec = SPECS[construct]
    levels = {"operand": spec.operands, "body": spec.bodies,
              "suite": spec.suites, "access": ACCESS}
    # item 95: `FunctionDef` and `AsyncFunctionDef` bind their own name, and a later use of it
    # is C2's scope class for them. Omitted where it would have one level, so no other compound
    # construct's cores change.
    _trailing = _scopeuse.trailing_levels(construct)
    if len(_trailing) > 1:
        levels["trailing"] = _trailing
    combos = spread(levels, n, rng,
                    weights=_weights.for_construct(construct))
    cores = []
    for i, axes in enumerate(combos):
        def render(menu: Menu, cell: Cell, r: np.random.Generator, _a=axes) -> CoreText:
            text = spec.shell(_a["operand"], _a["body"], _a["suite"], menu, cell,
                              r, _a["access"])
            if spec.frame == "async":
                return CoreText(_async_frame(list(text.lines), menu, cell, r),
                                text.anchor_line + 1)
            # the trailing use sits after the construct's whole suite, at the construct's own
            # indentation, which is what puts it in the same block as the binding
            return CoreText(_scopeuse.apply_trailing(_a.get("trailing", "none"), list(text.lines),
                                                     menu, cell, r,
                                                     construct=construct),
                            text.anchor_line)
        cores.append(Core(core_id=f"{construct}/c{i}", construct=construct,
                          family="compound", axes=axes,
                          frame_class="async def + header+suite" if spec.frame == "async"
                          else FRAME_CLASS,
                          companions=spec.companions, render=render))
    return cores
