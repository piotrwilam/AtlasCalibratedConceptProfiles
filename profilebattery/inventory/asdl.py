"""Parse CPython's `Parser/Python.asdl` — the inventory's source of truth.

The abstract grammar is the file CPython generates the `ast` module from, so taking
the inventory from it rather than from a hand-picked list is what makes "why these
constructs?" answerable: the answer is "every constructor of Python's own grammar, less
a checkable exclusion predicate" (report S6).

The format is small enough to parse directly. A *sum* type lists constructors separated
by `|`; a *product* type is a bare field list and has no constructor of its own, which
is already half of predicate R.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

COMMENT = re.compile(r"--.*$", re.MULTILINE)
ATTRIBUTES = re.compile(r"\battributes\s*\([^)]*\)")
#: A type definition is a line of the form `name = ...`. ASDL field lists carry no `=`,
#: and alternatives begin with `|`, so this is unambiguous — and unlike an
#: indentation-column rule it does not break if CPython reformats the file.
TYPEDEF = re.compile(r"^[ \t]*(\w+)[ \t]*=[ \t]*(.*?)(?=^[ \t]*\w+[ \t]*=|\Z)",
                     re.MULTILINE | re.DOTALL)
CONSTRUCTOR = re.compile(r"(\w+)\s*(?:\(|$|\|)")


@dataclass
class AsdlType:
    name: str
    kind: str                      # "sum" | "product"
    constructors: list[str] = field(default_factory=list)


def parse(text: str) -> dict[str, AsdlType]:
    """Type name -> AsdlType, in file order."""
    body = COMMENT.sub("", text)
    start = body.index("{")
    end = body.rindex("}")
    body = body[start + 1:end]
    out: dict[str, AsdlType] = {}
    for name, rhs in TYPEDEF.findall(body):
        rhs = ATTRIBUTES.sub("", rhs).strip()
        if rhs.startswith("("):
            out[name] = AsdlType(name, "product", [])
            continue
        cons: list[str] = []
        for alt in _split_alternatives(rhs):
            alt = alt.strip()
            if not alt:
                continue
            m = re.match(r"(\w+)", alt)
            if m:
                cons.append(m.group(1))
        out[name] = AsdlType(name, "sum", cons)
    return out


def _split_alternatives(rhs: str) -> list[str]:
    """Split on `|` at paren depth 0 — field lists may contain none, but be exact."""
    parts, depth, cur = [], 0, []
    for ch in rhs:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def load(path: str) -> dict[str, AsdlType]:
    with open(path) as fh:
        return parse(fh.read())
