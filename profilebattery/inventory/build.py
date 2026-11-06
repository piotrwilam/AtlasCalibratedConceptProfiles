"""Assemble `inventory_v2.json` — handout S1.

Everything here is derived: the construct list from CPython's ASDL under predicate R,
the builtins from the `builtins` module, the token collisions from the token table, the
contrast class from the token's lexical kind, and the B subtypes from the pool's own
applicability rules. What is *declared* rather than derived — frame class, components —
is declared in one table each, so the judgement is visible and reviewable.

Tier is left null here and filled by `experiments/assign_tiers.py` once the S2
calibration tables exist: the deep tier is a union of four claim-driven sets with a
frequency arm (`profilebattery/inventory/tier.py`), so it cannot be known at build
time. A null tier in a shipped inventory means that step has not been run.
"""

from __future__ import annotations
from profilebattery.paths import data_root

import builtins as _builtins
import csv
import json
import keyword
import os
import pathlib

from .asdl import load
from .predicate_r import classify
from .tokens import NODE_TOKENS, NO_TOKEN, builtin_tokens, siblings, tokens_of

DATA = (data_root() / "inventory")
ASDL = DATA / "Python-3.12.13.asdl"
V1_LABELS = pathlib.Path(os.path.expanduser(
    "~/Code/No-Deeper-No-Later_private/data/compositional_level.csv"))
V1_FAMILIES = pathlib.Path(os.path.expanduser(
    "~/Code/No-Deeper-No-Later_private/data/families.csv"))

#: the obligatory frame a construct drags — declared, not derived
FRAME_CLASS = {
    **{c: "header+suite" for c in ("FunctionDef", "AsyncFunctionDef", "ClassDef", "For",
                                   "AsyncFor", "While", "If", "With", "AsyncWith", "Try",
                                   "TryStar", "Match")},
    **{c: "requires function" for c in ("Return", "Global", "Nonlocal", "Yield",
                                        "YieldFrom", "Await")},
    **{c: "requires loop" for c in ("Break", "Continue")},
    **{c: "requires match" for c in ("MatchValue", "MatchSequence", "MatchMapping",
                                     "MatchClass", "MatchStar", "MatchAs", "MatchOr")},
    **{c: "requires type parameter list" for c in ("TypeVar", "ParamSpec",
                                                   "TypeVarTuple")},
    **{c: "comprehension" for c in ("ListComp", "SetComp", "DictComp", "GeneratorExp")},
}

#: companions entailed by the construct's own syntax — exempt from the companion cap
COMPONENTS = {
    "Try": ("ExceptHandler",), "TryStar": ("ExceptHandler",),
    "Match": ("MatchAs", "Pass"),
    "For": ("Break", "Continue"), "AsyncFor": ("Break", "Continue"),
    "While": ("Break", "Continue"),
    "Return": ("FunctionDef",), "Global": ("FunctionDef", "Assign"),
    "Nonlocal": ("FunctionDef",), "Yield": ("FunctionDef",),
    "YieldFrom": ("FunctionDef",), "Await": ("AsyncFunctionDef",),
    "Break": ("For",), "Continue": ("For",),
    "Slice": ("Subscript",),
}


def contrast_class(token: str | None) -> str:
    if token is None:
        return "structural"
    if keyword.iskeyword(token) or token in ("match", "case", "type", "_"):
        return "keyword-token"
    if token.isidentifier():
        return "named-builtin"
    return "structural"


def verifier_kind(construct: str, is_builtin: bool) -> str:
    if construct in ("True", "False", "None"):
        return "keyword-constant"
    return "name" if is_builtin else "ast-node"


def _v1_labels() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if V1_LABELS.exists():
        for row in csv.DictReader(V1_LABELS.open()):
            out.setdefault(row["concept"], {}).update(
                compositional_level=row.get("compositional_level"),
                category=row.get("category"))
    if V1_FAMILIES.exists():
        for row in csv.DictReader(V1_FAMILIES.open()):
            out.setdefault(row["concept"], {}).update(family=row.get("family"))
    return out


def covered_constructs() -> dict[str, str]:
    """construct -> A-side family grammar, for those a family can generate."""
    from ..generate.families import FAMILY_OF
    return dict(FAMILY_OF)


def build() -> dict:
    types = load(str(ASDL))
    kept, excluded = classify(types)
    labels, covered = _v1_labels(), covered_constructs()
    bt = builtin_tokens()

    from ..generate.bpool import applicable_subtypes

    entries = {}
    for construct, owning in sorted(kept.items()):
        token = NODE_TOKENS.get(construct)
        if construct in NO_TOKEN:
            token = None
        subs = applicable_subtypes(construct, token) if token else ()
        entries[construct] = {
            "member_kind": "ast-node", "asdl_type": owning,
            "token": token, "tokens": list(tokens_of(construct)) if token else [],
            "siblings": list(siblings(construct)) if token else [],
            "contrast_class": contrast_class(token),
            "frame_class": FRAME_CLASS.get(construct, "operand position"),
            "components": list(COMPONENTS.get(construct, ())),
            "b_subtypes": list(subs) or ["A-against-C-only"],
            "verifier": verifier_kind(construct, False),
            "a_family": covered.get(construct),
            "v1": labels.get(construct, {}),
            "tier": None,
        }
    for name, token in sorted(bt.items()):
        obj = getattr(_builtins, name)
        if obj is True or obj is False or obj is None:
            kind = "keyword-constant"
        elif isinstance(obj, type) and issubclass(obj, BaseException):
            kind = "exception"
        elif isinstance(obj, type):
            kind = "type"
        elif callable(obj):
            kind = "function"
        else:
            kind = "constant"
        entries[name] = {
            "member_kind": "builtin", "builtin_kind": kind, "token": token,
            "siblings": list(siblings(name)),
            "contrast_class": "named-builtin",
            "frame_class": "builtin reference", "components": [],
            "b_subtypes": list(applicable_subtypes(name, token)),
            "verifier": verifier_kind(name, True),
            "a_family": "builtin", "v1": labels.get(name, {}), "tier": None,
        }

    return {"python_version": "3.12.13",
            "asdl_sha256": _sha(ASDL),
            "counts": {"ast": sum(1 for e in entries.values()
                                  if e["member_kind"] == "ast-node"),
                       "builtin": sum(1 for e in entries.values()
                                      if e["member_kind"] == "builtin"),
                       "total": len(entries), "excluded": len(excluded)},
            "excluded": excluded,
            "verification": verification(entries, excluded),
            "constructs": entries}


def verification(entries: dict, excluded: dict) -> dict:
    """The two completeness tables the handout requires, plus coverage gaps."""
    kw_map: dict[str, list[str]] = {k: [] for k in keyword.kwlist}
    for c, e in entries.items():
        for t in e.get("tokens") or ([e["token"]] if e.get("token") else []):
            if t in kw_map:
                kw_map[t].append(c)
    unmapped = sorted(k for k, v in kw_map.items() if not v)

    public = {n for n in dir(_builtins) if not n.startswith("_")}
    accounted = {n for n, e in entries.items() if e["member_kind"] == "builtin"}
    from .tokens import SITE_ADDITIONS
    missing = sorted(public - accounted - set(SITE_ADDITIONS))

    no_family = sorted(c for c, e in entries.items()
                       if e["member_kind"] == "ast-node" and not e["a_family"])
    return {"keywords_total": len(keyword.kwlist),
            "keyword_map": {k: sorted(v) for k, v in sorted(kw_map.items())},
            "keywords_mapped": len(keyword.kwlist) - len(unmapped),
            "keywords_unmapped": unmapped,
            "builtins_unaccounted": missing,
            "ast_constructs_without_a_family": no_family}


def _sha(path: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inv = build()
    out = DATA / "inventory_v2.json"
    out.write_text(json.dumps(inv, indent=2, sort_keys=False))
    v = inv["verification"]
    print(f"inventory_v2.json: {inv['counts']['ast']} AST + "
          f"{inv['counts']['builtin']} builtins = {inv['counts']['total']}; "
          f"{inv['counts']['excluded']} excluded")
    print(f"keywords mapped: {v['keywords_mapped']}/{v['keywords_total']}"
          + (f"  UNMAPPED: {v['keywords_unmapped']}" if v["keywords_unmapped"] else ""))
    print(f"builtins unaccounted: {v['builtins_unaccounted'] or 'none'}")
    print(f"AST constructs with no A-side family ({len(v['ast_constructs_without_a_family'])}): "
          f"{', '.join(v['ast_constructs_without_a_family'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
