"""One name for a construct on the reading side: the tree's name against the contract's name.

**The defect this exists for.** `validation/battery/SPLIT.json` records P₁ under the *construct*
name -- `dict`, `list`, `set`, `slice`, `tuple` -- while the derived tree, the record and
`TWINS.json` carry those five under the *stem* `dict__builtin` and so on, because macOS folds
case and `dict_A` and `Dict_A` would be one file (`build_battery.set_stem`). A direct
`set(SPLIT["p1"]) & set(tree)` therefore drops exactly five constructs, silently: it gave
**|P₂ ∩ P₁| = 161** where the answer is **166**. 161 is also the value of a legitimate
sensitivity of the same quantity, so the two are confusable and the error does not announce
itself.

**The rule, derived and not listed.** A construct takes the suffix in the tree when it is a
builtin *and* another construct's name folds onto it under `str.lower`. That is `set_stem`'s rule,
read from `inventory_v2.json`, so the two sides cannot drift: if a future inventory adds a sixth
collision, both sides pick it up and `tree_names` covers it with no edit here. The five are
asserted in the tests as the value the rule currently takes, not as its definition.

S3's name resolution calls `tree_name` on anything coming from a contract artefact and
`contract_name` on anything coming from the tree. Nothing else should re-implement either.
"""
from __future__ import annotations
from profilebattery.paths import data_root

import functools
import json
import os
import pathlib

SUFFIX = "__builtin"
D = data_root()
INVENTORY = D / "inventory/inventory_v2.json"


@functools.lru_cache(maxsize=4)
def _inventory(path: str | None = None) -> dict:
    return json.loads(pathlib.Path(path or INVENTORY).read_text())


@functools.lru_cache(maxsize=4)
def suffixed(path: str | None = None) -> frozenset[str]:
    """The constructs whose tree name carries `__builtin`, derived from the inventory.

    `set_stem`'s rule: a builtin whose name folds onto another construct's name. Kept as a set
    rather than a function per call so that a caller can assert what the rule currently covers.
    """
    cons = _inventory(path)["constructs"]
    folded: dict[str, list[str]] = {}
    for name in cons:
        folded.setdefault(name.lower(), []).append(name)
    return frozenset(name for name, e in cons.items()
                     if e.get("member_kind") == "builtin" and len(folded[name.lower()]) > 1)


def tree_name(construct: str, path: str | None = None) -> str:
    """The name the tree, the record and `TWINS.json` use. Idempotent."""
    if construct.endswith(SUFFIX):
        return construct
    return f"{construct}{SUFFIX}" if construct in suffixed(path) else construct


def contract_name(name: str) -> str:
    """The name `SPLIT.json`, `SCOPE_SCHEDULE.json` and the plan use. Idempotent."""
    return name[: -len(SUFFIX)] if name.endswith(SUFFIX) else name


def tree_names(constructs, path: str | None = None) -> list[str]:
    """`tree_name` over an iterable, order preserved and duplicates kept."""
    return [tree_name(c, path) for c in constructs]


def resolve(constructs, available, path: str | None = None) -> tuple[list[str], list[str]]:
    """`(found, missing)` against a set of names the tree actually has.

    Takes either spelling on either side, so a caller cannot get the direction wrong. `missing`
    is returned rather than raised because a reading may legitimately run on part of a set -- but
    it is returned, not dropped, so a silent five-construct loss is impossible.
    """
    have = {tree_name(a, path) for a in available} | {contract_name(a) for a in available}
    found, missing = [], []
    for c in constructs:
        t = tree_name(c, path)
        (found if (t in have or contract_name(c) in have) else missing).append(t)
    return found, missing
