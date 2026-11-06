"""The tree-name / contract-name rule, and the five-construct loss it exists to prevent."""
import json
import pathlib

import pytest

from profilebattery.profiles import names
from profilebattery.paths import data_root

REAL = data_root() / "inventory/inventory_v2.json"
real_only = pytest.mark.skipif(not REAL.exists(), reason="the real inventory is not on this host")

FAKE = {"constructs": {
    "Dict": {"member_kind": "ast-node"}, "dict": {"member_kind": "builtin"},
    "Set": {"member_kind": "ast-node"}, "set": {"member_kind": "builtin"},
    "For": {"member_kind": "ast-node"}, "len": {"member_kind": "builtin"},
}}


@pytest.fixture
def fake(tmp_path):
    p = tmp_path / "inv.json"
    p.write_text(json.dumps(FAKE))
    names.suffixed.cache_clear()
    names._inventory.cache_clear()
    yield str(p)
    names.suffixed.cache_clear()
    names._inventory.cache_clear()


def test_the_rule_is_builtin_and_case_colliding(fake):
    assert names.suffixed(fake) == {"dict", "set"}
    # a builtin with no collision keeps its name; an ast node never takes the suffix
    assert names.tree_name("len", fake) == "len"
    assert names.tree_name("For", fake) == "For"
    assert names.tree_name("Dict", fake) == "Dict"
    assert names.tree_name("dict", fake) == "dict__builtin"


def test_both_directions_are_idempotent(fake):
    assert names.tree_name(names.tree_name("dict", fake), fake) == "dict__builtin"
    assert names.contract_name(names.contract_name("dict__builtin")) == "dict"
    assert names.contract_name("dict") == "dict"
    assert names.tree_name("dict__builtin", fake) == "dict__builtin"


def test_resolve_accepts_either_spelling_on_either_side(fake):
    tree = ["dict__builtin", "Dict", "For"]
    found, missing = names.resolve(["dict", "Dict", "For"], tree, fake)
    assert found == ["dict__builtin", "Dict", "For"] and missing == []
    # and the same when the caller already has tree names
    found, missing = names.resolve(["dict__builtin"], tree, fake)
    assert found == ["dict__builtin"] and missing == []


def test_a_genuinely_absent_construct_is_returned_not_dropped(fake):
    found, missing = names.resolve(["dict", "Nonlocal"], ["dict__builtin"], fake)
    assert found == ["dict__builtin"]
    assert missing == ["Nonlocal"]           # returned, so a loss cannot be silent


@real_only
def test_on_the_real_inventory_the_rule_covers_exactly_the_five():
    names.suffixed.cache_clear()
    names._inventory.cache_clear()
    assert names.suffixed() == {"dict", "list", "set", "slice", "tuple"}


@real_only
def test_P1_against_the_tree_is_169_with_the_rule_and_164_without_it():
    """The regression this module exists for, on the real artefacts."""
    D = data_root()
    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    inputs = D / "validation/p2/P2_INPUTS.json"
    if not inputs.exists():
        pytest.skip("P2_INPUTS.json is not on this host")
    tree = set(json.loads(inputs.read_text()))
    p1 = split["p1"]
    assert len(p1) == 169
    # without the rule, five P1 members do not match the tree
    assert len(set(p1) & tree) == 164
    # with it, all 169 resolve and nothing is missing
    found, missing = names.resolve(p1, tree)
    assert missing == []
    assert len(set(found) & tree) == 169
