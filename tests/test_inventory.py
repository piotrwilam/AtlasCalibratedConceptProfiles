"""Tests for §1 — the inventory derived from CPython's abstract grammar.

The point of deriving rather than hand-picking is that "why these constructs?" has a
mechanical answer, so these tests check the derivation, not a stored list: predicate R
against the pinned ASDL, the wrapper rule against `ast.unparse`, and the two
completeness tables the handout requires.
"""

import ast
import json
import os
import pathlib

import keyword
import pytest

from profilebattery.inventory.asdl import load, parse
from profilebattery.inventory.build import ASDL, build, contrast_class, verifier_kind
from profilebattery.inventory.predicate_r import (CONTAINER_TYPES, CONTEXT_TYPES,
                                                  OPERATOR_TYPES, WRAPPER_CONSTRUCTORS,
                                                  classify)
from profilebattery.inventory.tokens import NO_TOKEN, siblings, tokens_of

PINNED_SHA = "ecc26bc2818ebddec685f96663c811de8915beaa74a7cb1ace6eea3920679ac7"


@pytest.fixture(scope="module")
def types():
    if not ASDL.exists():
        pytest.skip("pinned ASDL not downloaded")
    return load(str(ASDL))


@pytest.fixture(scope="module")
def inventory(types):
    return build()


def test_asdl_is_the_pinned_file():
    import hashlib
    if not ASDL.exists():
        pytest.skip("pinned ASDL not downloaded")
    assert hashlib.sha256(ASDL.read_bytes()).hexdigest() == PINNED_SHA


def test_asdl_parser_handles_sums_and_products():
    t = parse("""
    module Python
    {
        mod = Module(stmt* body) | Interactive(stmt* body)
        arguments = (arg* args, arg? vararg)
        boolop = And | Or
        stmt = Pass | If(expr test, stmt* body)
               attributes (int lineno)
    }
    """)
    assert t["mod"].kind == "sum" and t["mod"].constructors == ["Module", "Interactive"]
    assert t["arguments"].kind == "product" and t["arguments"].constructors == []
    assert t["boolop"].constructors == ["And", "Or"]
    assert t["stmt"].constructors == ["Pass", "If"]


# --- predicate R ----------------------------------------------------------------

def test_r_keeps_sixty_three(types):
    kept, excluded = classify(types)
    assert len(kept) == 63, sorted(kept)
    assert len(kept) + 143 == 206


def test_r_excludes_each_declared_category(types):
    _, excluded = classify(types)
    for marker in ("Load", "Store", "Del"):
        assert "context marker" in excluded[marker]
    for marker, folded in (("Add", "BinOp"), ("And", "BoolOp"), ("Not", "UnaryOp"),
                           ("Eq", "Compare")):
        assert folded in excluded[marker]
    for container in ("Module", "Interactive", "Expression", "FunctionType"):
        assert "module container" in excluded[container]
    for helper in ("arguments", "arg", "keyword", "alias", "withitem", "comprehension",
                   "match_case"):
        assert "product type" in excluded[helper]
    assert "helper" in excluded["ExceptHandler"]


@pytest.mark.parametrize("src,node_type", [
    ("obj.method()", "Expr"),
    ("match x:\n case 1: pass", "MatchValue"),
    ("match x:\n case None: pass", "MatchSingleton"),
])
def test_wrapper_rule_is_verified_not_asserted(src, node_type):
    """A wrapper unparses to exactly its child, so it has no syntax of its own."""
    assert node_type in WRAPPER_CONSTRUCTORS
    node = next(n for n in ast.walk(ast.parse(src))
                if type(n).__name__ == node_type)
    child = node.value
    # MatchSingleton holds a raw constant rather than a node, so the wrapper claim is
    # "unparses to the literal" rather than "unparses to its child node"
    expected = ast.unparse(child) if isinstance(child, ast.AST) else repr(child)
    assert ast.unparse(node) == expected


def test_non_wrappers_contribute_tokens():
    """The rule must not over-reach: Yield has one field and still carries a token."""
    node = next(n for n in ast.walk(ast.parse("def f():\n yield 1"))
                if isinstance(n, ast.Yield))
    assert ast.unparse(node) != ast.unparse(node.value)
    assert "Yield" not in WRAPPER_CONSTRUCTORS


# --- completeness tables --------------------------------------------------------

def test_every_keyword_maps_to_a_construct(inventory):
    v = inventory["verification"]
    assert v["keywords_total"] == len(keyword.kwlist) == 35
    assert v["keywords_unmapped"] == [], v["keywords_unmapped"]


def test_every_public_builtin_is_accounted_for(inventory):
    assert inventory["verification"]["builtins_unaccounted"] == []


def test_counts_match_the_handout(inventory):
    assert inventory["counts"] == {"ast": 63, "builtin": 143, "total": 206,
                                   "excluded": 48}


# --- per-construct attributes ---------------------------------------------------

def test_every_construct_has_the_declared_attributes(inventory):
    for name, e in inventory["constructs"].items():
        assert e["contrast_class"] in ("keyword-token", "named-builtin", "structural")
        assert e["verifier"] in ("ast-node", "name", "keyword-constant")
        assert e["frame_class"]
        assert e["b_subtypes"], name


def test_tokenless_constructs_fall_back_to_a_against_c(inventory):
    for name in NO_TOKEN:
        e = inventory["constructs"][name]
        assert e["token"] is None
        assert e["b_subtypes"] == ["A-against-C-only"]
        assert e["contrast_class"] == "structural"


def test_contrast_class_follows_the_token_kind():
    assert contrast_class("for") == "keyword-token"
    assert contrast_class("match") == "keyword-token"       # soft keyword
    assert contrast_class("len") == "named-builtin"
    assert contrast_class("[") == "structural"
    assert contrast_class(None) == "structural"


def test_verifier_kind_splits_the_three_member_kinds():
    assert verifier_kind("For", False) == "ast-node"
    assert verifier_kind("len", True) == "name"
    assert verifier_kind("None", True) == "keyword-constant"


def test_tier_is_absent_until_calibration(inventory):
    assert all(e["tier"] is None for e in inventory["constructs"].values())


def test_secondary_tokens_do_not_displace_the_primary():
    assert tokens_of("If")[0] == "if" and "elif" in tokens_of("If")
    assert tokens_of("Compare")[0] == "==" and "is" in tokens_of("Compare")


def test_family_coverage_gap_is_recorded(inventory):
    """The gap is a recorded field, and it shrinks as families are built.

    21 at first; the three async compound statements were added because the No-Later
    composite class contains them and narrowing the claim to "except the async ones"
    would read as convenience rather than rule.
    """
    gap = inventory["verification"]["ast_constructs_without_a_family"]
    assert len(gap) == 18
    assert {"Await", "TryStar", "TypeAlias", "Name", "Constant"} <= set(gap)
    assert not {"AsyncFor", "AsyncWith", "AsyncFunctionDef"} & set(gap)
    covered = [c for c, e in inventory["constructs"].items()
               if e["member_kind"] == "ast-node" and e["a_family"]]
    assert len(covered) == 45
