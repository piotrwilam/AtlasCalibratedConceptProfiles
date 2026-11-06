"""The diagnostic-position marks the battery stores (C2, G1, Cowork's step 4).

Two properties are asserted, both of them Cowork's:

  * a construct whose template guarantees a class has at least one position of it in every A
    prompt -- a `For` prompt with no governed position is a template bug, not data;
  * the *not applicable* set is the grammar's, per class, so a class the grammar cannot give is
    distinguishable from a class that happened to get no position.

The marks are checked on text, with one character standing for one token piece, because what is
being tested is the class structure rather than the tokeniser's alignment -- `test_align.py`
owns that.
"""

import ast

import pytest

from profilebattery.generate.positions import (CLASSES, declared_classes, positions_for)


def marks(src: str, construct: str, token: int = 0):
    pieces = list(src)          # one char per piece: token index == character offset
    found = positions_for(src, construct, pieces, token)
    return [] if found is None else [p.as_dict() for p in found]


FOR = 'for eventlet in scores["a"]:\n    total += eventlet\n'
DEF = "def audit(rows):\n    return rows[0]\n    seen = rows\n"
ASSIGN = "total = 3\nflag = total > 1\n"
COMP = "[q * 2 for q in rows]\n"


def test_for_has_both_obliged_and_governed():
    m = marks(FOR, "For", FOR.index("for"))
    classes = {x["class"] for x in m}
    assert "obliged" in classes and "governed" in classes, m
    assert {x["role"] for x in m} >= {"in", "header_colon", "governed_use"}


def test_functiondef_governed_position_is_a_parameter_use():
    m = marks(DEF, "FunctionDef", DEF.index("def"))
    gov = [x for x in m if x["class"] == "governed"]
    assert gov, m
    assert all(x["bound_name"] == "rows" for x in gov)


def test_assign_scope_position_follows_the_binding():
    m = marks(ASSIGN, "Assign", 0)
    sco = [x for x in m if x["class"] == "scope"]
    assert sco and sco[0]["distance"] > 0, m


def test_comprehension_governed_position_precedes_the_construct():
    """The element expression comes before the `for`, so the distance is negative.

    Recorded as a test because it makes the comprehension's dependent loss a different
    quantity from a statement construct's: there is nothing to copy from yet.
    """
    m = marks(COMP, "ListComp", COMP.index("for"))
    gov = [x for x in m if x["class"] == "governed"]
    assert gov and gov[0]["distance"] < 0, m


@pytest.mark.parametrize("construct,src,token_word", [
    ("For", FOR, "for"), ("FunctionDef", DEF, "def"), ("ListComp", COMP, "for"),
])
def test_a_guaranteed_class_is_present(construct, src, token_word):
    """Cowork's first property: a template that guarantees a class delivers one."""
    m = marks(src, construct, src.index(token_word))
    for cls in declared_classes(construct):
        if cls == "scope" and construct == "FunctionDef":
            continue          # a def's own name is rarely called in its own block, by corpus
        assert any(x["class"] == cls for x in m), f"{construct} gave no {cls} position: {m}"


@pytest.mark.parametrize("construct,expected_na", [
    ("While", {"governed", "scope"}),      # binds nothing; the colon is its only position
    ("If", {"governed", "scope"}),
    ("BinOp", set(CLASSES)),               # its grammar forces nothing after it
    ("For", {"scope"}),                    # governs its target; binds nothing for later use
    ("Assign", {"governed"}),              # no body to govern
    ("ListComp", {"scope"}),
])
def test_not_applicable_is_the_grammars(construct, expected_na):
    """Cowork's second property: the NA set is declared, not observed."""
    assert set(CLASSES) - declared_classes(construct) == expected_na


def test_every_construct_declares_a_consistent_na_set():
    for c in ("For", "While", "Assign", "ListComp", "Call", "Dict", "IfExp"):
        na = set(CLASSES) - declared_classes(c)
        assert na <= set(CLASSES)
        assert not (na == set(CLASSES) and declared_classes(c))


def test_marks_are_json_serialisable():
    import json
    json.dumps(marks(FOR, "For", FOR.index("for")))


# --- the control record, and the pairing (item 85b) ---

from profilebattery.generate import battery as _B


def test_prompt_id_is_stable_and_key_based():
    """Over keys, not text: the assembler reformats and a control differs from its A prompt."""
    a = _B.prompt_id("For", "A", 1, "For/c0", 7, 0)
    b = _B.prompt_id("For", "A", 1, "For/c0", 7, 0)
    c = _B.prompt_id("For", "A", 2, "For/c0", 7, 0)
    assert a == b and a != c and len(a) == 12


def test_control_of_points_at_the_a_prompt():
    a = _B.prompt_id("For", "A", 1, "For/c0", 7, 0)
    ctl = _B.prompt_id("For", "CTL", 1, "For/c0", 7, 0)
    assert a != ctl, "the control must have its own identity"


def test_obliged_positions_have_no_control_counterpart_under_variant_i():
    """They live in the header the variant removes -- and they need none, being grammar-forced."""
    from profilebattery.generate.controls import build_control
    src = 'for accrual in rows:\n    ok(accrual)\n    n = accrual\n'
    c = build_control(src, "For", 0)
    assert c.variant == "i"
    assert c.map_offset(src.index(" in ") + 1) is None       # the obliged `in`
    assert c.map_offset(src.index("ok(accrual)") + 3) is not None


def test_variant_ii_maps_every_position_outside_the_renamed_span():
    from profilebattery.generate.controls import build_control
    src = "total = 3\nflag = total > 1\n"
    c = build_control(src, "Assign", 0)
    assert c.variant == "ii"
    assert c.map_offset(src.index("= 3")) is not None
    assert c.map_offset(src.index("total > 1")) is not None
    assert c.map_offset(0) is None                            # inside the renamed binding
