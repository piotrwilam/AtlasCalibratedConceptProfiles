"""The matched control set's template table (G4, Cowork's step 5).

The table is asserted, not computed: a control is a decision about a construct's grammar, and a
construct that acquires a dependent class without acquiring a control decision must fail here
rather than quietly go uncontrolled.
"""

from profilebattery.generate import controls
from profilebattery.generate.positions import GOVERNED_BINDERS, SCOPE_BINDERS


def test_every_construct_with_a_dependent_class_is_decided():
    a = controls.audit()
    assert a["undecided"] == [], (
        f"these constructs have a dependent class and no control decision: {a['undecided']}")


def test_variants_match_the_class_they_serve():
    """Variant (i) removes the construct, so it can only serve a governed use; the scope family
    already *is* an assignment, so (i) would be a no-op and only (ii) says anything."""
    for t in controls.TEMPLATES:
        if t.serves == "scope":
            assert t.variant == "ii", f"{t.construct}: variant (i) is a no-op for the scope class"
        else:
            assert t.variant in ("i", "ii")


def test_the_comprehensions_are_excused_for_the_right_reason():
    for c in ("ListComp", "SetComp", "GeneratorExp", "DictComp"):
        why = controls.reason_not_needed(c)
        assert why and ("precedes" in why or "as ListComp" in why), c


def test_control_count_is_small_enough_to_be_cheap():
    """The compute estimate rests on the number of control sets, so pin it.

    The anchored pass covers an A set and a B set for each of the battery's 206 constructs, so
    412 sets. Eleven control sets, each A-sized and loss-only, is **2.7% more forward passes**
    (arithmetic) -- not the third that a control for every set would have cost, and not the
    8-17% estimated when the count was thought to be five or ten against the pilot's sixty.
    The bound is here to catch the control set silently growing to cover everything.
    """
    n = controls.audit()["n_control_sets"]
    assert n <= 12, f"{n} control sets: the compute estimate needs redoing"
    assert n / (206 * 2) < 0.05


def test_every_template_changes_only_the_header():
    """Both variants leave the body alone -- that is what makes the positions matchable."""
    for t in controls.TEMPLATES:
        assert t.a_header and t.control_header
        assert t.a_header != t.control_header
        assert t.note, f"{t.construct} has no note saying why its variant is the one it is"


def test_governed_and_scope_binders_are_disjoint_except_functiondef():
    """`FunctionDef` is the one construct in both: its parameters are governed, its own name is
    a scope binding. Its control serves the governed class, the one it actually has positions
    for -- a def's own name is rarely called in its own block, by corpus."""
    both = GOVERNED_BINDERS & SCOPE_BINDERS
    assert both == {"FunctionDef", "AsyncFunctionDef"}
    assert controls.template_for("FunctionDef").serves == "governed"


# --- the transform (item 85) ---

import ast

from profilebattery.generate.controls import build_control

FOR_PLAIN = 'for accrual in rows_by_key["book"]:\n    audit_ok(accrual)\n    n = accrual\n'
FOR_JUMP = "for accrual in scores:\n    if accrual < 3:\n        continue\n    ok(accrual)\n"
WITH = "with open(path) as handle:\n    total += handle\n"
DEF = "def audit(rows):\n    return rows[0]\n    seen = rows\n"
ASSIGN = "total = 3\nflag = total > 1\n"
IMPORTFROM = "from os import path\nroot = path.join('a')\n"


def _built(src, construct, word):
    return build_control(src, construct, src.index(word))


def test_every_control_parses():
    for src, c, w in ((FOR_PLAIN, "For", "for"), (FOR_JUMP, "For", "for"),
                      (WITH, "With", "with"), (DEF, "FunctionDef", "def"),
                      (ASSIGN, "Assign", "total"), (IMPORTFROM, "ImportFrom", "from")):
        got = _built(src, c, w)
        assert got is not None, c
        ast.parse(got.text)


def test_variant_i_removes_the_construct_and_keeps_the_name():
    got = _built(FOR_PLAIN, "For", "for")
    assert got.variant == "i" and not got.fell_back
    assert "for accrual in" not in got.text
    assert got.bound_name == "accrual" and "accrual = next(iter(" in got.text
    assert got.text.count("accrual") == FOR_PLAIN.count("accrual")


def test_variant_i_falls_back_when_the_body_cannot_be_dedented():
    """`continue` outside a loop is a syntax error, not implausible code."""
    got = _built(FOR_JUMP, "For", "for")
    assert got.variant == "ii" and got.fell_back
    assert "Continue" in got.reason
    assert "for " in got.text          # the loop is kept, so `continue` stays legal


def test_variant_ii_leaves_the_use_free():
    got = _built(ASSIGN, "Assign", "total")
    assert got.variant == "ii" and got.substitute
    assert f"{got.substitute} = 3" in got.text
    assert "flag = total > 1" in got.text      # the use is untouched, and now unbound


def test_a_substitute_shares_no_prefix_with_the_original():
    """`total_alt` would tokenise with `total`'s pieces, so copying would still half-work."""
    got = _built(ASSIGN, "Assign", "total")
    assert not got.substitute.startswith(got.bound_name[:3])
    assert not got.bound_name.startswith(got.substitute[:3])


def test_functiondef_takes_variant_ii_because_dedenting_a_return_is_a_syntax_error():
    got = _built(DEF, "FunctionDef", "def")
    assert got.variant == "ii" and not got.fell_back
    assert got.text.startswith("def audit(")
    assert "return rows[0]" in got.text        # the use is free, the body untouched


def test_imports_rename_through_as_so_the_attribute_stays_real():
    got = _built(IMPORTFROM, "ImportFrom", "from")
    assert got.variant == "ii"
    assert f"from os import path as {got.substitute}" in got.text
    assert "root = path.join" in got.text
    ast.parse(got.text)


def test_the_transform_is_deterministic():
    a = _built(ASSIGN, "Assign", "total")
    b = _built(ASSIGN, "Assign", "total")
    assert a.text == b.text and a.substitute == b.substitute


def test_no_control_for_an_excused_construct():
    assert build_control("[q for q in rows]\n", "ListComp", 7) is None


def test_everything_outside_the_header_is_byte_identical_under_variant_ii():
    got = _built(ASSIGN, "Assign", "total")
    assert got.text.split("\n")[1:] == ASSIGN.split("\n")[1:]
