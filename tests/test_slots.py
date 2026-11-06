"""Item 96b: the slot table, the corrected forced set, and E10's floor.

The three things these tests exist to stop coming back:

  * the slot was the literal's own token for every role, so `For`'s `in` measured the entropy of
    a position that always holds `in` and read 0.0;
  * a header's `:` was declared forced, so its floor was written as zero when the position
    genuinely admits a continuation;
  * the floor was pooled over a slot that is several syntactic positions once the tokeniser
    merges the literal with the expression in front of it.
"""

import ast
import collections

import pytest

from profilebattery.generate.align import line_start_offsets
from profilebattery.generate.floors import (UNCLEAN_TOLERANCE, _is_clean,
                                            conditional_entropy, floors_for_set)
from profilebattery.generate.positions import (BARE_COLON_CONSTRUCTS, FORCED_ROLES,
                                               SLOT_ALTERNATIVES, SLOT_RULE,
                                               TWO_USE_OF, obliged_positions,
                                               positions_for)


def marks(text, kind):
    tree = ast.parse(text)
    node = next(n for n in ast.walk(tree) if type(n).__name__ == kind)
    return {p.role: p for p in obliged_positions(node, text, line_start_offsets(text))}


# --- the slot is where the grammar branches ----------------------------------------------

def test_for_in_slot_is_after_the_first_target_component():
    """The bug that made E10's floor read 0.0: the slot must sit where `,` is still legal."""
    simple = "for x in items:\n    pass\n"
    tuple_ = "for x, y in pairs:\n    pass\n"
    assert simple[marks(simple, "For")["in"].slot:][:3] == " in"
    assert tuple_[marks(tuple_, "For")["in"].slot:][:2] == ", "


def test_for_in_slot_is_not_the_literal():
    text = "for x in items:\n    pass\n"
    m = marks(text, "For")["in"]
    assert m.slot < m.char, "a slot equal to the literal's own position has no entropy"


def test_comprehension_in_slot_uses_the_first_generator():
    text = "r = [a for a, b in pairs]\n"
    assert text[marks(text, "ListComp")["in"].slot:][:2] == ", "


def test_comp_for_slot_is_the_literal_not_the_element_end():
    """Python's AST omits parentheses, so an element's span ends *inside* its closing paren.

    `elt_end`, tried first, put the slot on that paren. The slot is the literal's own character,
    which `locate` then maps to the piece `|for` -- the leading space included, so the branch
    against `]` and `,` is what the entropy is over.
    """
    text = "r = [(v * 2 if v else 0) for v in xs]\n"
    m = marks(text, "ListComp")["comp_for"]
    assert text[m.slot:][:3] == "for"
    assert text[m.slot - 1] == " ", "the piece carries the space, so the slot starts the word"


def test_dictcomp_element_ends_after_the_value():
    text = "d = {k: v for k, v in items}\n"
    assert text[marks(text, "DictComp")["comp_for"].slot:][:3] == "for"


@pytest.mark.parametrize("role", sorted(SLOT_RULE))
def test_every_role_has_a_rule_and_a_declared_alternative_set(role):
    assert role in SLOT_RULE
    if SLOT_RULE[role] is None:
        assert role in FORCED_ROLES, "only a forced role may have no slot"
    else:
        assert SLOT_ALTERNATIVES.get(role), f"{role} has a slot but no declared alternatives"


# --- forced is narrower than it looks ----------------------------------------------------

def test_header_colon_is_not_forced():
    for text, kind in (("if x > 0:\n    pass\n", "If"),
                       ("for x in y:\n    pass\n", "For"),
                       ("while x:\n    pass\n", "While"),
                       ("def f(a):\n    pass\n", "FunctionDef"),
                       ("class C:\n    pass\n", "ClassDef")):
        m = marks(text, kind)["header_colon"]
        assert m.forced is False, f"{kind}: an expression precedes the colon"
        assert m.floor_nats is None, f"{kind}: the floor is measured, not zero"


def test_bare_colons_are_forced():
    m = marks("try:\n    f()\nexcept E:\n    pass\n", "Try")
    assert m["bare_colon"].forced is True
    assert m["bare_colon"].floor_nats == 0.0


def test_except_with_a_type_is_expected_but_bare_except_is_forced():
    bare = marks("try:\n    f()\nexcept:\n    pass\n", "ExceptHandler")
    typed = marks("try:\n    f()\nexcept ValueError:\n    pass\n", "ExceptHandler")
    assert "bare_colon" in bare and bare["bare_colon"].forced is True
    assert "header_colon" in typed and typed["header_colon"].forced is False


def test_else_and_finally_colons_are_found_though_they_have_no_ast_node():
    roles = collections.Counter(
        p.role for p in obliged_positions(
            next(n for n in ast.walk(ast.parse(
                "try:\n    f()\nexcept E:\n    pass\nelse:\n    g()\nfinally:\n    h()\n"))
                 if isinstance(n, ast.Try)),
            "try:\n    f()\nexcept E:\n    pass\nelse:\n    g()\nfinally:\n    h()\n",
            line_start_offsets(
                "try:\n    f()\nexcept E:\n    pass\nelse:\n    g()\nfinally:\n    h()\n")))
    assert roles["bare_colon"] == 3, "the `try:`, the `else:` and the `finally:`"


def test_ifexp_orelse_is_an_expression_and_yields_no_bare_colon():
    """`IfExp` has an `orelse` too, and treating it as a clause list was a crash."""
    assert "bare_colon" not in marks("z = a if c else b\n", "IfExp")


def test_elif_has_no_colon_of_its_own():
    text = "if a:\n    f()\nelif b:\n    g()\n"
    assert "bare_colon" not in marks(text, "If")


# --- 10b: the closers and the two-use marks ----------------------------------------------

def test_closing_brackets_are_expected_and_carry_a_slot():
    for text, kind, role in (("v = f(a, b)\n", "Call", "close_paren"),
                             ("v = [a, b]\n", "List", "close_bracket"),
                             ("v = {a: b}\n", "Dict", "close_brace")):
        m = marks(text, kind)[role]
        assert m.forced is False and m.slot is not None


def test_ternary_literals_are_marked():
    m = marks("z = a if c else b\n", "IfExp")
    assert {"ternary_if", "ternary_else"} <= set(m)
    assert all(m[r].slot is not None for r in ("ternary_if", "ternary_else"))


def test_two_use_constructs_carry_their_use():
    assert marks("for x in y:\n    pass\n", "For")["in"].use == "For"
    assert marks("r = [a for a in b]\n", "ListComp")["comp_for"].use == "ListComp"
    assert marks("v = [a, b]\n", "List")["close_bracket"].use is None


def test_the_five_two_use_groups_are_the_ones_amendment_a10_licensed():
    assert TWO_USE_OF["For"] == "for" and TWO_USE_OF["ListComp"] == "for"
    assert TWO_USE_OF["If"] == "if" and TWO_USE_OF["IfExp"] == "if"
    for c in ("List", "Subscript", "Dict", "Set"):
        assert c not in TWO_USE_OF, "the symbol groups are not piece-matched (L5)"


# --- the floor, and when it is refused ----------------------------------------------------

class Rec:
    def __init__(self, text, positions):
        self.text, self.positions = text, positions


def _pieces(text):
    """A stand-in for the tokeniser: chunks that each begin with their leading space.

    Enough to exercise the floor, because the one property that matters is the one the real
    tokeniser has -- a piece carries the space in front of the word, so the slot's piece is
    ` in` or `,` and not the bare character the slot's offset points at.
    """
    out, cur = [], ""
    for ch in text:
        if ch == " " and cur:
            out.append(cur)
            cur = " "
        elif ch.isalnum() or ch == "_":
            cur += ch
        else:
            if cur:
                out.append(cur)
            cur = ""
            out.append(ch)
    if cur:
        out.append(cur)
    return out


def _recs(texts, kind):
    """Records as the pipeline makes them: through `positions_for`, so `locate` has run."""
    out = []
    for t in texts:
        pieces = _pieces(t)
        spans, off = [], 0
        for pc in pieces:
            spans.append((off, off + len(pc)))
            off += len(pc)
        ct = next(i for i, pc in enumerate(pieces) if pc.strip() == kw_of(kind))
        got = positions_for(t, kind, pieces, ct)
        out.append(Rec(t, [p.as_dict() for p in got or []]))
    return out


def kw_of(kind):
    return {"For": "for", "If": "if", "While": "while"}[kind]


def test_floor_is_measured_where_the_set_branches():
    texts = ["for x in a:\n    pass\n"] * 3 + ["for x, y in a:\n    pass\n"]
    floors = floors_for_set(_recs(texts, "For"), _pieces)
    assert floors["in"] == pytest.approx(conditional_entropy(
        collections.Counter({" in": 3, ",": 1})), abs=1e-9)


def test_floor_is_refused_once_the_bundled_share_passes_the_tolerance():
    """The pieces are given directly: bundling is the real tokeniser's doing, not the grammar's.

    `for x in f(a):` arrives as `):\n` at the colon's slot, carrying the end of the iterable into
    the piece, so the slot is one position in that prompt and another in `for x in a:` -- two
    positions, pooled. The measured shares on the pilot battery are 0.75 for `For` and 1.00 for
    `FunctionDef`, against 0.0 wherever the slot is a single position, so the tolerance decides
    nothing that the data leaves close.
    """
    def floor_with(bundled, total):
        pieces = {}
        rows = []
        for i in range(total):
            text = f"prompt_{i}"
            pieces[text] = ["):\n" if i < bundled else ":\n"]
            rows.append(Rec(text, [{"class": "obliged", "role": "header_colon", "slot": 0}]))
        return floors_for_set(rows, pieces.__getitem__)["header_colon"]

    assert floor_with(1, 1000) is not None, "one unclean piece in a thousand is residue"
    assert floor_with(250, 1000) is None, "a quarter unclean is a slot pooling two positions"
    # and with nothing bundled the floor is the entropy of the branch itself
    assert floor_with(0, 1000) == pytest.approx(0.0)
