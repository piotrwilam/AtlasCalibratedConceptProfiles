"""Tests for the calibrated menu — the thing that lets a run stamp the licence.

The handout's rule is that nothing generated on the stub menu counts, so the two
properties that matter most here are provenance (a caller can always tell which slots are
corpus-backed) and the builtin filter (a name that collides with a builtin corrupts that
builtin's own A and B sets).
"""

import ast
import builtins as _builtins
import keyword

import numpy as np
import pytest

from profilebattery.generate.calibrated_menu import (CAL, MIN_USABLE, CalibratedMenu,
                                                     default_menu)
from profilebattery.generate.menu import DOMAINS, StubMenu

pytestmark = pytest.mark.skipif(not (CAL / "identifiers.json").exists(),
                                reason="calibration tables not built")


@pytest.fixture(scope="module")
def menu():
    return CalibratedMenu()


def test_the_stub_is_provisional_and_the_calibrated_one_is_not(menu):
    assert StubMenu.PROVISIONAL is True
    assert menu.PROVISIONAL is False
    assert type(default_menu()).__name__ == "CalibratedMenu"


def test_provenance_names_every_slot_and_domain(menu):
    p = menu.provenance()
    assert p["provisional"] is False
    for slot in ("plural", "singular", "field", "verb", "class", "int_literal",
                 "float_literal", "text", "padding"):
        assert set(p["sources"][slot]) == set(DOMAINS), slot
    # the slots the corpus genuinely could not fill say so rather than pretending
    assert all("synthesised" in v for v in p["sources"]["float_literal"].values())
    assert all("composed" in v for v in p["sources"]["text"].values())
    assert p["calibration_manifest"], "the table hashes travel with the menu"


def test_no_drawn_name_collides_with_a_builtin_or_keyword(menu):
    """`list = 5` in a core would both shadow the builtin and corrupt `list`'s own sets."""
    reserved = set(dir(_builtins)) | set(keyword.kwlist)
    rng = np.random.default_rng(0)
    for dom in DOMAINS:
        for _ in range(200):
            for slot in ("plural", "singular", "total", "class"):
                assert menu.name(dom, slot, rng) not in reserved
            assert menu.verb(dom, rng) not in reserved
            assert menu.field(dom, rng) not in reserved


def test_drawn_names_are_identifiers(menu):
    rng = np.random.default_rng(1)
    for dom in DOMAINS:
        for _ in range(100):
            for slot in ("plural", "singular", "total", "class"):
                assert menu.name(dom, slot, rng).isidentifier()


def test_padding_blocks_are_whole_and_parse(menu):
    """Slicing a block can cut a balanced triple-quoted string in half.

    That produced every one of the 39 invalid prompts in the first calibrated run: the
    fragment swallowed the core. Blocks are taken whole, so the budget may overshoot.
    """
    rng = np.random.default_rng(2)
    for dom in DOMAINS:
        for n in (3, 6):
            for _ in range(40):
                block = menu.padding(dom, n, rng)
                ast.parse("\n".join(block))        # raises if a string was cut


def test_literals_come_from_the_corpus_distribution(menu):
    rng = np.random.default_rng(3)
    ints = [int(menu.int_literal(rng)) for _ in range(400)]
    assert min(ints) >= 0
    assert sum(1 for i in ints if i <= 10) > 200, "corpus ints are small and head-heavy"
    floats = [float(menu.float_literal(rng)) for _ in range(200)]
    assert max(floats) < 20, "synthesised floats are kept small deliberately"


def test_thin_slots_fall_back_rather_than_ship_four_words(menu):
    """A slot with fewer than MIN_USABLE corpus values uses the declared words."""
    thin = [(slot, dom) for slot, per in menu.sources.items()
            for dom, src in per.items() if src.startswith("declared")]
    assert thin, "class names and three domains' verbs are known to be thin"
    for slot, dom in thin:
        assert not menu.tables.get(slot, {}).get(dom)
        rng = np.random.default_rng(4)
        got = menu.verb(dom, rng) if slot == "verb" else menu.name(dom, slot, rng)
        assert got and got.isidentifier()
