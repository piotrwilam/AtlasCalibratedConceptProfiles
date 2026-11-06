"""The B prompt's anchor, and the provenance of the battery (step 7, G6/G7).

Two defects measured on the pilot battery and fixed here.

**The B anchor was a substring match.** `verify_b` located the construct's token with
`str.find`, so `for` matched inside `form`, `force` and `formula`, `and` inside `band` and
`outstanding`, `not` inside `annotations` and `notes`, `with` inside `_without`. **52 B prompts
across the keyword-token sets** were anchored on a token that merely contains the letters, and
their profiles were therefore profiles of that token (exact, measured over 21,600 B prompts --
0.24%, which adds noise to one side of a difference rather than bias, so the pilot's figures
stand; it is fixed for the phase battery).

**And the fix had to be restricted.** Requiring the anchored *piece* to equal the token is right
for a keyword and wrong for a symbol: BPE merges `[` into `("[` and `:` into `":` or `):\\n`, so
"the piece is the token" is almost never true for punctuation, and requiring it re-anchored
2,191 B prompts away from the piece that actually contains the construct's character -- which is
`align.py`'s declared policy. So the piece-exact rule applies to keyword tokens only.
"""

from profilebattery.generate.assemble import _piece_exact_positions


def test_no_tokeniser_keeps_the_substring_path():
    assert _piece_exact_positions("for x in y:", "for", None) is None


def test_symbol_tokens_are_exempt():
    """BPE merges punctuation, so a piece-exact rule would move the anchor, not fix it."""
    class FakeTok:                              # never called, the exemption comes first
        def __call__(self, *a, **k):
            raise AssertionError("a symbol token must not reach the tokeniser path")
    for sym in ("[", ":", ",", "{", "*", "="):
        assert _piece_exact_positions("a[0]", sym, FakeTok()) is None


def test_keyword_tokens_use_the_piece_path():
    class Tok:
        def __call__(self, text, **k):
            return {"input_ids": list(range(len(text.split())))}

        def convert_ids_to_tokens(self, i):
            return ["▁for", "▁formula"][i % 2]
    got = _piece_exact_positions("for formula", "for", Tok())
    assert got is not None and len(got) == 1, got


def test_the_hash_is_written_where_the_prompts_are():
    """A battery that can exist without its hash is a battery whose hash identifies nothing."""
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "experiments"
           / "build_battery.py").read_text()
    assert "CONTENT_HASHES.json" in src
    assert "def content_hash" in src


def test_own_core_twins_use_different_seeds():
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "experiments"
           / "build_battery.py").read_text()
    assert "CORE_SEED = {1: 11, 2: 23}" in src, "the twins must draw their own cores"


def test_seeding_is_keyed_and_not_the_builtin_hash():
    """`hash()` is salted per process, so a battery seeded with it cannot be regenerated."""
    import inspect

    from profilebattery.generate import seeds
    src = inspect.getsource(seeds.stable_seed)
    assert "blake2b" in src and "hash(" not in src
