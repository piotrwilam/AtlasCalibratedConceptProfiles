"""Tests for §5 — the yardstick and the profile."""

import numpy as np
import pytest

from profilebattery.profiles.cells import cell_key
from profilebattery.profiles.profile import Profile, build, concept, core_bootstrap, \
    cell_of, pooled
from profilebattery.profiles.yardstick import DEFAULT_EPS, Cell, Yardstick, standardise
from profilebattery.paths import data_root


def make_yardstick(width=8, layers=2, sd=1.0):
    """Every cell present, so a prompt is never skipped for a missing yardstick."""
    from profilebattery.profiles.cells import all_cells

    cells = {}
    for l in range(layers):
        for c in all_cells():
            cells[(l, c)] = Cell(n=500, mean=np.zeros(width),
                                 var=np.full(width, sd ** 2),
                                 active=np.full(width, 500))
    return Yardstick(cells)


def test_standardisation_is_per_cell():
    ys = make_yardstick()
    ys.cells[(0, "b1:keyword")] = Cell(n=500, mean=np.full(8, 5.0),
                                       var=np.full(8, 4.0), active=np.full(8, 500))
    x = np.full(8, 5.0)
    assert np.allclose(standardise(x, ys, 0, "b0:keyword"), 5.0, atol=1e-3)
    assert np.allclose(standardise(x, ys, 0, "b1:keyword"), 0.0, atol=1e-3)


def test_the_floor_is_smooth_not_a_clip():
    """A hard clip makes an open parameter decide the magnitude of the selective units."""
    cells = {(0, "b0:keyword"): Cell(n=100, mean=np.zeros(4),
                                     var=np.array([1.0, 1.0, 1.0, 1e-8]),
                                     active=np.full(4, 100))}
    ys = Yardstick(cells, eps=0.1)
    sig = ys.sigma(0, "b0:keyword")
    assert sig[3] > 1e-4, "a near-zero sd must be lifted"
    assert sig[0] > 1.0, "and every unit is lifted a little — no cliff"
    assert np.all(np.diff(np.sort(sig)) >= 0)


def test_thin_support_is_per_unit_and_cell():
    cells = {(0, "b0:keyword"): Cell(n=1000, mean=np.zeros(3), var=np.ones(3),
                                     active=np.array([1000, 5, 400]))}
    ys = Yardstick(cells)
    assert list(ys.thin_units(0, "b0:keyword", min_active=100)) == [False, True, False]


def test_cell_of_uses_only_stored_fields():
    row = {"site_construct_token": 3, "n_tokens": 10, "anchor_piece": "for"}
    assert cell_of(row) == cell_key(1, "keyword")


def test_profile_standardises_each_prompt_against_its_own_cell():
    ys = make_yardstick(width=4, layers=2)
    rows = [{"site_construct_token": 0, "n_tokens": 10, "anchor_piece": "for",
             "core_id": "c0"},
            {"site_construct_token": 6, "n_tokens": 10, "anchor_piece": "for",
             "core_id": "c1"}]
    primary = np.stack([np.full((2, 4), 2.0), np.full((2, 4), 4.0)])
    p = build(primary, rows, ys, "For", "A")
    assert p.n == 2
    assert set(p.cells) == {cell_key(0, "keyword"), cell_key(3, "keyword")}
    # the smooth floor lifts every sigma0 a little, so every z shrinks by the same
    # factor — sqrt(1 + eps^2) at unit sd. That is the floor working, not an error.
    assert np.allclose(p.mean, 3.0, rtol=2 * DEFAULT_EPS ** 2)
    assert np.all(p.mean < 3.0)
    assert np.allclose(p.raw_mean, 3.0), "raw means travel unstandardised, unfloored"
    assert set(p.core_means) == {"c0", "c1"}


def test_bootstrap_resamples_cores_not_prompts():
    ys = make_yardstick(width=4, layers=1)
    rng = np.random.default_rng(0)
    rows, prim = [], []
    for c in range(4):
        for _ in range(25):
            rows.append({"site_construct_token": 0, "n_tokens": 10,
                         "anchor_piece": "for", "core_id": f"c{c}"})
            prim.append(np.full((1, 4), c) + rng.normal(scale=0.01, size=(1, 4)))
    p = build(np.stack(prim), rows, ys, "For", "A")
    assert len(p.core_means) == 4
    boot = core_bootstrap(p, n_boot=400, seed=1)
    # Cores differ hugely, prompts within a core barely. The per-prompt SE cannot see
    # that — it treats 100 prompts as 100 independent draws — so the core bootstrap must
    # dwarf it. This is the whole case for not reporting the per-prompt SE.
    assert np.all(boot[0] > p.se_prompt[0] * 3), "core leakage must inflate the error bar"
    # The reported SE is the between-core spread, which estimates the same quantity the
    # bootstrap does, so the two must now agree to well within a factor of two.
    assert np.all(boot[0] > p.se[0] * 0.5) and np.all(boot[0] < p.se[0] * 2.0), \
        "the reported SE and the core bootstrap must agree"


def test_pooled_b_respects_zero_weights():
    """Secondary other-role subtypes enter pooled B at weight 0 by the symbol table's rule."""
    ys = make_yardstick(width=4, layers=1)
    rows = [{"site_construct_token": 0, "n_tokens": 10, "anchor_piece": "for",
             "core_id": "c0"}]
    p1 = build(np.full((1, 1, 4), 1.0), rows, ys, "For", "B", subtype="string")
    p2 = build(np.full((1, 1, 4), 9.0), rows, ys, "For", "B", subtype="suite_header")
    out = pooled([p1, p2], weights={"string": 1.0, "suite_header": 0.0})
    assert np.allclose(out.mean, p1.mean), "a zero-weight subtype must not move pooled B"
    assert not np.allclose(out.mean, p2.mean)


def test_concept_adds_errors_in_quadrature():
    ys = make_yardstick(width=4, layers=1)
    rows = [{"site_construct_token": 0, "n_tokens": 10, "anchor_piece": "for",
             "core_id": "c0"}, {"site_construct_token": 0, "n_tokens": 10,
                                "anchor_piece": "for", "core_id": "c1"}]
    a = build(np.stack([np.full((1, 4), 3.0), np.full((1, 4), 5.0)]), rows, ys, "For", "A")
    b = build(np.stack([np.full((1, 4), 1.0), np.full((1, 4), 1.0)]), rows, ys, "For", "B")
    k = concept(a, b)
    assert np.allclose(k["delta"], 3.0, rtol=2 * DEFAULT_EPS ** 2)
    assert np.allclose(k["se"], np.sqrt(a.se ** 2 + b.se ** 2))


# --- the masked yardstick (Decision 2, deep tier) ---------------------------------

def test_removal_is_exact():
    """The masked yardstick is built by subtraction, so the subtraction must be exact.

    Keeping a full masked accumulator per construct would be 30 x 24 x 20 Welfords; only the
    excluded positions are accumulated, and they are a small fraction of C.
    """
    from profilebattery.profiles.yardstick import remove

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 6)) * 3 + 1

    def as_cell(a):
        return Cell(n=len(a), mean=a.mean(0), var=a.var(0, ddof=1),
                    active=np.full(a.shape[1], len(a)))

    got = remove(as_cell(x), as_cell(x[:120]))
    want = x[120:]
    assert got.n == 880
    assert np.allclose(got.mean, want.mean(0), atol=1e-10)
    assert np.allclose(got.var, want.var(0, ddof=1), atol=1e-8)


def test_masked_yardstick_deflates_where_the_token_sits_in_c():
    """A construct's own tokens inside C inflate sigma0 and so deflate every z.

    That deflation scales with the construct's corpus rate, which puts it on the programme's
    main covariate — hence the masked variant for the deep tier.
    """
    from profilebattery.profiles.yardstick import deflation, masked

    width = 32
    rng = np.random.default_rng(1)
    ordinary = rng.normal(size=(900, width))
    special = rng.normal(size=(100, width)) * 6          # the construct's own token
    both = np.vstack([ordinary, special])
    shared = Yardstick({(0, "b0:keyword"): Cell(n=len(both), mean=both.mean(0),
                                                var=both.var(0, ddof=1),
                                                active=np.full(width, len(both)))})
    excluded = {(0, "b0:keyword"): Cell(n=len(special), mean=special.mean(0),
                                        var=special.var(0, ddof=1),
                                        active=np.full(width, len(special)))}
    m, audit = masked(shared, excluded)
    assert m.support(0, "b0:keyword") == 900
    d = deflation(shared, m, 0, "b0:keyword")
    assert d < 0.8, f"masking should shrink sigma0 materially, got ratio {d:.3f}"
    assert np.allclose(m.mu(0, "b0:keyword"), ordinary.mean(0), atol=1e-10)
    assert audit["L0|b0:keyword"]["share"] == pytest.approx(0.1)
    assert audit["L0|b0:keyword"]["fallback"] is False


def test_masking_falls_back_when_the_excluded_share_is_large():
    """A frequent symbol's token can be a large share of its cell, and there the reverse
    merge subtracts near-equal quantities and its precision goes."""
    from profilebattery.profiles.yardstick import MAX_EXCLUDED_SHARE, masked

    width = 8
    total = Cell(n=1000, mean=np.ones(width), var=np.ones(width),
                 active=np.full(width, 1000))
    big = Cell(n=600, mean=np.ones(width), var=np.ones(width), active=np.full(width, 600))
    ys = Yardstick({(0, "b0:symbol"): total})
    m, audit = masked(ys, {(0, "b0:symbol"): big})
    assert audit["L0|b0:symbol"]["share"] > MAX_EXCLUDED_SHARE
    assert audit["L0|b0:symbol"]["fallback"]
    assert m.support(0, "b0:symbol") == 1000, "the shared cell is used unchanged"


def test_removal_of_everything_is_empty_not_negative():
    from profilebattery.profiles.yardstick import remove

    c = Cell(n=10, mean=np.ones(3), var=np.ones(3), active=np.full(3, 10))
    got = remove(c, c)
    assert got.n == 0 and np.all(got.var >= 0)


def test_accumulator_profile_meets_each_cell_own_yardstick(tmp_path):
    """Two cells with different yardsticks must not be pooled before standardising."""
    import numpy as np

    from profilebattery.profiles.profile import from_accumulators
    from profilebattery.profiles.yardstick import Cell, Yardstick
    from profilebattery.record.store import Welford, save_accumulators

    width = 4
    ys = Yardstick({(0, "b0:keyword"): Cell(n=500, mean=np.zeros(width),
                                            var=np.ones(width),
                                            active=np.full(width, 500)),
                    (0, "b4:newline-indent"): Cell(n=500, mean=np.full(width, 10.0),
                                                   var=np.ones(width),
                                                   active=np.full(width, 500))})
    acc = {}
    for cell, val in (("b0:keyword", 2.0), ("b4:newline-indent", 12.0)):
        w = Welford()
        w.update(np.full((10, width), val))
        acc[f"mk|c0|last|L0|{cell}"] = w
    save_accumulators(tmp_path / "a.npz", acc)
    p = from_accumulators(np.load(tmp_path / "a.npz"), ys, "For", "A", site="last")
    # both cells standardise to z = 2, so the count-weighted mean is 2 — pooling first and
    # standardising against one cell would have given 2 or -3 depending on which was chosen
    assert np.allclose(p.mean, 2.0, rtol=1e-3)
    assert set(p.cells) == {"b0:keyword", "b4:newline-indent"}


def test_accumulator_profile_refuses_stale_keys(tmp_path):
    import numpy as np

    from profilebattery.profiles.profile import from_accumulators
    from profilebattery.record.store import Welford, save_accumulators

    w = Welford()
    w.update(np.ones((2, 4)))
    save_accumulators(tmp_path / "old.npz", {"mk|c0|last|L0": w})
    with pytest.raises(ValueError, match="re-record"):
        from_accumulators(np.load(tmp_path / "old.npz"), make_yardstick(4, 1),
                          "For", "A", site="last")


def test_yardstick_streams_are_kept_apart(tmp_path):
    """C, M and X live in one file and are not interchangeable.

    C is the gated product (6,144 wide), M is m_k (2,240) whose spread is the variance of
    a linear combination and so cannot be derived from C, and X holds the excluded subsets
    for the masked variant, keyed by token and so five parts wide. Reading every `|n` key
    as a four-part C key crashes on X; mixing M into C would put an m_k denominator under
    an h profile, which no shape check downstream would catch at a single layer.
    """
    import numpy as np

    from profilebattery.profiles.yardstick import Yardstick

    p = tmp_path / "C_cells.npz"
    payload = {}
    for stream, width in (("C", 6), ("M", 3)):
        for cell in ("b0:keyword", "b1:symbol"):
            base = f"{stream}|L0|{cell}"
            payload[f"{base}|n"] = np.array([10])
            payload[f"{base}|mean"] = np.arange(width, dtype=np.float64)
            payload[f"{base}|m2"] = np.ones(width)
            payload[f"{base}|active"] = np.ones(width)
    # a five-part X key, which the old reader unpacked into four and died on
    payload["X|L0|b1:symbol|+=|n"] = np.array([2])
    payload["X|L0|b1:symbol|+=|mean"] = np.zeros(6)
    payload["X|L0|b1:symbol|+=|m2"] = np.ones(6)
    payload["X|L0|b1:symbol|+=|active"] = np.ones(6)
    np.savez(p, **payload)

    c = Yardstick.from_npz(p, stream="C")
    m = Yardstick.from_npz(p, stream="M")
    assert len(c.cells) == len(m.cells) == 2
    assert c.mu(0, "b0:keyword").shape == (6,)
    assert m.mu(0, "b0:keyword").shape == (3,)
    with pytest.raises(ValueError, match="no 'Z' cells"):
        Yardstick.from_npz(p, stream="Z")


def test_accumulator_route_equals_the_per_prompt_route():
    """The storage rule keeps per-prompt vectors at one reference, so the seven checks
    that read a built profile must get the same profile either way. They do, exactly:
    the accumulator route standardises each cell against its own yardstick and combines
    count-weighted, which is what the per-prompt route does prompt by prompt.
    """
    import gzip
    import io
    import json
    import os
    import pathlib

    import numpy as np

    from profilebattery.profiles.profile import build, from_accumulators
    from profilebattery.profiles.yardstick import Yardstick

    rec = data_root() / "pilot_record/untrained"
    batt = data_root() / "battery/twin1"
    ysp, shard, acc = (rec / "twin1__C_cells.npz", rec / "twin1__AnnAssign_A.npz",
                       rec / "twin1__AnnAssign_A_acc.npz")
    if not (ysp.exists() and shard.exists() and acc.exists()):
        pytest.skip("untrained reference not recorded")
    try:
        ys = Yardstick.from_npz(ysp)
        primary = np.load(io.BytesIO(shard.read_bytes()))["primary"]
        a = np.load(io.BytesIO(acc.read_bytes()))
    except (ValueError, KeyError, EOFError):
        pytest.skip("untrained reference is being written")
    rows = [json.loads(ln) for ln in gzip.open(batt / "AnnAssign_A.jsonl.gz", "rt")]
    pp = build(primary, rows, ys, "AnnAssign", "A")
    ac = from_accumulators(a, ys, "AnnAssign", "A", stream="h")
    assert pp.n == ac.n
    assert np.allclose(pp.mean, ac.mean, atol=1e-6)
