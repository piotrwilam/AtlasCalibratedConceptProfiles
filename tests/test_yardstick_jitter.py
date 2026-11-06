"""The yardstick's analytic jitter, standing in for a resample the C pass cannot support."""
import numpy as np
import pytest

from profilebattery.profiles.yardstick import Cell, Yardstick


def ys(n=2000, d=8, eps=0.0):
    rng = np.random.default_rng(0)
    cells = {(0, "b0:keyword"): Cell(n=n, mean=rng.normal(size=d),
                                     var=np.abs(rng.normal(size=d)) + 1.0,
                                     active=np.full(d, n)),
             (0, "b1:identifier"): Cell(n=n * 5, mean=rng.normal(size=d),
                                        var=np.abs(rng.normal(size=d)) + 1.0,
                                        active=np.full(d, n * 5))}
    return Yardstick(cells, eps)


def test_rel_se_follows_one_over_sqrt_2n():
    y = ys(n=2000)
    assert y.rel_se_sigma(0, "b0:keyword") == pytest.approx(1 / np.sqrt(2 * 1999))
    # a cell with five times the support has sqrt(5) times the precision
    assert y.rel_se_sigma(0, "b0:keyword") / y.rel_se_sigma(0, "b1:identifier") == \
        pytest.approx(np.sqrt(5), rel=1e-3)


def test_jitter_leaves_mu_and_n_untouched():
    """mu0 cancels in A - B, so only sigma0 may move."""
    y = ys()
    j = y.jittered(np.random.default_rng(1))
    for key in y.cells:
        assert np.array_equal(y.cells[key].mean, j.cells[key].mean)
        assert y.cells[key].n == j.cells[key].n
        assert np.array_equal(y.cells[key].active, j.cells[key].active)


def test_jitter_moves_sigma_by_about_its_rel_se():
    y = ys(n=200, d=4000)
    r = y.rel_se_sigma(0, "b0:keyword")
    ratios = []
    for seed in range(40):
        j = y.jittered(np.random.default_rng(seed))
        ratios.append(j.sigma(0, "b0:keyword") / y.sigma(0, "b0:keyword"))
    spread = np.std(np.concatenate(ratios))
    assert spread == pytest.approx(r, rel=0.15)


def test_jitter_is_reproducible_and_variable():
    y = ys()
    a = y.jittered(np.random.default_rng(7)).sigma(0, "b0:keyword")
    b = y.jittered(np.random.default_rng(7)).sigma(0, "b0:keyword")
    c = y.jittered(np.random.default_rng(8)).sigma(0, "b0:keyword")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sigma_stays_positive_under_heavy_jitter():
    y = ys(n=3)                      # rel SE ~ 0.5, so the factor can go negative unclipped
    for seed in range(20):
        assert np.all(y.jittered(np.random.default_rng(seed)).sigma(0, "b0:keyword") > 0)
