r"""B10 — the spacing-adjusted turnover curve, fitted on all pairs.

Paper 2 §4.3 says the change "per 1,000 steps" is roughly constant across pretraining. True
1,000-step pairs exist only in the dense block: **19 of them, iterations 1,000 to 20,000 (to 84
B tokens)**. The other 15 recorded neighbour pairs are ×1.25 ratio pairs with gaps from 5,000 to
91,000 steps, so any per-1,000-step figure past 84 B comes from dividing a wide pair's turnover
by its step count -- which assumes turnover is linear in steps, and A38 measured that it is not
(Spearman -0.813 against spacing).

This fits on **all 1,404 pairs** rather than neighbours, because neighbouring pairs confound the
start point with the gap. Model fixed in `RELATEDNESS.json` before fitting:

    turnover(t, D) = A(t) * (1 - exp(-r(t) * D))

per start point t, with the per-step rate at t being the slope at D -> 0, A(t) * r(t).

    python experiments/b10_fit.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from experiments.atlas import CARRIERS, K, provenance
from experiments.relatedness import OUT2

DENSE_END = 20000          # iterations; past here no true 1,000-step pair exists


def fit_exp(d, y):
    """B10's model: y(D) = A (1 - exp(-r D)); the per-step rate is the slope at D -> 0.

    A is swept coarsely and r taken by least squares on -log(1 - y/A) = r*D, keeping the sweep
    point with the smallest SSE on the original scale. Module-level so B3's all-pairs stability
    fit is **the same code**, not a re-implementation of the same formula.
    """
    import numpy as np
    d, y = np.asarray(d, float), np.asarray(y, float)
    if d.size < 4:
        return None
    A0 = min(max(y.max(), 1e-6), 1.0)
    best = None
    for A in np.linspace(max(A0, 0.05), 1.0, 20):
        z = 1.0 - np.clip(y / A, 1e-9, 1 - 1e-9)
        r = float((-np.log(z) * d).sum() / (d * d).sum())
        pred = A * (1 - np.exp(-r * d))
        sse = float(((pred - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, A, r)
    return {"A": best[1], "r": best[2], "slope0": best[1] * best[2]}


def main() -> None:
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = [int(re.search(r"(\d+)$", r).group(1)) for r in refs]
    names = sorted(p1_set())
    idx = load_index(CARRIERS, refs)
    sets = {}
    for r in refs:
        d = idx.get(r, {})
        sets[r] = {c: np.sort(np.asarray(d[f"twin1/{c}|product"], np.int64))
                   for c in names if f"twin1/{c}|product" in d}
    print(f"{len(refs)} references, {len(names)} constructs, "
          f"{len(refs)*(len(refs)-1)//2} pairs", flush=True)

    rows = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            a, b = refs[i], refs[j]
            dstep = iters[j] - iters[i]
            for c in names:
                if c in sets[a] and c in sets[b]:
                    keep = np.intersect1d(sets[a][c], sets[b][c]).size / K
                    rows.append((iters[i], dstep, 1.0 - keep, c))
    print(f"{len(rows):,} (start, gap, turnover) points", flush=True)

    starts = sorted({r[0] for r in rows})
    rng = np.random.default_rng(20261022)
    per_t, fitted = {}, 0
    for t in starts:
        sub = [(d, y, c) for (s, d, y, c) in rows if s == t]
        if len(sub) < 6:
            continue
        D = np.array([x[0] for x in sub], float)
        Y = np.array([x[1] for x in sub], float)
        C = np.array([x[2] for x in sub])

        def fit(mask):
            return fit_exp(D[mask], Y[mask])

        f = fit(np.ones(len(D), bool))
        if f is None:
            continue
        boot = []
        uc = np.unique(C)
        for _ in range(200):
            pick = set(rng.choice(uc, len(uc), replace=True).tolist())
            m = np.array([c in pick for c in C])
            g = fit(m)
            if g:
                boot.append(g["slope0"])
        per_t[int(t)] = {
            "n_points": len(sub), "A": f["A"], "r_per_step": f["r"],
            "per_step_rate": f["slope0"],
            "per_1000_steps": f["slope0"] * 1000,
            "ci95": ([float(np.quantile(boot, 0.025)) * 1000,
                      float(np.quantile(boot, 0.975)) * 1000] if len(boot) > 20 else None)}
        fitted += 1
    dense = [v["per_1000_steps"] for t, v in per_t.items() if t <= DENSE_END]
    tail = [v["per_1000_steps"] for t, v in per_t.items() if t > DENSE_END]
    print(f"\nfitted at {fitted} start points")
    print(f"  per-1,000-step turnover, start <= 84 B (iter {DENSE_END}): "
          f"median {np.median(dense):.4f}  n={len(dense)}")
    print(f"  per-1,000-step turnover, start  > 84 B:                   "
          f"median {np.median(tail):.4f}  n={len(tail)}")
    ts = sorted(per_t)
    lv = [per_t[t]["per_1000_steps"] for t in ts]
    rho = float(np.corrcoef(np.log(ts), lv)[0, 1])
    verdict = ("falling" if rho < -0.3 else "rising" if rho > 0.3 else "roughly constant")
    print(f"  correlation of the rate with log start step: {rho:+.3f} -> {verdict}")
    out = {"what": "turnover fitted on all pairs as A(t)(1-exp(-r(t)D)); the per-step rate is "
                   "the slope at D->0",
           "model_fixed_in": "validation/paper2/RELATEDNESS.json rules_fixed_22_Oct.b10_model",
           "provenance_of_the_claim": {
               "true_1000_step_pairs": 19, "span": "iterations 1,000-20,000, to 84 B tokens",
               "tail_pairs": 15, "tail_spacing": "x1.25, gaps 5,000-91,000 steps",
               "so": "a per-1,000-step figure past 84 B can only come from dividing a wide "
                     "pair's turnover by its step count"},
           "n_points": len(rows), "per_start": per_t,
           "median_per_1000_steps_dense": float(np.median(dense)),
           "median_per_1000_steps_tail": float(np.median(tail)),
           "rate_vs_log_start_corr": rho, "verdict": verdict,
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["validation/carriers/*.npz"])}
    OUT2.mkdir(parents=True, exist_ok=True)
    (OUT2 / "B10_SPACING_ADJUSTED.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {OUT2 / 'B10_SPACING_ADJUSTED.json'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
