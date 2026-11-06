r"""A1.3 — §4.5's clustering and purity on the ONE-POOLED-CELL carrier sets.

§4.6 shows that which neurons are named depends on the yardstick; §4.5 reads the categories the
shared neurons recover. The two have never been run against each other: §4.5's clustering uses
the record's 20-cell sets, so "shared neurons recover identity, not structure" is a statement
about one yardstick. This rebuilds every construct's carrier set under **one pooled cell** at
the last checkpoint and reruns §4.5 unchanged on them.

The construction is `ablation_splits.masks`' own, with the `v` split dropped and the population
widened from the 16 position-marked constructs to all of P1: pool the 20-cell yardstick to one
cell, restandardise the A and B accumulators through it, take the top 5% of |A - B|.

**The purity tests are corrected for multiplicity.** §4.5 reports a p-value per category per
cut; across cuts and categories that is dozens of tests, and the draft quotes the small ones.
Holm is applied over every test this pass runs, and both the raw and the adjusted p are kept.

    python experiments/pooled_sets_169.py [--build] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import time

import numpy as np

from experiments.atlas import CARRIERS, D, K, LAST, provenance
from experiments.relatedness import OUT2

SETS = D / "validation/paper1/pooled_sets"


def build(limit=0):
    from experiments.b9_b10 import stream as _rs
    from experiments.b10 import pooled_yardstick
    from experiments.atlas import dataset_record, p1_set
    from profilebattery.profiles.profile import from_accumulators
    from profilebattery.profiles.yardstick import Yardstick
    t0 = time.time()
    names = sorted(p1_set())
    if limit:
        names = names[:limit]
    SETS.mkdir(parents=True, exist_ok=True)
    ys20 = {t: Yardstick.from_npz(dataset_record(LAST, f"{t}__C_cells.npz"))
            for t in ("twin1", "twin2")}
    ysp = {t: pooled_yardstick(ys20[t], "one") for t in ("twin1", "twin2")}
    done = 0
    for c in names:
        fp = SETS / f"{c}.npz"
        if fp.exists():
            continue
        try:
            got = {}
            for t in ("twin1", "twin2"):
                parts = {}
                for k in ("A", "B"):
                    acc = np.load(_rs(LAST, f"{t}__{c}_{k}_acc.npz")[1])
                    parts[k] = from_accumulators(acc, ysp[t], c, k, stream="h").mean
                v = np.asarray(parts["A"] - parts["B"], np.float64).ravel()
                got[t] = np.sort(np.argsort(-np.abs(v))[:K])
            np.savez_compressed(fp, twin1=got["twin1"], twin2=got["twin2"])
            done += 1
            if done % 10 == 0:
                print(f"  {done} built ({(time.time()-t0)/60:.1f} min)", flush=True)
        except Exception as exc:                                       # noqa: BLE001
            print(f"  {c}: SKIP {type(exc).__name__}: {str(exc)[:60]}", flush=True)
    print(f"built {done}; {len(list(SETS.glob('*.npz')))} on disk "
          f"({(time.time()-t0)/60:.1f} min)")


def holm(pvals):
    """Holm-Bonferroni: returns adjusted p in the input order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, run = len(pvals), [0.0] * len(pvals), 0.0
    for r, i in enumerate(idx):
        adj = min(1.0, (m - r) * pvals[i])
        run = max(run, adj)
        out[i] = run
    return out


def main() -> None:
    import scipy.cluster.hierarchy as H
    from scipy.spatial.distance import squareform
    from experiments.atlas import B6_CUTS, _b6_categories, _purity, inventory, p1_set
    from experiments.r1_turnover import load_index
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.build:
        build(a.limit)
    t0 = time.time()
    have = {p.stem: p for p in SETS.glob("*.npz")}
    names = sorted(have)
    if len(names) < 20:
        raise SystemExit(f"only {len(names)} pooled sets on disk; run with --build first")
    sets = {c: np.asarray(np.load(have[c])["twin1"], np.int64) for c in names}
    n = len(names)
    print(f"{n} constructs with a one-pooled-cell set at {LAST}", flush=True)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            inter = int(np.intersect1d(sets[names[i]], sets[names[j]]).size)
            J[i, j] = J[j, i] = inter / (len(sets[names[i]]) + len(sets[names[j]]) - inter)
    Dm = 1.0 - J
    np.fill_diagonal(Dm, 0.0)
    Z = H.linkage(squareform(Dm, checks=False), method="average")
    cats = _b6_categories(names, inventory())
    cuts = sorted({*B6_CUTS, *(len({v for v in c.values() if v is not None})
                               for c in cats["categories"].values())} - {0, 1})
    rng = np.random.default_rng(20261018)
    per_cut, tests = {}, []
    for k in cuts:
        cl = H.fcluster(Z, t=k, criterion="maxclust")
        block = {}
        for name, c in cats["categories"].items():
            r = _purity([c[x] for x in names], cl, rng)
            block[name] = r
            for lab, d in r.items():
                if isinstance(d, dict) and d.get("p") is not None:
                    tests.append((k, name, lab, d))
        per_cut[k] = {"n_clusters_realised": int(len(set(cl.tolist()))),
                      "sizes": sorted(collections.Counter(cl.tolist()).values(), reverse=True),
                      "categories": block}
    adj = holm([t[3]["p"] for t in tests])
    for (k, name, lab, d), q in zip(tests, adj):
        d["p_holm"] = float(q)
    sig_raw = sum(1 for t in tests if t[3]["p"] < 0.05)
    sig_adj = sum(1 for t in tests if t[3].get("p_holm", 1) < 0.05)
    out = {"what": "§4.5's clustering and purity on the ONE-POOLED-CELL carrier sets of §4.6, "
                   f"at {LAST}",
           "yardstick": "the record's 20 cells pooled to one, then the A and B accumulators "
                        "restandardised through it; the top 5% of |A - B|",
           "n_constructs": n, "cuts": cuts,
           "multiplicity": {"n_tests": len(tests), "correction": "Holm-Bonferroni over every "
                            "test this pass runs", "n_significant_raw": sig_raw,
                            "n_significant_after_holm": sig_adj},
           "per_cut": per_cut,
           "significant_after_holm": [
               {"cut": k, "category": name, "label": lab, "purity": d["purity"],
                "p": d["p"], "p_holm": d["p_holm"]}
               for (k, name, lab, d) in tests if d.get("p_holm", 1) < 0.05],
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance([f"record/{LAST}/twin1__<c>_{{A,B}}_acc.npz",
                                     f"record/{LAST}/twin{{1,2}}__C_cells.npz"])}
    (OUT2.parent / "paper1" / "POOLED_SETS_45.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(f"\n{len(tests)} purity tests over {len(cuts)} cuts and "
          f"{len(cats['categories'])} categories")
    print(f"  significant at 0.05 raw: {sig_raw};  after Holm: {sig_adj}")
    for e in out["significant_after_holm"][:10]:
        print(f"    cut {e['cut']:3d} {e['category']:14s} {str(e['label']):18s} "
              f"purity {e['purity']:.3f}  p {e['p']:.4f} -> holm {e['p_holm']:.4f}")
    print(f"-> POOLED_SETS_45.json  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
