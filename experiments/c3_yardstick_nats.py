r"""[TBC: 2] -- Appendix C.3's yardstick comparison restated in nats over the random mean.

C.3 compares three yardsticks -- the code-coordinate split, the 20-cell split and one pooled
cell -- by how much removing each one's carrier set raises the loss, and states the comparison in
**control standard deviations**. That unit does not survive: at 83.9 B and 2,034 B the control's
spread is about twice its own mean and still rising with the mask count, so its denominator has
not converged (ABLATION_NATS.json). The artefacts already carry `rise_over_random_mean`, the
same effect in nats, which needs no spread at all; this restates the comparison on it.

**The statistic is paired.** C.3's pre-registered test is the median *code-coordinate minus one
cell*, and that is the median of the per-construct differences, not the difference of the two
medians: at 4.19 B the paired median is -0.55 control units (C.3's figure) while the difference
of medians is +1.19. Pairing is also the right reading, since both yardsticks are measured on the
same construct at the same reference against the same random masks. Both are reported below so
the two can never again be confused.

Sign tests are over the 16 position-marked constructs, one-sided for "the effect is positive" on
each yardstick and two-sided for the paired differences, which have no prior direction.

**This reads the 20-mask run** (`validation/paper1/ablation_splits/`), the only one that carries
all three yardsticks; the 200-mask run of 23 Oct re-measured the 20-cell split alone. The mask
count is stated in the artefact, and the nats figures do not depend on the control's spread,
which is the quantity the small mask count made unreliable.

    python experiments/c3_yardstick_nats.py
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np
from scipy.stats import binomtest

from experiments.atlas import D, provenance

IN = D / "validation/paper1/ablation_splits"
OUT = D / "validation/paper1"
REFS = {"pretrain_iter_0001000": 4.19, "pretrain_iter_0020000": 83.9,
        "pretrain_iter_0484865": 2034.0}
YARDSTICKS = [("v", "code-coordinate"), ("twenty", "20-cell"), ("one", "one pooled cell")]
N_BOOT = 10_000
SEED = 20261024


def _ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    b = np.median(x[idx], axis=1)
    return {"median": float(np.median(x)),
            "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def _sign(x, alternative="greater"):
    x = np.asarray(x, float)
    k = int((x > 0).sum())
    return {"n_above_zero": k, "n": int(x.size),
            "p": float(binomtest(k, x.size, 0.5, alternative=alternative).pvalue)}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    doc = {
        "what": "[TBC: 2] Appendix C.3's three-yardstick ablation comparison, in nats over the "
                "random sets' mean rather than in control standard deviations",
        "why": "the control's spread is long-tailed at 83.9 B and 2,034 B -- about twice its "
               "own mean and still rising with the mask count -- so a control standard "
               "deviation has no converged denominator there. Nats over the random mean need "
               "no spread.",
        "definition": {
            "rise_over_random_mean": "the loss with the yardstick's carrier set zeroed, minus "
                                     "the mean loss over the size-matched random masks, in "
                                     "nats, at the construct's own positions. Signed.",
            "yardsticks": {k: v for k, v in YARDSTICKS},
            "paired_difference": "per construct, one yardstick's rise minus another's, then "
                                 "the median over constructs. C.3's pre-registered test is "
                                 "this paired median for code-coordinate minus one cell; the "
                                 "difference of the two medians is a different number and is "
                                 "reported beside it so they cannot be confused.",
            "sign_test": "count above zero over the 16 constructs; one-sided for a yardstick's "
                         "own effect, two-sided for a paired difference, which has no prior "
                         "direction",
            "interval": f"construct bootstrap, {N_BOOT} resamples, 95% percentile, seed {SEED}",
            "mask_count": "20 random masks (the only run carrying all three yardsticks); the "
                          "nats figures do not use the control's spread",
        },
        "fixed_before_computing": True, "per_reference": {},
    }

    rows = []
    for ref, tb in REFS.items():
        f = IN / f"ABLATION_{ref}.json"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        a = json.loads(f.read_text())
        cs = a["constructs"]
        names = sorted(c for c in cs if all(k in cs[c] for k, _ in YARDSTICKS))
        if not names:
            raise SystemExit(f"{ref}: no construct carries all three yardsticks")
        nats = {k: np.array([cs[c][k]["rise_over_random_mean"] for c in names], float)
                for k, _ in YARDSTICKS}
        sd = {k: np.array([cs[c][k]["over_random_in_sd"] for c in names], float)
              for k, _ in YARDSTICKS}

        e = {"tokens_b": tb, "n_constructs": len(names), "n_random_masks": a.get("n_random"),
             "constructs": names, "per_yardstick": {}, "paired_differences": {}}
        for k, label in YARDSTICKS:
            e["per_yardstick"][k] = {
                "label": label, **_ci(nats[k], rng), "sign_test": _sign(nats[k]),
                "median_control_units_for_reference": float(np.median(sd[k]))}
        for ka, kb in (("v", "one"), ("v", "twenty"), ("twenty", "one")):
            d_nats = nats[ka] - nats[kb]
            e["paired_differences"][f"{ka}_minus_{kb}"] = {
                "label": f"{dict(YARDSTICKS)[ka]} minus {dict(YARDSTICKS)[kb]}",
                **_ci(d_nats, rng), "sign_test": _sign(d_nats, "two-sided"),
                "difference_of_medians_nats": float(np.median(nats[ka]) - np.median(nats[kb])),
                "median_control_units_for_reference": float(np.median(sd[ka] - sd[kb]))}
        doc["per_reference"][ref] = e

        for c in names:
            r = {"reference": ref, "tokens_b": tb, "construct": c}
            for k, _ in YARDSTICKS:
                r[f"{k}_nats_over_random"] = cs[c][k]["rise_over_random_mean"]
                r[f"{k}_control_units"] = cs[c][k]["over_random_in_sd"]
            r["v_minus_one_nats"] = (cs[c]["v"]["rise_over_random_mean"]
                                     - cs[c]["one"]["rise_over_random_mean"])
            rows.append(r)

    # the pre-registered test, stated once, at all three references
    pre = {ref: doc["per_reference"][ref]["paired_differences"]["v_minus_one"]
           for ref in REFS if ref in doc["per_reference"]}
    doc["pre_registered_test"] = {
        "statistic": "median over constructs of (code-coordinate rise - one-cell rise), paired",
        "in_nats": {r: {"median": v["median"], "ci95": v["ci95"],
                        "n_above_zero": v["sign_test"]["n_above_zero"],
                        "n": v["sign_test"]["n"], "p": v["sign_test"]["p"]}
                    for r, v in pre.items()},
        "in_control_units_as_C3_stated_it": {
            r: v["median_control_units_for_reference"] for r, v in pre.items()}}

    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance(["validation/paper1/ablation_splits/ABLATION_*.json"])
    (OUT / "C3_YARDSTICK_NATS.json").write_text(json.dumps(doc, indent=1, default=float))

    cols = list(rows[0])
    head = ("**Table C.3 — the three yardsticks' ablation effects, in nats over the random "
            "sets' mean.** Each figure is the loss with that yardstick's carrier set zeroed "
            "minus the mean loss over 20 size-matched random masks, at the construct's own "
            "positions, signed. Control units are kept beside for reference only: the "
            "control's spread is long-tailed at the two later references and does not settle "
            "with more masks, which is why the comparison is now stated in nats.")
    md = [head, "", "| " + " | ".join(cols) + " |",
          "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        md.append("| " + " | ".join(
            "—" if r.get(c) is None else
            (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    (OUT / "TABLE_C3_yardsticks.md").write_text("\n".join(md) + "\n")
    with open(OUT / "TABLE_C3_yardsticks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows([[r.get(c) for c in cols] for r in rows])

    print(f"  {'reference':<10s} " + "".join(f"{lab:>26s}" for _, lab in YARDSTICKS))
    for ref, e in doc["per_reference"].items():
        cells = ""
        for k, _ in YARDSTICKS:
            y = e["per_yardstick"][k]
            cells += (f"{y['median']:+8.3f} ({y['sign_test']['n_above_zero']:2d}/"
                      f"{y['sign_test']['n']}, p={y['sign_test']['p']:.3f})")
        print(f"  {e['tokens_b']:8.1f} B {cells}")
    print("\n  the pre-registered test, code-coordinate minus one cell, paired, in nats:")
    for ref, v in pre.items():
        print(f"    {REFS[ref]:8.1f} B  {v['median']:+.3f} "
              f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  "
              f"{v['sign_test']['n_above_zero']}/{v['sign_test']['n']} above zero, "
              f"p = {v['sign_test']['p']:.3f}   "
              f"(C.3 in control units: {v['median_control_units_for_reference']:+.2f})")
    print(f"-> {OUT / 'C3_YARDSTICK_NATS.json'}, TABLE_C3_yardsticks.{{md,csv}}")


if __name__ == "__main__":
    main()
