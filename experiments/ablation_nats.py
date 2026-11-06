r"""A1.1, second pass — the ablation in nats, because the control unit is unstable.

Control units leave the body of both papers. The reason is measured, not stylistic: at 83.9 B
and 2,034 B the control's standard deviation is about **twice its own mean** (SD/mean 2.16 and
2.21) and it still *rises* with the mask count (0.0839 -> 0.0931 -> 0.1013 at 50/100/200 at the
end) while the mean is flat after 100. New draws land in the tail rather than shifting the
centre, so the denominator of a control unit has not converged and more masks would lower the
effect again rather than settle it. At 4.19 B the same ratio is 1.04 and the SD does settle.

So the primary column is in nats -- the carriers' rise minus the random sets' mean rise -- with
the ratio of the two beside it. Control units survive only in Table D.1, each cell carrying its
mask count and its stability flag, so a reader can see where the unit is trustworthy.

Two cautions, both from the data:

* The **ratio** cannot be taken per construct and then averaged. At the end of training the
  control mean runs from -0.004 to +0.26 nats, so a per-construct ratio divides by something
  near zero or negative and explodes or flips sign. The headline ratio here is therefore the
  **ratio of medians** (median carrier rise / median control mean), which is signed and stable;
  per-construct ratios are reported only where the control mean is positive, with the number
  excluded stated.
* Intervals are a **construct bootstrap**: the 16 constructs are the sample, resampled with
  replacement; it is their spread, not the control masks', that the interval describes.

Reads ABLATION_STABLE.json (written by ablation_stable.py from the 200-mask Colab run) and
rewrites TABLE_D1_ablation.{md,csv}.

    python experiments/ablation_nats.py
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np
from scipy.stats import binomtest

from experiments.ablation_full import NESTED_AT
from experiments.atlas import D, provenance
from experiments.relatedness import OUT2

SRC = OUT2 / "ABLATION_STABLE.json"
REFS_B = {"pretrain_iter_0001000": 4.19, "pretrain_iter_0020000": 83.9,
          "pretrain_iter_0484865": 2034.0}
N_BOOT = 10_000
SEED = 20261023


def _ci(x: np.ndarray, stat, rng) -> dict:
    """Construct bootstrap: resample the constructs, not the masks."""
    if len(x) == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    draws = np.array([stat(x[i]) for i in idx])
    return {"point": float(stat(x)), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "n": int(len(x))}


def _ci_ratio(num: np.ndarray, den: np.ndarray, rng) -> dict:
    """Ratio of medians, bootstrapped as a pair so each draw keeps construct identity."""
    if len(num) == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    idx = rng.integers(0, len(num), size=(N_BOOT, len(num)))
    draws = np.array([np.median(num[i]) / np.median(den[i])
                      if np.median(den[i]) != 0 else np.nan for i in idx])
    draws = draws[np.isfinite(draws)]
    return {"point": float(np.median(num) / np.median(den)),
            "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
            "n": int(len(num)), "n_bootstrap_draws_usable": int(len(draws))}


def _sign(x: np.ndarray) -> dict:
    """One-sided sign test that the effect is positive, over the constructs."""
    if len(x) == 0:
        return {"n_above_zero": None, "n": 0, "p": None}
    k = int((x > 0).sum())
    return {"n_above_zero": k, "n": int(len(x)),
            "p": float(binomtest(k, len(x), 0.5, alternative="greater").pvalue)}


def main() -> None:
    t0 = time.time()
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}; run experiments/ablation_stable.py first")
    src = json.loads(SRC.read_text())
    rng = np.random.default_rng(SEED)

    doc = {
        "what": "the ablation effect in nats, the unit the papers' bodies now use",
        "why_not_control_units":
            "at 83.9 B and 2,034 B the control SD is ~2.2x its own mean and still rising with "
            "the mask count while the mean is flat, so the unit's denominator has not "
            "converged; the same ratio is 1.04 at 4.19 B. Measured on the 200-mask run's "
            "nested prefixes (50/100/200).",
        "definition": {
            "rise_over_random_nats":
                "PRIMARY. (loss rise with the carrier set zeroed) minus (mean loss rise over "
                "the size-matched random sets), in nats, at the construct's own positions. "
                "Signed; no absolute value is taken anywhere.",
            "ratio_rise_to_random":
                "the carriers' rise divided by the random sets' mean rise. Reported as a "
                "RATIO OF MEDIANS over the constructs, because per construct the denominator "
                "runs down to and below zero at the later references. Per-construct ratios "
                "are given only where the control mean is > 0.",
            "sign_test":
                "count of constructs with rise_over_random_nats > 0, one-sided binomial "
                "against p = 0.5, over the 16 constructs at that reference.",
            "interval":
                f"construct bootstrap, {N_BOOT} resamples of the 16 constructs with "
                f"replacement, 95% percentile interval, seed {SEED}. It describes the spread "
                f"over constructs, not over control masks.",
            "old_set_at_the_end":
                "the 84 B carrier set removed at the end of pretraining, under the SAME masks "
                "as the current set: against the random sets (drifted - random mean) and "
                "against the current set (drifted - current).",
            "control_units":
                "retained ONLY in Table D.1, with each cell's mask count and stability flag.",
        },
        "fixed_before_computing": True,
        "n_bootstrap": N_BOOT, "seed": SEED, "nested_prefixes": list(NESTED_AT),
        "per_reference": {},
    }

    rows = []
    for ref, e in src["per_reference"].items():
        pc = e["per_construct"]
        names = sorted(pc)
        cur = np.array([pc[c]["loss_rise_nats"] for c in names], float)
        ctl = np.array([pc[c]["control_mean_nats"] for c in names], float)
        rise = cur - ctl
        pos = ctl > 0
        per_ratio = np.divide(cur, ctl, out=np.full_like(cur, np.nan), where=pos)

        old = np.array([pc[c]["old_set_rise_nats"] if pc[c]["old_set_rise_nats"] is not None
                        else np.nan for c in names], float)
        has_old = np.isfinite(old)
        old_vs_random = (old - ctl)[has_old]
        old_vs_current = (old - cur)[has_old]

        doc["per_reference"][ref] = {
            "tokens_b": REFS_B.get(ref), "n_constructs": len(names),
            "median_rise_over_random_nats": _ci(rise, np.median, rng),
            "ratio_of_medians_rise_to_random": _ci_ratio(cur, ctl, rng),
            "sign_test_rise_over_random": _sign(rise),
            "median_carrier_rise_nats": float(np.median(cur)),
            "median_control_mean_nats": float(np.median(ctl)),
            "per_construct_ratio": {
                "median_where_control_mean_positive":
                    float(np.nanmedian(per_ratio)) if pos.any() else None,
                "n_used": int(pos.sum()),
                "n_excluded_control_mean_not_positive": int((~pos).sum()),
                "excluded": [c for c, p in zip(names, pos) if not p]},
            "old_set": {
                "n_constructs_with_an_old_set": int(has_old.sum()),
                "n_without": int((~has_old).sum()),
                "without": [c for c, h in zip(names, has_old) if not h],
                "median_vs_random_nats": _ci(old_vs_random, np.median, rng),
                "sign_test_vs_random": _sign(old_vs_random),
                "median_vs_current_nats": _ci(old_vs_current, np.median, rng),
                "sign_test_vs_current": _sign(old_vs_current),
                "n_old_below_current": int((old_vs_current < 0).sum())},
            "per_construct": {
                c: {"rise_over_random_nats": float(rise[i]),
                    "carrier_rise_nats": float(cur[i]),
                    "control_mean_nats": float(ctl[i]),
                    "ratio_rise_to_random": (None if not pos[i] else float(per_ratio[i])),
                    "old_set_rise_nats": (None if not has_old[i] else float(old[i])),
                    "old_set_over_random_nats": (None if not has_old[i]
                                                 else float(old[i] - ctl[i])),
                    "old_set_minus_current_nats": (None if not has_old[i]
                                                   else float(old[i] - cur[i]))}
                for i, c in enumerate(names)},
        }
        for i, c in enumerate(names):
            v = pc[c]
            rows.append({
                "reference": ref, "tokens_b": REFS_B.get(ref), "construct": c,
                "rise_over_random_nats": float(rise[i]),
                "carrier_rise_nats": float(cur[i]), "control_mean_nats": float(ctl[i]),
                "ratio_rise_to_random": (None if not pos[i] else float(per_ratio[i])),
                "old_set_over_random_nats": (None if not has_old[i]
                                             else float(old[i] - ctl[i])),
                "old_set_minus_current_nats": (None if not has_old[i]
                                               else float(old[i] - cur[i])),
                "effect_control_units": v["effect_control_units"],
                "old_set_control_units": v["old_set_control_units"],
                "control_sd_nats": v["control_sd_nats"],
                "n_masks_reached": v["n_masks_reached"],
                "control_sd_settled": v["control_sd_settled"]})

    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance(["validation/paper2/ABLATION_STABLE.json"])
    (OUT2 / "ABLATION_NATS.json").write_text(json.dumps(doc, indent=1, default=float))

    cols = ["reference", "tokens_b", "construct", "rise_over_random_nats", "carrier_rise_nats",
            "control_mean_nats", "ratio_rise_to_random", "old_set_over_random_nats",
            "old_set_minus_current_nats", "effect_control_units", "old_set_control_units",
            "control_sd_nats", "n_masks_reached", "control_sd_settled"]
    head = (
        "**Table D.1 — the ablation at three references, in nats, with control units kept for "
        "reference.** `rise_over_random_nats` is the carriers' loss rise minus the mean rise "
        "over the size-matched random sets, signed, at the construct's own positions; it is "
        "the unit the papers' bodies use. `ratio_rise_to_random` is the carriers' rise divided "
        "by the random mean, left blank where the random mean is not positive. "
        "`effect_control_units` is (rise − control mean) / control SD; masks are drawn once at "
        f"{max(NESTED_AT)} and the SD read at {', '.join(map(str, NESTED_AT))} from nested "
        "prefixes, and `control_sd_settled` is true where the SD changed by under 5% on the "
        "last doubling. **Control units are unreliable where that flag is false** — at 83.9 B "
        "and 2,034 B the control's SD is about twice its own mean and still rising with the "
        "mask count, which is why the body of both papers reports nats. `old_set_*` columns "
        "are the 84 B carrier set removed at the end of pretraining under the same masks.")
    md = [head, "", "| " + " | ".join(cols) + " |",
          "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        md.append("| " + " | ".join(
            "—" if r.get(c) is None else
            (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    (OUT2 / "TABLE_D1_ablation.md").write_text("\n".join(md) + "\n")
    with open(OUT2 / "TABLE_D1_ablation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows([[r.get(c) for c in cols] for r in rows])

    for ref, e in doc["per_reference"].items():
        m, r_, s = (e["median_rise_over_random_nats"], e["ratio_of_medians_rise_to_random"],
                    e["sign_test_rise_over_random"])
        print(f"  {ref[-7:]} ({e['tokens_b']:7.1f} B): "
              f"{m['point']:+.3f} nats [{m['lo']:+.3f}, {m['hi']:+.3f}]   "
              f"ratio of medians {r_['point']:5.2f}x [{r_['lo']:.2f}, {r_['hi']:.2f}]   "
              f"{s['n_above_zero']}/{s['n']} above zero, p = {s['p']:.3f}")
    o = doc["per_reference"]["pretrain_iter_0484865"]["old_set"]
    print(f"  84 B set at the end ({o['n_constructs_with_an_old_set']} constructs): "
          f"vs random {o['median_vs_random_nats']['point']:+.3f} nats "
          f"[{o['median_vs_random_nats']['lo']:+.3f}, {o['median_vs_random_nats']['hi']:+.3f}], "
          f"{o['sign_test_vs_random']['n_above_zero']}/{o['sign_test_vs_random']['n']}, "
          f"p = {o['sign_test_vs_random']['p']:.3f}")
    print(f"                       vs the current set "
          f"{o['median_vs_current_nats']['point']:+.3f} nats "
          f"[{o['median_vs_current_nats']['lo']:+.3f}, "
          f"{o['median_vs_current_nats']['hi']:+.3f}], "
          f"{o['n_old_below_current']}/{o['sign_test_vs_current']['n']} below the current set")
    print(f"-> {OUT2 / 'ABLATION_NATS.json'}, TABLE_D1_ablation.{{md,csv}}")


if __name__ == "__main__":
    main()
