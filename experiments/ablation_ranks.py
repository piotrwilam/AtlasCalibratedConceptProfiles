r"""A1.1b -- the ablation read as a RANK among the control masks, which a long tail cannot move.

A control unit divides by the control's standard deviation. At 83.9 B and 2,034 B that SD is
about **twice its own mean** and still *rising* with the mask count (0.0839 -> 0.0931 -> 0.1013
at 50/100/200 at the end) while the mean is flat after 100 -- new draws land in the tail rather
than shifting the centre. The denominator has not converged, so more masks lower the effect
again rather than settle it, and the unit cannot carry a column of Table D.1.

A **rank** needs no denominator. "The carrier set's loss rise exceeds all N random sets of the
same size" is exact whatever shape the control has, and a heavy tail works *against* the claim
instead of deflating a scale. The exact one-sided p is `(1 + #{random >= carrier}) / (1 + N)`.

**Why N is 320 and not 200.** The floor p of an N-mask control is `1/(N+1)`, and Holm's first
comparison across the 16 constructs is against `0.05/16 = 0.003125`. At N = 200 the floor is
0.00498, which exceeds that threshold, so Holm stops at step one and **no construct can clear
0.05 however the data fall** -- M would be 0 by arithmetic, not by evidence. N = 320 gives a
floor of 0.00312, just under the threshold, which is the smallest count that lets a clean sweep
survive the correction (319 exactly; 320 for a round number). Raising N costs nothing in
comparability: `random_masks` draws sequentially from one seeded generator, so the **first 200
of a 320-draw are identical to the 23 Oct draw**, and that prefix is what this script checks the
earlier run against.

Needs `random_per_mask` in the FULL_*.json, which `ablation_full.py` writes by default since
24 Oct. The 200-mask run of 23 Oct stored only the mean and SD, so it must be rerun -- see
`notebooks/colab_ablation_ranks.ipynb`.

    python experiments/ablation_ranks.py --in $ATLAS_DATA/validation/ablation_ranks

On success it rewrites TABLE_D1_ablation.{md,csv} in the A1.1b form -- nats primary, rank, exact
p, Holm flag; **no control units and no `settled` flag** -- and prints the fixed clause with its
numbers filled in.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from experiments.atlas import D, provenance
from experiments.relatedness import OUT2

IN = D / "validation/ablation_ranks"
REFS_B = {"pretrain_iter_0001000": 4.19, "pretrain_iter_0020000": 83.9,
          "pretrain_iter_0484865": 2034.0}
ORDER = ["pretrain_iter_0001000", "pretrain_iter_0020000", "pretrain_iter_0484865"]
#: the 23 Oct run's mask count; the first this many draws of any larger run must reproduce it
BASELINE_N = 200
#: the mean and SD recomputed from that prefix must match the 23 Oct figures this closely
TOL = 1e-6
ALPHA = 0.05


def holm(p: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Holm-Bonferroni, step-down. Stops at the first failure, as the procedure requires."""
    p = np.asarray(p, float)
    m = p.size
    keep = np.zeros(m, bool)
    for i, idx in enumerate(np.argsort(p)):
        if p[idx] <= alpha / (m - i):
            keep[idx] = True
        else:
            break
    return keep


def rank_of(value: float, draws: np.ndarray) -> dict:
    """Where `value` sits among `draws`, and the exact one-sided p that goes with it."""
    n = int(draws.size)
    n_ge = int((draws >= value).sum())
    return {"n_masks": n, "n_random_at_or_above": n_ge, "n_random_below": n - n_ge,
            "rank_from_top": n_ge + 1, "share_below": (n - n_ge) / n,
            "exact_one_sided_p": (1 + n_ge) / (1 + n), "exceeds_every_mask": n_ge == 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(IN))
    ap.add_argument("--allow-vacuous-holm", action="store_true",
                    help="proceed even when the mask count makes every Holm result 0")
    args = ap.parse_args()
    t0 = time.time()

    import glob
    files = sorted(glob.glob(f"{args.inp}/FULL_*.json"))
    if not files:
        raise SystemExit(f"no FULL_*.json under {args.inp}")
    base = {}
    bp = OUT2 / "ABLATION_STABLE.json"
    if bp.exists():
        base = json.loads(bp.read_text())["per_reference"]

    doc = {"what": "A1.1b: the ablation read as a rank among the control masks",
           "why": "the control is long-tailed at 83.9 B and 2,034 B -- SD ~2.2x its own mean "
                  "and still rising at 200 masks -- so a control unit has no converged "
                  "denominator there. A rank has none to converge.",
           "definition": {
               "rank_from_top": "1 + the number of random masks whose loss rise is at or above "
                                "the carrier set's",
               "exact_one_sided_p": "(1 + #{random >= carrier}) / (1 + n_masks)",
               "floor": "the smallest p an n-mask control allows is 1/(n+1); at n = 320 that "
                        "is 0.00312, just under Holm's first threshold of 0.05/16 = 0.003125, "
                        "which is why 320 and not 200 were drawn",
               "holm": f"Holm-Bonferroni at alpha = {ALPHA} across the 16 constructs, "
                       f"separately at each reference",
               "old_set": "the same rank for the 84 B carrier set removed at the end, against "
                          "the SAME masks",
               "mask_check": f"the mean and SD recomputed from the FIRST {BASELINE_N} per-mask "
                             f"values are compared with the 23 Oct run's stored figures to "
                             f"within {TOL:g}; a mismatch means different masks and stops the "
                             f"read"},
           "fixed_before_computing": True, "baseline_prefix": BASELINE_N,
           "mask_check": {}, "per_reference": {}}

    rows, missing, counts = [], [], set()
    for f in files:
        a = json.loads(open(f).read())
        ref = a["reference"]
        per, checked = {}, 0
        for c, v in sorted(a["constructs"].items()):
            pm = v.get("random_per_mask")
            if not pm:
                missing.append(f"{ref}/{c}")
                continue
            draws = np.asarray(pm, float)
            counts.add(int(draws.size))
            if draws.size < BASELINE_N:
                raise SystemExit(
                    f"REFUSED: {ref}/{c} carries {draws.size} masks, fewer than the "
                    f"{BASELINE_N} of the 23 Oct run, so the two cannot be compared.")
            # the first BASELINE_N draws must BE the 23 Oct draws. random_masks() draws
            # sequentially from one seeded generator, so a larger run's prefix is identical;
            # if it is not, a different seed or a different k was used and nothing below is
            # comparable with what the papers already quote.
            b = (base.get(ref, {}).get("per_construct", {}) or {}).get(c)
            if b is not None:
                pre = draws[:BASELINE_N]
                dm = abs(float(pre.mean()) - b["control_mean_nats"])
                ds = abs(float(pre.std(ddof=1)) - b["control_sd_nats"])
                if dm > TOL or ds > TOL:
                    raise SystemExit(
                        f"REFUSED: {ref}/{c}: the first {BASELINE_N} per-mask values do not "
                        f"reproduce the 23 Oct control (mean off by {dm:.3g}, SD by {ds:.3g}). "
                        f"Different masks were drawn, so the two runs are not comparable.")
                checked += 1
            e = {"carrier_rise_nats": v["current"],
                 "rise_over_random_nats": v["current"] - v["random_mean"],
                 "control_mean_nats": float(draws.mean()),
                 "current": rank_of(v["current"], draws)}
            if v.get("drifted") is not None:
                e["old_set_rise_nats"] = v["drifted"]
                e["old_set"] = rank_of(v["drifted"], draws)
            per[c] = e
        if not per:
            continue

        names = sorted(per)
        pvals = np.array([per[c]["current"]["exact_one_sided_p"] for c in names])
        keep = holm(pvals)
        for c, k in zip(names, keep):
            per[c]["current"]["holm_significant"] = bool(k)
        # the 84 B set gets its own Holm family. It is a different question asked of the same
        # masks -- "does the OLD set still beat random" -- so correcting it inside the current
        # set's family would mix two hypotheses under one correction.
        old_names = [c for c in names if "old_set" in per[c]]
        if old_names:
            op = np.array([per[c]["old_set"]["exact_one_sided_p"] for c in old_names])
            ok = holm(op)
            for c, k in zip(old_names, ok):
                per[c]["old_set"]["holm_significant"] = bool(k)

        doc["mask_check"][ref] = {"n_constructs_checked_against_23_oct": checked,
                                  "n_constructs": len(per)}
        sweep = [c for c in names if per[c]["current"]["exceeds_every_mask"]]
        nm = int(max(per[c]["current"]["n_masks"] for c in names))
        doc["per_reference"][ref] = {
            "tokens_b": REFS_B.get(ref), "n_constructs": len(per), "n_masks": nm,
            "floor_p": 1 / (1 + nm),
            "holm_first_threshold": ALPHA / len(per),
            "floor_can_clear_holm": (1 / (1 + nm)) <= ALPHA / len(per),
            "n_exceeding_every_mask": len(sweep), "constructs_exceeding_every_mask": sweep,
            "n_holm_significant": int(keep.sum()),
            "constructs_holm_significant": [c for c, k in zip(names, keep) if k],
            "median_rank_from_top": float(np.median(
                [per[c]["current"]["rank_from_top"] for c in names])),
            "n_p_below_0.05_uncorrected": int((pvals < 0.05).sum()),
            "old_set": ({
                "n_constructs": len(old_names),
                "n_exceeding_every_mask": sum(
                    1 for c in old_names if per[c]["old_set"]["exceeds_every_mask"]),
                "constructs_exceeding_every_mask": [
                    c for c in old_names if per[c]["old_set"]["exceeds_every_mask"]],
                "n_holm_significant": sum(
                    1 for c in old_names if per[c]["old_set"]["holm_significant"]),
                "median_rank_from_top": float(np.median(
                    [per[c]["old_set"]["rank_from_top"] for c in old_names])),
                "n_p_below_0.05_uncorrected": int(sum(
                    1 for c in old_names
                    if per[c]["old_set"]["exact_one_sided_p"] < 0.05)),
            } if old_names else None),
            "per_construct": per}
        for c in names:
            e = per[c]
            rows.append({
                "reference": ref, "tokens_b": REFS_B.get(ref), "construct": c,
                "rise_over_random_nats": e["rise_over_random_nats"],
                "carrier_rise_nats": e["carrier_rise_nats"],
                "control_mean_nats": e["control_mean_nats"],
                "rank_from_top": e["current"]["rank_from_top"],
                "n_masks": e["current"]["n_masks"],
                "exact_one_sided_p": e["current"]["exact_one_sided_p"],
                "exceeds_every_mask": e["current"]["exceeds_every_mask"],
                "holm_significant": e["current"]["holm_significant"],
                "old_set_rise_nats": e.get("old_set_rise_nats"),
                "old_set_rank_from_top": (e.get("old_set") or {}).get("rank_from_top"),
                "old_set_p": (e.get("old_set") or {}).get("exact_one_sided_p"),
                "old_set_holm_significant": (e.get("old_set") or {}).get(
                    "holm_significant")})

    if missing:
        raise SystemExit(
            f"REFUSED: {len(missing)} construct-references carry no `random_per_mask`, so no "
            f"rank can be taken (first: {missing[0]}). Rerun with "
            f"notebooks/colab_ablation_ranks.ipynb.")
    if len(counts) > 1:
        raise SystemExit(f"REFUSED: mask counts differ across constructs: {sorted(counts)}")
    # A 200-mask run cannot satisfy A1.1b's third clause: its floor p of 1/201 = 0.00498
    # exceeds Holm's first threshold of 0.05/16, so M is 0 by arithmetic whatever the data
    # do. That is the exact failure an OLD copy of the notebook produces -- cell 1's git pull
    # updates the repo, not the notebook being executed -- so it is stopped here rather than
    # reported as a result. --allow-vacuous-holm says the zeros are wanted anyway.
    n_masks = max(counts) if counts else 0
    n_tests = max((e["n_constructs"] for e in doc["per_reference"].values()), default=16)
    if not args.allow_vacuous_holm and (1 / (1 + n_masks)) > ALPHA / n_tests:
        raise SystemExit(
            f"REFUSED: {n_masks} masks give a floor p of {1 / (1 + n_masks):.5f}, above Holm's "
            f"first threshold of {ALPHA}/{n_tests} = {ALPHA / n_tests:.5f}. No construct can "
            f"clear 0.05 however the data fall, so M would be 0 by arithmetic, not evidence. "
            f"This is what an old copy of the notebook produces: it runs 200 masks, and cell "
            f"1's git pull updates the repo, not the notebook. Rerun with at least "
            f"{int(np.ceil(n_tests / ALPHA)) - 1} masks (the notebook's default is 320), or "
            f"pass --allow-vacuous-holm if the zeros are wanted.")

    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance([args.inp.rstrip("/").split("/")[-1] + "/FULL_*.json"])

    # the clause, in the form A1.1b fixes. The numbers fill in; the sentence does not change.
    g = [doc["per_reference"].get(r, {}) for r in ORDER]
    N = [e.get("n_exceeding_every_mask") for e in g]
    M = [e.get("n_holm_significant") for e in g]
    nm = next((e["n_masks"] for e in g if e), None)
    n_c = next((e["n_constructs"] for e in g if e), 16)
    clause = (
        f"Taken one at a time against {nm} random sets, {N[0]} of {n_c} constructs exceed every "
        f"random set at 4.19 B tokens, {N[1]} at 84 B and {N[2]} at the end, and {M[0]}, {M[1]}, "
        f"{M[2]} remain clear at 0.05 after a Holm correction; the floor of 1/{(nm or 0) + 1} is "
        f"the resolution a {nm}-set control allows, and the paper's claim is the one over "
        f"constructs.")
    doc["clause_for_paper2_4_3_and_paper1_4_8"] = clause
    (OUT2 / "ABLATION_RANKS.json").write_text(json.dumps(doc, indent=1, default=float))

    # Table D.1, A1.1b form: control units and the settled flag are gone
    cols = ["reference", "tokens_b", "construct", "rise_over_random_nats", "carrier_rise_nats",
            "control_mean_nats", "rank_from_top", "n_masks", "exact_one_sided_p",
            "exceeds_every_mask", "holm_significant", "old_set_rise_nats",
            "old_set_rank_from_top", "old_set_p", "old_set_holm_significant"]
    head = (
        f"**Table D.1 — the ablation at three references, in nats, with each construct's rank "
        f"among {nm} random sets.** `rise_over_random_nats` is the carriers' loss rise minus "
        f"the mean rise over the size-matched random sets, signed, at the construct's own "
        f"positions; it is the unit the papers' bodies use. `rank_from_top` is 1 plus the "
        f"number of random sets at or above the carriers, and `exact_one_sided_p` is "
        f"(1 + that number) / (1 + {nm}); `holm_significant` applies Holm-Bonferroni at 0.05 "
        f"across the {n_c} constructs at each reference. The smallest p this control allows is "
        f"1/{(nm or 0) + 1} = {1 / ((nm or 0) + 1):.5f}. Control units are not reported: the "
        f"control's spread is long-tailed at the two later references and does not settle with "
        f"more draws, so it has no converged denominator. `old_set_*` columns are the 84 B "
        f"carrier set removed at the end, against the same masks.")
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

    for ref in ORDER:
        e = doc["per_reference"].get(ref)
        if not e:
            continue
        print(f"  {ref[-7:]} ({e['tokens_b']:7.1f} B): "
              f"{e['n_exceeding_every_mask']}/{e['n_constructs']} exceed every mask, "
              f"{e['n_holm_significant']} clear Holm, median rank "
              f"{e['median_rank_from_top']:.0f} of {e['n_masks']}")
        o = e.get("old_set")
        if o:
            print(f"                      84 B set: "
                  f"{o['n_exceeding_every_mask']}/{o['n_constructs']} exceed every mask, "
                  f"{o['n_holm_significant']} clear Holm, median rank "
                  f"{o['median_rank_from_top']:.0f} of {e['n_masks']}")
    if g and g[0]:
        ok = g[0]["floor_can_clear_holm"]
        print(f"\n  floor p = {g[0]['floor_p']:.5f}, Holm's first threshold = "
              f"{g[0]['holm_first_threshold']:.5f} -> a clean sweep "
              f"{'CAN' if ok else 'CANNOT'} clear Holm at this mask count")
    print(f"\n  {clause}")
    print(f"-> {OUT2 / 'ABLATION_RANKS.json'}, TABLE_D1_ablation.{{md,csv}}")


if __name__ == "__main__":
    main()
