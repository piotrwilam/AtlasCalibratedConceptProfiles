r"""A1.1 — the ablation unit, with the control's own spread shown to have settled.

Two draws of the random control gave **+5.16 and +3.7 control units** at the end of training for
the same quantity. A number that moves by a third between draws of its own denominator cannot
carry an abstract, so the control is re-estimated until its standard deviation stops moving:
masks are drawn **once at 200**, and the SD is read at **50, 100 and 200** from nested prefixes
of that one set. The settle criterion is the item's: the SD changes by under 5% when the mask
count doubles.

Reading nested prefixes rather than running three passes costs 200 forward passes a cell instead
of 350, and makes the three figures differ only in sample size -- not in which masks were drawn,
which a three-run design would confound with the thing being measured.

Run `ablation_full.py --n-random 200` first (it needs CUDA); this reads its output.

    python experiments/ablation_stable.py
"""
from __future__ import annotations

import csv
import glob
import json
import re
import time

import numpy as np

from experiments.ablation_full import NESTED_AT
from experiments.atlas import D, provenance
from experiments.relatedness import OUT2

IN = D / "validation/ablation"
REFS_B = {"pretrain_iter_0001000": 4.19, "pretrain_iter_0020000": 83.9,
          "pretrain_iter_0484865": 2034.0}


def main() -> None:
    t0 = time.time()
    doc = {"what": "the ablation effect with the control's spread shown to have settled",
           "definition": {
               "effect_nats": "the rise in loss at the construct's positions when its carrier "
                              "set is zeroed, over the unablated baseline",
               "control": "size-matched random sets, drawn once at the largest mask count",
               "effect_control_units": "(effect - control mean) / control SD, SIGNED",
               "nested_prefixes": list(NESTED_AT),
               "settle_criterion": "the control SD changes by < 5% when the mask count doubles",
               "also_reported": "the 84 B set removed at the end, under the SAME masks"},
           "fixed_before_computing": True, "per_reference": {}}
    rows = []
    for f in sorted(glob.glob(str(IN / "FULL_*.json"))):
        a = json.loads(open(f).read())
        ref = a["reference"]
        cs = a["constructs"]
        per = {}
        for c, v in sorted(cs.items()):
            if v.get("current_over_random_in_sd") is None:
                continue
            st = v.get("control_stability") or {}
            per[c] = {
                "loss_rise_nats": v.get("current"),
                "control_mean_nats": v.get("random_mean"),
                "control_sd_nats": v.get("random_sd"),
                "effect_control_units": v.get("current_over_random_in_sd"),
                "old_set_rise_nats": v.get("drifted"),
                "old_set_control_units": v.get("drifted_over_random_in_sd"),
                "overlap_share": v.get("overlap_share"),
                "n_masks_reached": v.get("n_masks_reached", a.get("n_random")),
                "control_sd_settled": v.get("control_sd_settled"),
                "control_stability": st}
            rows.append({"reference": ref, "tokens_b": REFS_B.get(ref), "construct": c,
                         **per[c]})
        def med(k):
            x = [e[k] for e in per.values() if isinstance(e.get(k), (int, float))]
            return float(np.median(x)) if x else None
        nm = [e["n_masks_reached"] for e in per.values() if e.get("n_masks_reached")]
        settled = [e["control_sd_settled"] for e in per.values()
                   if e.get("control_sd_settled") is not None]
        doc["per_reference"][ref] = {
            "tokens_b": REFS_B.get(ref), "n_constructs": len(per),
            "median_loss_rise_nats": med("loss_rise_nats"),
            "median_control_mean_nats": med("control_mean_nats"),
            "median_control_sd_nats": med("control_sd_nats"),
            "median_effect_control_units": med("effect_control_units"),
            "median_old_set_control_units": med("old_set_control_units"),
            "n_masks_reached": (int(max(nm)) if nm else None),
            "n_constructs_with_settled_control": int(sum(settled)) if settled else None,
            "per_construct": per}
    # the item IS the settled control, so a run without one is a stop, not a table with
    # blanks in it. The first version wrote a table from the old 20-mask output with
    # `control_sd_settled: None` in every row, which looks like a result and is not one.
    n_stab = sum(1 for r in rows if r.get("control_stability"))
    if rows and n_stab == 0:
        raise SystemExit(
            f"REFUSED: {len(rows)} rows found but NONE carries a control_stability block, so "
            f"the control's spread has not been shown to settle -- which is the whole of "
            f"A1.1. The files on disk are the 20-mask run of 22 Oct.\n"
            f"Run first, on a CUDA machine:\n"
            f"  python experiments/ablation_full.py --n-random 200\n"
            f"then place its FULL_*.json under {IN}")
    if not rows:
        raise SystemExit(
            "no FULL_*.json with a control_stability block was found.\n"
            "Run first, on a CUDA machine:\n"
            "  python experiments/ablation_full.py --n-random 200 --all-constructs\n"
            "then place its output under $ATLAS_DATA/validation/ablation/")
    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance([str(IN.relative_to(D)) + "/FULL_*.json"])
    (OUT2 / "ABLATION_STABLE.json").write_text(json.dumps(doc, indent=1, default=float))

    cols = ["reference", "tokens_b", "construct", "loss_rise_nats", "control_mean_nats",
            "control_sd_nats", "effect_control_units", "old_set_control_units",
            "n_masks_reached", "control_sd_settled"]
    head = ("**Table D.1 — the ablation at three references, with the control's spread shown "
            "to have settled.** Effect in control units is (rise − control mean) / control "
            f"SD, signed. Masks are drawn once at {max(NESTED_AT)} and the SD read at "
            f"{', '.join(map(str, NESTED_AT))} from nested prefixes; `control_sd_settled` is "
            "true where the SD changed by under 5% on the last doubling.")
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
        print(f"  {ref[-7:]} ({e['tokens_b']:7.1f} B): median effect "
              f"{e['median_effect_control_units']:.2f} control units, "
              f"{e['n_masks_reached']} masks, settled for "
              f"{e['n_constructs_with_settled_control']}/{e['n_constructs']}")
    print(f"-> {OUT2 / 'ABLATION_STABLE.json'}, TABLE_D1_ablation.{{md,csv}}")


if __name__ == "__main__":
    main()
