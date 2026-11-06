r"""Item 10 -- accuracy and AUC against the probe's MARGIN, across the four references.

Accuracy and AUC are capped: once a probe separates the two sets perfectly they both read 1.000
and stop carrying information, however far apart the sets go on. The **margin** -- the signed
distance from the decision boundary, in the probe's own units -- is not capped, so it can keep
moving after the two capped measures have stopped. Whether it does is what section 4.3 of paper 1
needs to know, and it is a question about the measure, not about the model: a margin that also
flattens says the separation itself has settled.

Per construct at each of the four references: accuracy, AUC, mean margin, and **the control's
margin beside each**, because a margin has no natural scale and only means something against the
control probe trained on shuffled labels. The ceiling share counts constructs at accuracy >= 0.99
and AUC >= 0.99.

Reads PROBE.json, which `b9_b10.py collect` writes.

    python experiments/probe_margins.py
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from experiments.atlas import D, provenance

OUT = D / "validation/paper1"
SRC = OUT / "PROBE.json"
REFS = ("untrained", "pretrain_iter_0001000", "pretrain_iter_0020000", "pretrain_iter_0484865")
REFS_B = {"untrained": 0.0, "pretrain_iter_0001000": 4.19,
          "pretrain_iter_0020000": 83.9, "pretrain_iter_0484865": 2034.0}
TRAINED = REFS[1:]
#: the probe is keyed `<space>|<layer rule>|L<n>` and the layer differs by construct, so the
#: space alone does not name a probe. The verdict layer is the one the rest of the programme
#: reads; `peak_norm` is carried beside it as the secondary.
SPACE = "resid_2240|verdict"
SPACE_2 = "resid_2240|peak_norm"
CEIL = 0.99


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return float(np.median(xs)) if xs else None


def main() -> None:
    t0 = time.time()
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}; run `python -m experiments.b9_b10 pass1` first")
    src = json.loads(SRC.read_text())
    rows_in = src["rows"]

    doc = {
        "what": "item 10: accuracy, AUC and the probe's mean margin at the four references",
        "why": "accuracy and AUC are capped at 1.000, so they stop carrying information once "
               "the probe separates the sets perfectly; the margin is not capped and can keep "
               "moving. Whether it does is what section 4.3 asks.",
        "definition": {
            "space": SPACE,
            "accuracy": "held-out accuracy, eight folds held out by core, mean over folds",
            "auc": "held-out ROC AUC, same folds",
            "mean_margin": "mean signed distance from the decision boundary over held-out rows, "
                           "in the probe's own units; positive means on the correct side",
            "control_mean_margin": "the same quantity for the control probe, which is what "
                                   "gives the margin a scale",
            "share_on_the_right_side": "fraction of held-out rows with a positive margin",
            "ceiling": f"accuracy >= {CEIL} and AUC >= {CEIL}",
            "references": list(REFS)},
        "fixed_before_computing": True, "per_reference": {}, "per_construct": {}}

    table = []
    for ref in REFS:
        acc, auc, mar, cmar, cacc = [], [], [], [], []
        for key, r in rows_in.items():
            rf, c = key.split("|", 1)
            if rf != ref:
                continue
            pr = r.get("probes") or {}
            kk = [k for k in pr if k.startswith(SPACE)]
            if not kk:
                continue
            p = pr[sorted(kk)[0]]
            doc.setdefault("probe_keys_seen", {})[sorted(kk)[0]] = ref
            acc.append(p.get("accuracy"))
            auc.append(p.get("auc"))
            mar.append(p.get("mean_margin"))
            cmar.append(p.get("control_mean_margin"))
            cacc.append(p.get("control_accuracy"))
            doc["per_construct"].setdefault(c, {})[ref] = {
                "accuracy": p.get("accuracy"), "auc": p.get("auc"),
                "mean_margin": p.get("mean_margin"),
                "control_mean_margin": p.get("control_mean_margin"),
                "control_accuracy": p.get("control_accuracy"),
                "control_auc": p.get("control_auc"),
                "share_on_the_right_side": p.get("share_on_the_right_side")}
            table.append({"reference": ref, "tokens_b": REFS_B[ref], "construct": c,
                          "accuracy": p.get("accuracy"), "auc": p.get("auc"),
                          "mean_margin": p.get("mean_margin"),
                          "control_mean_margin": p.get("control_mean_margin"),
                          "control_accuracy": p.get("control_accuracy"),
                          "share_on_the_right_side": p.get("share_on_the_right_side")})
        if not acc:
            continue
        at_ceiling = sum(1 for a, u in zip(acc, auc)
                         if a is not None and u is not None and a >= CEIL and u >= CEIL)
        doc["per_reference"][ref] = {
            "tokens_b": REFS_B[ref], "n_constructs": len(acc),
            "median_accuracy": _med(acc), "median_auc": _med(auc),
            "median_mean_margin": _med(mar), "median_control_mean_margin": _med(cmar),
            "median_control_accuracy": _med(cacc),
            "n_at_ceiling": at_ceiling, "share_at_ceiling": at_ceiling / len(acc),
            "median_margin_over_control": (
                None if not _med(cmar) else _med(mar) / _med(cmar))}

    # the margin is the whole item, so a run that has none is a stop, not a table of blanks.
    # The PROBE.json of 18 Oct predates `mean_margin` in probe(); reading it here gave a
    # TypeError on a None, which is a worse way to find out than being told.
    n_mar = sum(1 for r in table if r.get("mean_margin") is not None)
    if table and n_mar == 0:
        raise SystemExit(
            f"REFUSED: {len(table)} probe rows and NOT ONE carries `mean_margin`. This "
            f"PROBE.json predates the margin being added to probe(); rerun "
            f"`python -m experiments.b9_b10 pass1 --force`.")
    if table and n_mar < len(table):
        raise SystemExit(
            f"REFUSED: {len(table) - n_mar} of {len(table)} rows carry no `mean_margin`, so "
            f"the medians would be taken over a mixed population.")

    # does the margin still move where accuracy and AUC do not?
    tr = [doc["per_reference"][r] for r in TRAINED if r in doc["per_reference"]]
    if len(tr) >= 2:
        def spread(k):
            v = [e[k] for e in tr if e.get(k) is not None]
            return (max(v) - min(v)) if v else None
        a_sp, u_sp, m_sp = spread("median_accuracy"), spread("median_auc"), \
            spread("median_mean_margin")
        first, last = tr[0]["median_mean_margin"], tr[-1]["median_mean_margin"]
        growth = (None if not first else (last - first) / abs(first))
        capped = (a_sp is not None and a_sp < 0.01) and (u_sp is not None and u_sp < 0.01)
        # FIRST-TO-LAST HIDES A TURN. The margin rises to 84 B and falls back, so the
        # end-to-end change (+7%) is a quarter of the actual excursion (19% of the first).
        # Each leg is therefore tested on the SAME constructs, paired, which is also what
        # says whether the turn is systematic or sampling.
        from scipy.stats import binomtest, wilcoxon
        names = [c for c in doc["per_construct"]
                 if all(r in doc["per_construct"][c]
                        and doc["per_construct"][c][r]["mean_margin"] is not None
                        for r in TRAINED)]
        M = np.array([[doc["per_construct"][c][r]["mean_margin"] for r in TRAINED]
                      for c in names], float)
        legs = {}
        for ia, ib, lab in ((0, 1, "4.19B_to_83.9B"), (1, 2, "83.9B_to_end"),
                            (0, 2, "4.19B_to_end")):
            dl = M[:, ib] - M[:, ia]
            k = int((dl > 0).sum())
            legs[lab] = {"n": int(dl.size), "median_change": float(np.median(dl)),
                         "n_up": k, "n_down": int(dl.size - k),
                         "sign_test_p": float(binomtest(k, dl.size, 0.5).pvalue),
                         "wilcoxon_p": float(wilcoxon(dl).pvalue)}
        doc["margin_by_leg_paired"] = legs
        turns = legs["4.19B_to_83.9B"]["median_change"] > 0 > \
            legs["83.9B_to_end"]["median_change"]
        doc["across_the_trained_references"] = {
            "median_accuracy_range": a_sp, "median_auc_range": u_sp,
            "median_margin_range": m_sp,
            "median_margin_first_to_last": [first, last],
            "margin_change_as_a_fraction_of_the_first": growth,
            "accuracy_and_auc_are_flat": bool(capped),
            "margin_turns": bool(turns),
            "sentence": (
                f"Across the three trained references the median accuracy moves by {a_sp:.3f} "
                f"and the median AUC by {u_sp:.3f} -- both pinned at the ceiling -- while the "
                f"median margin moves over a range of {m_sp:.3f}, "
                f"{m_sp / abs(first):.0%} of its value at 4.19 B. "
                + (f"It does not move in one direction: it rises to 83.9 B for "
                   f"{legs['4.19B_to_83.9B']['n_up']} of "
                   f"{legs['4.19B_to_83.9B']['n']} constructs "
                   f"(p = {legs['4.19B_to_83.9B']['sign_test_p']:.1g}) and falls again to the "
                   f"end for {legs['83.9B_to_end']['n_down']} of "
                   f"{legs['83.9B_to_end']['n']} (p = "
                   f"{legs['83.9B_to_end']['sign_test_p']:.1g}), leaving it "
                   f"{abs(growth):.0%} {'above' if growth > 0 else 'below'} its starting value. "
                   if turns else
                   f"End to end it moves {abs(growth or 0):.0%} "
                   f"{'up' if (growth or 0) > 0 else 'down'}. ")
                + ("So the margin still moves where accuracy and AUC do not."
                   if capped and m_sp and m_sp / abs(first) > 0.1 else
                   "So the margin does not carry a movement the capped measures miss."))}

    # a table of no rows is not a result. The first version of this printed an empty header
    # and exited 0 because the probe is keyed `<space>|<rule>|L<n>`, not `<space>`.
    if not table:
        raise SystemExit(
            f"REFUSED: PROBE.json holds {len(rows_in)} rows but none carries a probe whose key "
            f"starts with {SPACE!r}. Keys seen on the first row: "
            f"{sorted((list(rows_in.values())[0].get('probes') or {}))}")
    missing = [r for r in REFS if r not in doc["per_reference"]]
    if missing:
        raise SystemExit(
            f"REFUSED: no rows at {missing} -- the pass has not finished all four references, "
            f"so the across-references reading would be taken on a partial run.")
    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance(["validation/paper1/PROBE.json"])
    (OUT / "PROBE_MARGINS.json").write_text(json.dumps(doc, indent=1, default=float))

    cols = ["reference", "tokens_b", "construct", "accuracy", "auc", "mean_margin",
            "control_mean_margin", "control_accuracy", "share_on_the_right_side"]
    head = ("**Table — the probe at four references.** Accuracy and AUC are held out over eight "
            "folds, held out by core. The margin is the mean signed distance from the decision "
            f"boundary in the probe's own units, with the control probe's margin beside it; a "
            f"construct is at the ceiling when accuracy and AUC both reach {CEIL}.")
    md = [head, "", "| " + " | ".join(cols) + " |",
          "|" + "|".join("---" for _ in cols) + "|"]
    for r in table:
        md.append("| " + " | ".join(
            "—" if r.get(c) is None else
            (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    (OUT / "TABLE_probe_margins.md").write_text("\n".join(md) + "\n")
    with open(OUT / "TABLE_probe_margins.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows([[r.get(c) for c in cols] for r in table])

    print(f"  {'reference':<22s} {'n':>4s} {'acc':>7s} {'AUC':>7s} {'margin':>9s} "
          f"{'control':>9s} {'at ceiling':>12s}")
    for ref in REFS:
        e = doc["per_reference"].get(ref)
        if not e:
            continue
        print(f"  {ref:<22s} {e['n_constructs']:4d} {e['median_accuracy']:7.3f} "
              f"{e['median_auc']:7.3f} {e['median_mean_margin']:9.3f} "
              f"{e['median_control_mean_margin']:9.3f} "
              f"{e['n_at_ceiling']:5d}/{e['n_constructs']:<3d} "
              f"{e['share_at_ceiling']:.0%}")
    if "across_the_trained_references" in doc:
        print(f"\n  {doc['across_the_trained_references']['sentence']}")
    print(f"-> {OUT / 'PROBE_MARGINS.json'}, TABLE_probe_margins.{{md,csv}}")


if __name__ == "__main__":
    main()
