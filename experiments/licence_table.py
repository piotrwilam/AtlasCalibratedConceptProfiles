r"""The per-construct licence table, with the four re-run checks folded in.

`LICENCE_TABLE.json` (B1) holds all 188 constructs as the suite recorded them. Four of its
checks -- 2 convergence, 3 length, 7a null scrambled, 7b null split half -- paired per-prompt
vectors with prompt rows in jsonl order while the arrays are stored length-sorted, so each
vector carried another prompt's core, cell and padding level (A51, Section 10). They were re-run
in the recorder's order for the 31 constructs that have per-prompt hidden vectors.

This table carries **both**: the published value, the corrected value, and the change, per
construct. A reader of the paper should be able to see which rows moved and by how much, rather
than find a silently different number from the one the earlier draft quoted.

    python experiments/licence_table.py
"""
from __future__ import annotations

import json

import numpy as np

from experiments.atlas import LAST, OUT, fetch, provenance


def first_n_at(curve, thr=0.99):
    for p in curve or []:
        c = p.get("cosine")
        if c is not None and c >= thr:
            return p.get("n")
    return None


def main() -> None:
    pub = json.loads((OUT / "LICENCE_TABLE.json").read_text())
    new = json.loads((OUT / "RECHECK_PER_PROMPT.json").read_text())["rows"]
    corrected = {k.split("|")[1]: v for k, v in new.items() if k.startswith(LAST)}
    rows, moved = {}, {c: 0 for c in ("2 convergence", "3 length",
                                      "7a null_scrambled", "7b null_split_half")}
    for name, r in pub["constructs"].items():
        tn = r.get("tree_name") or name
        e = {k: r[k] for k in ("construct", "tree_name", "in_p1", "member_kind",
                               "contrast_class", "a_family", "cores", "n_a", "n_b",
                               "gate_failed", "profile_source") if k in r}
        e["checks_published"] = {k: r[k] for k in r if k.startswith("check")}
        c = corrected.get(tn)
        e["per_prompt_recheck"] = bool(c)
        if c:
            conv, leng = c["2 convergence"], c["3 length"]
            e["checks_corrected"] = {
                "check2_convergence": {
                    "n_full": conv.get("n_full"),
                    "n_for_cosine_0.99": first_n_at(conv.get("curve")),
                    "final_cosine": (conv.get("curve") or [{}])[-1].get("cosine")},
                "check3_length": {"index_median": leng.get("index_median"),
                                  "flagged_share": leng.get("flagged_share"),
                                  "min_pair_cosine": min(
                                      [p["cosine"] for p in leng.get("pairs") or []
                                       if p.get("cosine") is not None], default=None)},
                "check8_nulls__suite_7": {
                    "scrambled": {k: c["7a null_scrambled"].get(k)
                                  for k in ("median_norm", "p95_norm", "layer", "n_rep")},
                    "split_half": {k: c["7b null_split_half"].get(k)
                                   for k in ("median_norm", "p95_norm", "layer", "n_rep")}}}
            d = {}
            p3 = (r.get("check3_length") or {})
            for lab, a, b in (("check3_min_pair_cosine", p3.get("min_pair_cosine"),
                               e["checks_corrected"]["check3_length"]["min_pair_cosine"]),
                              ("check8_scrambled_median_norm",
                               ((r.get("check8_nulls__suite_7") or {}).get("scrambled")
                                or {}).get("median_norm"),
                               c["7a null_scrambled"].get("median_norm")),
                              ("check8_split_half_median_norm",
                               ((r.get("check8_nulls__suite_7") or {}).get("split_half")
                                or {}).get("median_norm"),
                               c["7b null_split_half"].get("median_norm"))):
                if a is not None and b is not None:
                    d[lab] = {"published": a, "corrected": b, "change": b - a}
            e["changes"] = d
            for k in moved:
                moved[k] += 1
        rows[name] = e
    n_rc = sum(1 for v in rows.values() if v["per_prompt_recheck"])
    out = {"what": "the per-construct licence table, with the four per-prompt checks re-run in "
                   "the recorder's order and both values carried",
           "reference": LAST, "n_constructs": len(rows),
           "n_with_per_prompt_recheck": n_rc,
           "re_run_checks": sorted(moved),
           "why": "the four re-run checks paired per-prompt vectors with prompt rows in jsonl "
                  "order while the arrays are stored length-sorted (A51); the other checks read "
                  "the per-core accumulators and are order-independent",
           "constructs": rows,
           "provenance": provenance(["validation/paper1/LICENCE_TABLE.json",
                                     "validation/paper1/RECHECK_PER_PROMPT.json"],
                                    reference=LAST)}
    (OUT / "LICENCE_TABLE_CORRECTED.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"{len(rows)} constructs; {n_rc} carry corrected per-prompt checks")
    for lab in ("check3_min_pair_cosine", "check8_scrambled_median_norm",
                "check8_split_half_median_norm"):
        ch = [v["changes"][lab] for v in rows.values() if v.get("changes", {}).get(lab)]
        if ch:
            dd = [x["change"] for x in ch]
            print(f"  {lab:32s} n={len(ch):2d}  median published "
                  f"{np.median([x['published'] for x in ch]):7.4f} -> corrected "
                  f"{np.median([x['corrected'] for x in ch]):7.4f}  "
                  f"(median change {np.median(dd):+.4f})")
    print(f"-> {OUT / 'LICENCE_TABLE_CORRECTED.json'}")


if __name__ == "__main__":
    main()
