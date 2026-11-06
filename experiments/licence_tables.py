r"""Appendix B's three licence tables, rendered (paper 1).

`LICENCE_TABLE_CORRECTED.json` (21 Oct) holds every value: the suite as published, the four
per-prompt checks re-run in the recorder's order for the 31 constructs with per-prompt hidden
vectors, and the change per row. Nothing is recomputed here except two things the table needs
and the artefact does not carry: the **gate's own share** (the modal anchor-piece share, from
the battery) and the **layer-matched real norm** that turns checks 7a and 7b from a null level
into the ratio the paper quotes.

**Three tables, because the checks have three different populations** -- a single 188-row table
was four-ninths dashes:

  * **B.1**, all 188 rows: the gate and checks 1, 4, 5, 8, which the suite ran everywhere.
  * **B.2**, the 31 with per-prompt vectors: checks 2, 3-index, 3-flagged, 7a, 7b, as
    published / corrected / change. The other 157 are `profile_source: accumulators` and these
    four checks cannot be computed without per-prompt vectors.
  * **B.3**, the 8 constructs that can have a companion: check 6.

**Two numbering systems meet here and they disagree.** The suite's own numbering is in the JSON
keys, and the keys carry the OTHER system's number as a suffix: `check7_sigma0_floor__suite_8_eps`
holds the paper's check **8** and `check8_nulls__suite_7` holds the paper's **7a** and **7b**.
The map is read off the docstrings in `profilebattery/validation/checks.py`, never off the keys.

    python experiments/licence_tables.py
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from experiments.atlas import D, provenance
from experiments.build_battery import BUILTIN_STEM_SUFFIX
from experiments.relatedness import OUT2

OUT = D / "validation/paper1"
REF = "pretrain_iter_0484865"
TOKENS_B = 2034
NULL_LAYER = 12
WIDTH = 6144

B1_COLS = [("1", "twin agreement (median cosine)", 4),
           ("4", "re-weighting (alt-arm cosine)", 4),
           ("5", "calibration inflation", 2),
           ("8", "sigma_0 floor (twin units)", 4)]
B2_COLS = [("2", "convergence (final cosine)", 4),
           ("3 idx", "length index (median)", 2),
           ("3 flag", "flagged share", 3),
           ("7a", "real norm / scrambled p95", 2),
           ("7b", "real norm / split-half p95", 2)]


#: the gate's pass criterion, from section 3.1
GATE = 0.80
#: piece_shares() returns the four shares in the order it reads them
SHARE_ORDER = ("twin1_A", "twin1_B", "twin2_A", "twin2_B")


def gate_shares_full() -> dict:
    """Per construct, the four modal anchor-piece shares and their minimum.

    `piece_shares` keys by the construct a battery stem names, which strips the `__builtin`
    suffix: `dict` holds the builtin sense's four shares and `Dict` the grammar node's.
    """
    from experiments.split_p1 import piece_shares
    out = {}
    for c, v in piece_shares().items():
        if len(v) != 4:
            continue
        out[c] = {**dict(zip(SHARE_ORDER, (float(x) for x in v))),
                  "min": float(min(v)), "passes": bool(min(v) >= GATE)}
    return out


def gate_shares() -> dict:
    return {c: v["min"] for c, v in gate_shares_full().items()}


#: check 2's threshold, and the grid the curve is sampled on. Table B.1 quotes the prompt
#: count at which the cosine first reaches COS99, so the count itself belongs in the released
#: per-construct file rather than only the half-sample cosine (Settle chat, 5 Nov).
COS99 = 0.99


def convergence_counts() -> tuple[dict, list]:
    """Per construct: the first grid point whose cosine reaches 0.99, and the grid."""
    doc = json.loads((D / "validation/paper1/RECHECK_PER_PROMPT.json").read_text())
    out, grid = {}, []
    for key, v in doc["rows"].items():
        ref, c = key.split("|")
        if ref != REF:
            continue
        curve = (v.get("2 convergence") or {}).get("curve") or []
        if not curve:
            continue
        grid = [pt["n"] for pt in curve]
        out[c] = next((pt["n"] for pt in curve if pt["cosine"] >= COS99), None)
    return out, grid


def real_norms(constructs) -> dict:
    """Layer-12 norm of each construct's concept profile at REF.

    This globbed a local cache and returned whatever happened to be in it. The cache is
    `validation/paper2/concept_cache`, which is derived and was cleared: the glob then
    returned nothing, no error was raised, and B.2's four null columns printed "---" for all
    31 constructs (found 5 Nov after a re-run put that silence into the released CSV). It now
    asks `concept_store.concept_of`, which fetches the member it needs and caches it, and
    refuses rather than return a short dict.
    """
    from experiments.concept_store import concept_of
    # fail fast rather than retry twenty times per construct: on a clean machine the first
    # fetch is already the answer, and the old path spent six and a half minutes before
    # saying so (clean-environment run, 5 Nov).
    from profilebattery.profiles import resolve
    try:
        resolve.profile(REF, "twin1", constructs[0], ("concept",))
    except Exception as exc:                                        # noqa: BLE001
        raise SystemExit(
            f"licence_tables: cannot read profiles/{REF}/twin1/{constructs[0]}.npz "
            f"({type(exc).__name__}). Table B.1's two null ratios need the profile of each "
            f"of the {len(constructs)} per-prompt constructs, and the public dataset carries "
            f"one worked example; the ratios themselves are released in NULL_RATIOS.json. "
            f"Nothing written.") from None
    out = {}
    for c in constructs:
        v = concept_of(REF, "twin1", c)
        out[c] = float(np.linalg.norm(np.asarray(v).reshape(24, WIDTH)[NULL_LAYER]))
    missing = [c for c in constructs if c not in out]
    if missing:
        raise SystemExit(f"licence_tables: no concept profile for {missing}")
    return out


def emit(name, head, cols, rows, fmts, tex_align=None, tex_wanted=True):
    """One table in markdown and CSV, and in LaTeX unless `tex` is False.

    `rows` are lists of already-formatted strings. B.1 and B.2 are no longer printed in the
    paper -- Appendix B gives each check's spread instead, and the per-construct values are
    the released CSVs -- so those two emit no .tex.
    """
    md = [head, "", "| " + " | ".join(cols) + " |",
          "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    al = tex_align or ("l" + "r" * (len(cols) - 1))
    cap = head.split("**")[1] if "**" in head else name
    tex = [r"\begin{longtable}{" + al + "}",
           r"\caption{" + cap.replace("&", r"\&").replace("—", "---") + r"}\\",
           r"\toprule",
           " & ".join(c.replace("Δ", r"$\Delta$").replace("_", r"\_") for c in cols) + r" \\",
           r"\midrule", r"\endfirsthead", r"\toprule",
           " & ".join(c.replace("Δ", r"$\Delta$").replace("_", r"\_") for c in cols) + r" \\",
           r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for r in rows:
        tex.append(" & ".join(x.replace("_", r"\_").replace("—", "---")
                              .replace("†", r"$\dagger$") for x in r) + r" \\")
    tex.append(r"\end{longtable}")
    if not tex_wanted:
        return [f"{name}.md", f"{name}.csv"]
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n")
    return [f"{name}.md", f"{name}.csv", f"{name}.tex"]


def main() -> None:
    t0 = time.time()
    doc = json.loads((OUT / "LICENCE_TABLE_CORRECTED.json").read_text())
    C = doc["constructs"]
    gs_full = gate_shares_full()
    gs = {c: v["min"] for c, v in gs_full.items()}
    cc, cgrid = convergence_counts()
    print(f"{len(C)} constructs; gate shares {sum(1 for v in gs.values() if v)}; "
          f"check-2 counts {sum(1 for v in cc.values() if v)}", flush=True)
    P = lambda e, k: (e.get("checks_published") or {}).get(k)
    Q = lambda e, k: (e.get("checks_corrected") or {}).get(k)

    def kind(n, e):
        # `dict__builtin` and its four siblings carry no member_kind in the licence table --
        # the suffix is there because the token is both a grammar node and a builtin name, so
        # the suffix itself says which sense this row is. Without this they printed
        # "unclassified" (Settle chat, 5 Nov).
        if str(n).endswith(BUILTIN_STEM_SUFFIX):
            return "builtin"
        return {"ast-node": "grammar", "builtin": "builtin"}.get(
            e.get("member_kind")) or "unclassified"

    def gate_share(n):
        # gate_shares() keys by construct_of_stem, which strips the suffix, so a lookup by
        # the licence table's name missed and the share printed as nothing at all.
        v = gs.get(n)
        if v is None and n.endswith(BUILTIN_STEM_SUFFIX):
            v = gs.get(n[: -len(BUILTIN_STEM_SUFFIX)])
        return v

    def num(v, dp):
        return "—" if v is None else f"{v:.{dp}f}"

    order = sorted(C, key=lambda n: (kind(n, C[n]), n))
    files = []

    # ---------- B.1 ----------
    head = (f"**Table B.1 — the licence suite at every construct.** All {len(C)} constructs, "
            f"reference `{REF}` ({TOKENS_B:,} B tokens). These four checks and the gate were "
            f"run on the whole battery; the four checks that need per-prompt vectors are "
            f"Table B.2 and the companion check is Table B.3. Sorted by kind, then name.")
    cols = ["construct", "kind", "gate"] + [c for c, _, _ in B1_COLS]
    rows = []
    for n in order:
        e = C[n]
        g = gate_share(n)
        r = [n, kind(n, e), (f"{'fail' if e.get('gate_failed') else 'pass'}"
                          + (f" ({g:.3f})" if g is not None else ""))]
        r.append(num((P(e, "check1_twin_unit") or {}).get("median_cosine"), 4))
        c4 = P(e, "check4_nu") or []
        r.append(num(c4[0].get("cosine") if c4 else None, 4))
        # the core bootstrap's factor, not the prompt SE's: checks.py's `calibration` tests
        # both "because they estimate different things and the programme uses the second",
        # and the paper's intervals are the core bootstrap's (Settle chat, 5 Nov). The two
        # medians differ: 0.9972 over the 188 against 0.9099.
        r.append(num((P(e, "check6_calibration") or {}).get("bootstrap_inflation"), 2))
        r.append(num((P(e, "check7_sigma0_floor__suite_8_eps") or {}).get(
            "max_in_twin_units"), 4))
        rows.append(r)
    files += emit("TABLE_B1", head, cols, rows, None, "lll" + "r" * len(B1_COLS),
                  tex_wanted=False)

    # ---------- B.2 ----------
    rk = [n for n in order if C[n].get("per_prompt_recheck")]
    rn = real_norms(rk)
    # the norms and the nulls' percentiles behind B.1's 7a and 7b rows, as their own
    # artefact: the profiles they come from are in the working dataset and not in the
    # release, so without this the two ratios are in no released file but this CSV.
    nulls = {}
    for n in rk:
        row = {"real_norm_layer12": rn[n]}
        for tag, key in (("scrambled", "7a"), ("split_half", "7b")):
            blk = (Q(C[n], "check8_nulls__suite_7") or {}).get(tag) or {}
            row[f"{key}_null_p95"] = blk.get("p95_norm")
            row[f"{key}_ratio"] = _ratio(rn[n], Q(C[n], "check8_nulls__suite_7"), tag)
        nulls[n] = row
    # the gate's shares as their own artefact, on Piotr's word of 5 Nov: Table B.1's gate row
    # and B.2's nineteen failures had no field to point at, only this script's `gate` column.
    # The licence table's own name is the key, so `dict__builtin` is findable under it as
    # well as under the stem name `piece_shares` uses.
    gate_doc = {}
    for n in order:
        base = (n[: -len(BUILTIN_STEM_SUFFIX)] if n.endswith(BUILTIN_STEM_SUFFIX) else n)
        e = gs_full.get(n) or gs_full.get(base)
        if e is None:
            continue
        gate_doc[n] = {**e, "stem": base if base != n else None,
                       "gate_failed_in_licence_table": bool(C[n].get("gate_failed"))}
    disagree = [n for n, v in gate_doc.items()
                if v["passes"] == v["gate_failed_in_licence_table"]]
    if disagree:
        raise SystemExit("licence_tables: the share and the licence table's gate flag "
                         f"disagree for {disagree}")
    (OUT / "GATE_SHARES.json").write_text(json.dumps(
        {"what": "the read-site gate per construct: the modal anchor-piece share in each "
                 "twin and on each side, and the minimum the gate is taken on",
         "definition": {
             "share": "the share of a construct's prompts whose read site lands on the "
                      "single most common anchor piece, over the indented scopes "
                      "(function, method) only",
             "min": "the minimum over the four (twin, side) combinations -- what the gate "
                    "compares against",
             "pass": f"min >= {GATE} (section 3.1)",
             "stem": "set where the battery's file stem differs from the licence table's "
                     "name, which is the case for the five constructs whose token is both a "
                     "grammar node and a builtin"},
         "threshold": GATE, "n": len(gate_doc),
         "n_pass": sum(1 for v in gate_doc.values() if v["passes"]),
         "n_fail": sum(1 for v in gate_doc.values() if not v["passes"]),
         "per_construct": gate_doc,
         "provenance": provenance(["phase_battery/twin{1,2}/*_{A,B}.jsonl.gz"])},
        indent=1, default=float))
    files.append("GATE_SHARES.json")
    (OUT / "NULL_RATIOS.json").write_text(json.dumps(
        {"what": f"the layer-{NULL_LAYER} norm of each construct's concept profile at {REF}, "
                 f"the two nulls' 95th percentiles after the re-run of 19 Oct 2026, and the "
                 f"ratios Table B.1's 7a and 7b rows quote",
         "reference": REF, "twin": "twin1", "layer": NULL_LAYER,
         "definition": {"ratio": "real_norm_layer12 / the null's p95_norm at the same layer",
                        "pass": "above 1"},
         "per_construct": nulls,
         "provenance": provenance([f"profiles/{REF}/twin1/<c>.npz",
                                   "validation/paper1/LICENCE_TABLE_CORRECTED.json"])},
        indent=1, default=float))
    files.append("NULL_RATIOS.json")
    head2 = (f"**Table B.2 — the four per-prompt checks, published against corrected.** The "
             f"{len(rk)} constructs with per-prompt hidden vectors; the other "
             f"{len(C)-len(rk)} are recorded as accumulators and these checks cannot be "
             f"computed for them. Re-run in the recorder's order after the per-prompt join "
             f"fault of 19 October (A51 §10): the arrays are stored length-sorted while the "
             f"prompt rows are in jsonl order, so each vector carried another prompt's core, "
             f"cell and padding level. The length index and flagged share exist only in the "
             f"corrected reading and have no published counterpart. Checks 7a and 7b are the "
             f"layer-{NULL_LAYER} real norm over the null's p95, so the change is the null's.")
    head2 += (f" The first column is check 2's own figure: the smallest number of prompts, on "
              f"the grid {', '.join(str(n) for n in cgrid[:-1])} and {cgrid[-1]}, whose "
              f"profile reaches a cosine of {COS99} with the profile on all 720 "
              f"(`RECHECK_PER_PROMPT.json :: rows.{REF}|<c>.'2 convergence'.curve`). The "
              f"half-sample cosine beside it is a different quantity: the cosine at the "
              f"largest grid point, not a count.")
    cols2 = ["construct", "kind", "2 prompts to 0.99"]
    for c, _, _ in B2_COLS:
        cols2 += [f"{c} pub", f"{c} cor", f"{c} Δ"]
    rows2 = []
    for n in rk:
        e = C[n]
        r = [n, kind(n, e), "—" if cc.get(n) is None else str(cc[n])]
        real = rn.get(n)
        spec = [
            (lambda: (P(e, "check2_convergence") or {}).get("final_cosine"),
             lambda: (Q(e, "check2_convergence") or {}).get("final_cosine"), 4),
            (lambda: None, lambda: (Q(e, "check3_length") or {}).get("index_median"), 2),
            (lambda: None, lambda: (Q(e, "check3_length") or {}).get("flagged_share"), 3),
            (lambda: _ratio(real, P(e, "check8_nulls__suite_7"), "scrambled"),
             lambda: _ratio(real, Q(e, "check8_nulls__suite_7"), "scrambled"), 2),
            (lambda: _ratio(real, P(e, "check8_nulls__suite_7"), "split_half"),
             lambda: _ratio(real, Q(e, "check8_nulls__suite_7"), "split_half"), 2)]
        for getp, getc, dp in spec:
            pub, cor = getp(), getc()
            ch = (cor - pub) if isinstance(pub, (int, float)) and isinstance(
                cor, (int, float)) else None
            r += [num(pub, dp), num(cor, dp),
                  "—" if ch is None else f"{ch:+.{dp}f}"]
        rows2.append(r)
    files += emit("TABLE_B2", head2, cols2, rows2, None,
                  "llr" + "r" * (len(cols2) - 3), tex_wanted=False)

    # ---------- B.3 ----------
    # The asymmetry columns come from CHECK6_COMPANIONS.json, not from the licence suite's
    # own `check10_leakage` block: the suite's are the pre-run, padding-blind counts, and
    # they disagree with the paper (ClassDef 1.031 / 0.625 / 16 against 1.064 / 0.388 / 33),
    # so a reader opening this CSV saw different numbers from Table B.3. One source now, and
    # a refusal rather than a silent fall back to the superseded block.
    comp = [n for n in order if (P(C[n], "check10_leakage") or {}).get("asymmetry_l2")
            is not None]
    c6doc = D / "validation/paper1/CHECK6_COMPANIONS.json"
    if not c6doc.exists():
        raise SystemExit("licence_tables: B.3 needs CHECK6_COMPANIONS.json (run "
                         "experiments/check6_companions.py first)")
    b3 = json.loads(c6doc.read_text())["table_b3"]
    asym = {r[0]: r[1:] for r in b3["rows"]}
    missing = [n for n in comp if n not in asym]
    if missing:
        raise SystemExit(f"licence_tables: CHECK6_COMPANIONS has no row for {missing}")
    head3 = (f"**Table B.3 — the companion check.** The {len(comp)} constructs that can have a "
             f"companion; the check does not apply to the other {len(C)-len(comp)}. The "
             f"asymmetry is the difference between A's and B's share of a companion, counted "
             f"over prompt text with the construct's own components exempt (the inventory's "
             f"`components`); L2 is over all companions, max is the largest single one; whole "
             f"counts every line and pad-0 the padding-free prompts alone. Only the concept "
             f"norm is at layer {NULL_LAYER}.")
    cols3 = ["construct", "kind", "concept norm", "L2 whole", "L2 pad-0",
             "max whole", "max pad-0", "companions"]
    rows3 = []
    for n in comp:
        c6 = P(C[n], "check10_leakage") or {}
        l2w, l2p, mxw, mxp, ncomp = asym[n]
        rows3.append([n, kind(n, C[n]), num(c6.get("concept_norm"), 2),
                      num(l2w, 3), num(l2p, 3), num(mxw, 3), num(mxp, 3), str(ncomp)])
    files += emit("TABLE_B3", head3, cols3, rows3, None, "llrrrrrr")

    meta = {"what": "Appendix B's three licence tables, rendered from LICENCE_TABLE_CORRECTED",
            "reference": REF, "tokens_billions": TOKENS_B,
            "split_rationale": "the checks have three populations; one 188-row table was "
                               "four-ninths dashes",
            "n_rows": {"B1": len(rows), "B2": len(rows2), "B3": len(rows3)},
            "paper_to_suite_numbering": {
                "1": "check1_twin_unit", "2": "check2_convergence", "3": "check3_length",
                "4": "check4_nu", "5": "check6_calibration", "6": "check10_leakage",
                "7a": "check8_nulls__suite_7.scrambled",
                "7b": "check8_nulls__suite_7.split_half",
                "8": "check7_sigma0_floor__suite_8_eps",
                "note": "read off the docstrings in profilebattery/validation/checks.py; the "
                        "JSON keys carry the other numbering as a suffix"},
            "layer_matched_real_norm": {"layer": NULL_LAYER, "n_available": len(rn),
                                        "source": f"the local concept store at {REF}"},
            "sort": "kind, then construct name", "files": files,
            "minutes": (time.time() - t0) / 60,
            "provenance": provenance(["validation/paper1/LICENCE_TABLE_CORRECTED.json",
                                      "phase_battery (the gate's own share)",
                                      "the local concept store (layer-12 real norms)"])}
    (OUT2 / "LICENCE_TABLES.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"B.1 {len(rows)} rows, B.2 {len(rows2)} rows, B.3 {len(rows3)} rows")
    print(f"  {len(files)} files -> {OUT}")
    print(f"  ({meta['minutes']:.1f} min)")


def _ratio(real, block, sub):
    if not real or not block:
        return None
    p = ((block.get(sub) or {}) if isinstance(block, dict) else {}).get("p95_norm")
    return (real / p) if p else None


if __name__ == "__main__":
    main()
