r"""Check 6's companion asymmetry, recomputed on the STORED battery.

Three things this settles.

**What `asymmetry_max` is.** `run_validation.py:377` builds the asymmetry dict as
`{r["companion"]: r["diff"] for r in comp[c]["shares"]}` from `probe/companions.json`, and
`checks.companion_leakage` returns `max |diff|` over it. So it is the **logged companion
share of A minus that of B** -- a count over prompt *texts*, by `ast.walk`. Nothing in it comes
from activations. The `layer=12` argument reaches only `concept_norm`, the norm of the
A-minus-B profile at that layer, which is reported beside it and not inside it.

**Which prompts it describes.** `audit_companions.py` does not read the battery: it calls
`build_cores`/`assemble` to make a fresh sample, with `abs(hash(...))` seeds that PYTHONHASHSEED
salts per process. Its `n_a` is 360 against the battery's 1,440 a set over both twins. So the
stored shares describe *a* draw from the generator, not the battery the model was recorded on.
This recomputes them on the battery itself.

**Two scopings, because `body_start` is not stored.** The generator's own audit skips the
padding, passing `prompt.body_start` into `present`. The stored rows keep `padding` only as a
*level* (0, 1 or 2), and the real boundary depends on what the formatter did to the padding, so
body_start cannot be recovered. Both bounds are therefore reported:

* `whole` -- every prompt, counted from line 1. Padding is corpus-typical code, so this
  overcounts both sides; it is an upper bound on each share.
* `nopad` -- the `padding == 0` prompts only, where `body_start == 1` and the count is exactly
  the generator's. A is exactly a third of each set by design (480 of 1,440); B is not
  (198 to 507), so this subset shifts B's cell mix. Exact scoping, narrower population.

Neither is the stored figure, and the two bracket it. The comparison against
`probe/companions.json` is printed so the gap is visible rather than assumed.

**Two component lists.** The 0.15 cap exempts a construct's own components.
`generate/audit.py :: COMPONENTS` is the list the code exempts -- one entry, `Match`; the
inventory's `components` field, from `inventory/build.py :: COMPONENTS`, is a different list,
and they disagree for FOUR of these eight: `For` and `While` get `['Break','Continue']` from
the inventory and `()` from the code, `Try` gets `['ExceptHandler']` against `()`, and `Match`
gets `['MatchAs','Pass']` against four Match-pattern names. (An earlier version of this note
said five; the four are the ones that differ.) Only `Match`'s figures move materially between
them -- asymmetry max 0.980 under the code's list against 0.186 under the inventory's, L2
1.056 against 0.395 -- while `For` and `While` shift only in L2 and in the companion count
(0.668 -> 0.644 over 38 -> 36, and 1.058 -> 1.043 over 36 -> 34). The figures are reported
under both, and under neither.

    .venv/bin/python experiments/check6_companions.py
"""
from __future__ import annotations

import collections
import csv
import gzip
import json
import time

import numpy as np

from experiments.atlas import D, provenance

BAT = D / "phase_battery"
OUT = D / "validation/paper1"
TWINS = ("twin1", "twin2")
#: the design's cap on |share(A) - share(B)|, components exempt
CAP = 0.15
#: the eight constructs check 6 is reported for in Table B.3
EIGHT = ("ClassDef", "For", "FunctionDef", "If", "Match", "Try", "While", "With")
#: B's sub-sets, the sibling-role ones folded together
OUT_OF_ROLE = ("docstring", "comment", "string", "identifier")


def read(path) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def subset_of(row: dict) -> str:
    st = str(row.get("subtype"))
    return "sibling-role" if st.startswith("other_role") else st


def main() -> None:
    t0 = time.time()
    from profilebattery.generate.audit import COMPONENTS, present
    inv = json.loads((D / "inventory/inventory_v2.json").read_text())["constructs"]
    stored = json.loads((D / "probe/companions.json").read_text())

    doc = {
        "what": "check 6's companion asymmetry recomputed on the stored battery, for the "
                "eight constructs Table B.3 reports",
        "definition": {
            "share": "the fraction of a set's prompts in which the companion construct "
                     "appears, by profilebattery.generate.audit.present on the prompt text",
            "diff": "share(A) - share(B)",
            "asymmetry_max": "max |diff|, and asymmetry_l2 the L2 norm of the diff vector "
                             "-- the two quantities Table B.3 carries. checks.companion_"
                             "leakage takes them over whatever companions companions.json "
                             "logs, and audit() has already dropped audit.COMPONENTS before "
                             "logging, so the like-for-like replacement is the "
                             "excl_audit_components column",
            "layer_12": "reaches only concept_norm, not the asymmetry",
            "cap": f"the design caps |diff| at {CAP} for these constructs, components exempt",
            "scoping": {
                "whole": "all prompts, counted from line 1: padding included, an upper bound",
                "nopad": "the padding == 0 prompts only, where body_start == 1 and the count "
                         "is exactly the generator's; A is a third of each set by design, B "
                         "is not, so B's cell mix shifts"},
            "component_lists": "audit.COMPONENTS is what the code exempts; the inventory's "
                               "`components` field is a different list; both reported",
            "source": "phase_battery, both twins -- NOT the regenerated sample that "
                      "probe/companions.json was built from"},
        "fixed_before_computing": True,
        "per_construct": {}}
    rows = []

    for c in EIGHT:
        files = {s: [BAT / t / f"{c}_{s}.jsonl.gz" for t in TWINS] for s in ("A", "B")}
        if not all(p.exists() for ps in files.values() for p in ps):
            doc["per_construct"][c] = {"error": "set missing from the battery"}
            continue
        a_rows = [r for p in files["A"] for r in read(p)]
        b_rows = [r for p in files["B"] for r in read(p)]

        cnt = {k: collections.Counter() for k in
               ("a_whole", "a_nopad", "b_whole", "b_nopad")}
        n = {"a_whole": len(a_rows), "b_whole": len(b_rows), "a_nopad": 0, "b_nopad": 0}
        by_sub = collections.defaultdict(collections.Counter)
        n_sub = collections.Counter()

        for r in a_rows:
            seen = present(r["text"], c, 1)
            cnt["a_whole"].update(seen)
            if r["padding"] == 0:
                n["a_nopad"] += 1
                cnt["a_nopad"].update(seen)
        for r in b_rows:
            seen = present(r["text"], c, 1)
            cnt["b_whole"].update(seen)
            sub = subset_of(r)
            n_sub[sub] += 1
            by_sub[sub].update(seen)
            if r["padding"] == 0:
                n["b_nopad"] += 1
                cnt["b_nopad"].update(seen)

        ex_audit = set(COMPONENTS.get(c, ()))
        ex_inv = set(inv.get(c, {}).get("components") or ())
        subs = [s for s in OUT_OF_ROLE if s in n_sub] + \
               (["sibling-role"] if "sibling-role" in n_sub else [])

        shares = []
        for comp in sorted(set(cnt["a_whole"]) | set(cnt["b_whole"])):
            s = {"companion": comp,
                 "exempt_audit": comp in ex_audit,
                 "exempt_inventory": comp in ex_inv}
            for scope in ("whole", "nopad"):
                a = cnt[f"a_{scope}"][comp] / max(1, n[f"a_{scope}"])
                b = cnt[f"b_{scope}"][comp] / max(1, n[f"b_{scope}"])
                s[f"a_{scope}"], s[f"b_{scope}"], s[f"diff_{scope}"] = a, b, a - b
            s["b_by_subset"] = {k: by_sub[k][comp] / n_sub[k] for k in subs}
            shares.append(s)
            rows.append({"construct": c, "companion": comp,
                         "a_share": s["a_whole"], "b_share": s["b_whole"],
                         "diff": s["diff_whole"],
                         "a_share_nopad": s["a_nopad"], "b_share_nopad": s["b_nopad"],
                         "diff_nopad": s["diff_nopad"],
                         "exempt_audit": s["exempt_audit"],
                         "exempt_inventory": s["exempt_inventory"],
                         **{f"b_{k}": s["b_by_subset"][k] for k in subs}})

        def mx(scope, keep):
            v = [abs(s[f"diff_{scope}"]) for s in shares if keep(s)]
            return max(v) if v else None

        def l2(scope, keep):
            v = [s[f"diff_{scope}"] for s in shares if keep(s)]
            return float(np.linalg.norm(v)) if v else None

        def ncomp(keep):
            return sum(1 for s in shares if keep(s))

        ways = {
            "all": lambda s: True,
            "excl_audit_components": lambda s: not s["exempt_audit"],
            "excl_inventory_components": lambda s: not s["exempt_inventory"]}
        doc["per_construct"][c] = {
            "n": n, "b_subsets": dict(n_sub), "n_companions": len(shares),
            "asymmetry_max": {f"{w}_{scope}": mx(scope, keep)
                              for w, keep in ways.items()
                              for scope in ("whole", "nopad")},
            "asymmetry_l2": {f"{w}_{scope}": l2(scope, keep)
                             for w, keep in ways.items()
                             for scope in ("whole", "nopad")},
            "n_companions_by_way": {w: ncomp(keep) for w, keep in ways.items()},
            "stored_asymmetry_max": (max(abs(r["diff"]) for r in stored[c]["shares"])
                                     if c in stored else None),
            "stored_asymmetry_l2": (float(np.linalg.norm(
                [r["diff"] for r in stored[c]["shares"]])) if c in stored else None),
            "stored_n_companions": len(stored[c]["shares"]) if c in stored else None,
            "stored_n_a": stored.get(c, {}).get("n_a"),
            "stored_n_b": stored.get(c, {}).get("n_b"),
            "over_cap_whole_excl_audit": sorted(
                (s["companion"] for s in shares
                 if not s["exempt_audit"] and abs(s["diff_whole"]) > CAP),
                key=lambda k: -max(abs(s["diff_whole"]) for s in shares
                                   if s["companion"] == k)),
            "mean_b_share_by_subset": {
                k: float(np.mean([s["b_by_subset"][k] for s in shares])) for k in subs},
            "shares": shares}

    ok = {k: v for k, v in doc["per_construct"].items() if "error" not in v}
    summ = {}
    for w in ("all", "excl_audit_components", "excl_inventory_components"):
        for scope in ("whole", "nopad"):
            vals = [v["asymmetry_max"][f"{w}_{scope}"] for v in ok.values()]
            vals = [x for x in vals if x is not None]
            summ[f"{w}_{scope}"] = {
                "median": float(np.median(vals)), "min": float(min(vals)),
                "max": float(max(vals)), "n_over_cap": sum(1 for x in vals if x > CAP),
                "n": len(vals)}
    doc["summary"] = {
        "n_constructs": len(ok), "cap": CAP, "asymmetry_max": summ,
        "n_companion_pairs_over_cap_whole_excl_audit":
            sum(len(v["over_cap_whole_excl_audit"]) for v in ok.values()),
        "stored_median": float(np.median([v["stored_asymmetry_max"] for v in ok.values()
                                          if v["stored_asymmetry_max"] is not None]))}
    doc["table_b3"] = {
        "what": "the replacement for Table B.3's asymmetry columns: the battery's own, "
                "excl_audit_components -- the same exemption companions.json applies",
        "columns": ["construct", "asymmetry_l2_whole", "asymmetry_l2_nopad",
                    "asymmetry_max_whole", "asymmetry_max_nopad", "n_companions"],
        "caption_note": "the asymmetry columns are counts over prompt text and have no "
                        "layer; only the concept norm is at layer 12",
        "rows": [[c,
                  v["asymmetry_l2"]["excl_audit_components_whole"],
                  v["asymmetry_l2"]["excl_audit_components_nopad"],
                  v["asymmetry_max"]["excl_audit_components_whole"],
                  v["asymmetry_max"]["excl_audit_components_nopad"],
                  v["n_companions_by_way"]["excl_audit_components"]]
                 for c, v in doc["per_construct"].items() if "error" not in v]}
    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance(["phase_battery/twin{1,2}/*_{A,B}.jsonl.gz",
                                    "probe/companions.json",
                                    "inventory/inventory_v2.json"])
    (OUT / "CHECK6_COMPANIONS.json").write_text(json.dumps(doc, indent=1, default=float))

    head = ["construct", "companion", "a_share", "b_share", "diff",
            "a_share_nopad", "b_share_nopad", "diff_nopad",
            "exempt_audit", "exempt_inventory"]
    cols = head + sorted({k for r in rows for k in r} - set(head))
    with open(OUT / "CHECK6_COMPANIONS.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows([[r.get(k) for k in cols] for r in rows])

    print(f"  asymmetry_max per construct, whole prompt | padding==0 only\n")
    print(f"  {'construct':<12s} {'comps':>5s} {'all':>13s} {'-audit':>13s} "
          f"{'-inventory':>13s} {'stored':>7s}")
    for c, v in doc["per_construct"].items():
        if "error" in v:
            print(f"  {c:<12s} {v['error']}")
            continue
        m = v["asymmetry_max"]
        cell = lambda w: f"{m[w+'_whole']:.3f}|{m[w+'_nopad']:.3f}"
        print(f"  {c:<12s} {v['n_companions']:5d} {cell('all'):>13s} "
              f"{cell('excl_audit_components'):>13s} "
              f"{cell('excl_inventory_components'):>13s} "
              f"{v['stored_asymmetry_max']:7.3f}")
    s = doc["summary"]["asymmetry_max"]
    print(f"\n  median over the eight, whole prompt: all {s['all_whole']['median']:.3f}, "
          f"minus the audit's components {s['excl_audit_components_whole']['median']:.3f}, "
          f"minus the inventory's {s['excl_inventory_components_whole']['median']:.3f}")
    print(f"  over the {CAP} cap (whole, minus the audit's components): "
          f"{s['excl_audit_components_whole']['n_over_cap']} of "
          f"{s['excl_audit_components_whole']['n']} constructs, "
          f"{doc['summary']['n_companion_pairs_over_cap_whole_excl_audit']} companion pairs")
    print(f"  stored median (the regenerated sample): {doc['summary']['stored_median']:.3f}")
    print(f"\n  mean B share by sub-set")
    for c, v in doc["per_construct"].items():
        if "error" in v:
            continue
        print(f"  {c:<12s} " + "  ".join(
            f"{k} {x:.3f} (n={v['b_subsets'][k]})"
            for k, x in v["mean_b_share_by_subset"].items()))
    print(f"\n  Table B.3 replacement (excl_audit_components, the exemption "
          f"companions.json itself applies)")
    print(f"  {'construct':<12s} {'L2 whole':>9s} {'L2 nopad':>9s} {'max whole':>10s} "
          f"{'max nopad':>10s} {'n comp':>7s}   {'was: L2':>8s} {'max':>6s} {'n':>4s}")
    for r in doc["table_b3"]["rows"]:
        v = doc["per_construct"][r[0]]
        print(f"  {r[0]:<12s} {r[1]:9.3f} {r[2]:9.3f} {r[3]:10.3f} {r[4]:10.3f} "
              f"{r[5]:7d}   {v['stored_asymmetry_l2']:8.3f} "
              f"{v['stored_asymmetry_max']:6.3f} {v['stored_n_companions']:4d}")
    print(f"\n-> {OUT / 'CHECK6_COMPANIONS.csv'}, CHECK6_COMPANIONS.json")


if __name__ == "__main__":
    main()
