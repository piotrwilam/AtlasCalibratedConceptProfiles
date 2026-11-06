r"""B_SUBTYPES.md -- what every construct's B set is made of, and whether B reuses A.

Set B holds the construct's token **out of its role**. Two kinds of subtype do that:

* **out of role entirely** -- the characters appear in a string, a comment, a docstring or
  inside an identifier, where they are not the construct's token at all;
* **another role** (`other_role:sibling:<donor>`) -- the token IS the keyword, doing a
  different grammatical job, in a construct that shares it. `For`'s B draws `for` from
  `ListComp`, `DictComp`, `SetComp`, `GeneratorExp` and `AsyncFor`.

The second kind is what makes the A - B difference a difference of **role** rather than of
token, so it matters how many constructs have one, and it matters whether those prompts are
the donor's own A prompts. `families.sibling_b_cores` builds them from the donor's A grammar,
so the **shape** is shared by construction. Whether the rendered **text** is shared is a
different question and is answered here by comparing the prompt strings themselves.

Counts are over both twins unless a column says otherwise.

    python experiments/b_subtypes.py
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import re
import time

from experiments.atlas import D, provenance

BAT = D / "phase_battery"
OUT = D / "validation/paper1"
TWINS = ("twin1", "twin2")
SIB = re.compile(r"^other_role:sibling:(.+)$")


def read(path: str) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        return [json.loads(l) for l in fh]


def main() -> None:
    t0 = time.time()
    constructs = sorted({os.path.basename(p).rsplit("_", 1)[0]
                         for p in glob.glob(str(BAT / "twin1" / "*_A.jsonl.gz"))})
    per, a_text, b_sib_text, a_cores = {}, {}, {}, {}
    sub_total = collections.Counter()

    for c in constructs:
        subs = collections.Counter()
        donors = collections.Counter()
        sib_rows, a_rows, b_all = [], [], []
        for tw in TWINS:
            fa, fb = BAT / tw / f"{c}_A.jsonl.gz", BAT / tw / f"{c}_B.jsonl.gz"
            if fa.exists():
                a_rows += read(str(fa))
            if not fb.exists():
                continue
            for r in read(str(fb)):
                b_all.append(r)
                st = str(r.get("subtype"))
                subs[st] += 1
                sub_total[st] += 1
                m = SIB.match(st)
                if m:
                    donors[m.group(1)] += 1
                    sib_rows.append(r)
        a_text[c] = {r["text"] for r in a_rows}
        a_cores[c] = {r["core_id"] for r in a_rows}
        b_sib_text[c] = {r["text"] for r in sib_rows}
        # how each set falls across the grid, by scope and by padding level: section A.2
        # item 4 quotes A's as balanced and B's as not, and until `backfill_paper1.py` wrote
        # SET_GRIDS.json the counts themselves were in no artefact
        def _grid(rows):
            g = {}
            for field, key in (("scope", "by_scope"), ("padding", "by_padding")):
                if rows and field in rows[0]:
                    cc = collections.Counter(r[field] for r in rows)
                    g[key] = {str(k): v for k, v in sorted(cc.items(), key=lambda kv: str(kv[0]))}
                    g[key + "_balanced"] = len(set(cc.values())) == 1
            return g

        per[c] = {"n_b": sum(subs.values()), "subtypes": dict(subs.most_common()),
                  "grid_a": _grid(a_rows), "grid_b": _grid(b_all),
                  "n_subtypes": len(subs),
                  "has_other_role": bool(donors),
                  "donors": dict(donors.most_common()),
                  "n_other_role_prompts": sum(donors.values()),
                  # the donor core a sibling prompt was built from, e.g.
                  # "For/B:sibling:ListComp/ListComp/c0" -> "ListComp/c0"
                  "sibling_core_ids": sorted({r["core_id"] for r in sib_rows})}

    # --- does any A set share TEXT with another construct's "other role" B part? ---
    shared_text, shared_shape = [], []
    for c, sib in b_sib_text.items():
        if not sib:
            continue
        for donor in per[c]["donors"]:
            if donor not in a_text:
                continue
            both = sib & a_text[donor]
            if both:
                shared_text.append({"b_construct": c, "donor": donor,
                                    "n_shared_prompts": len(both),
                                    "example": sorted(both)[0][:120]})
        # shapes: the donor core a sibling id names, against the donor's own A cores
        for cid in per[c]["sibling_core_ids"]:
            tail = cid.split("/B:sibling:", 1)[-1]
            donor = tail.split("/", 1)[0]
            dcore = tail.split("/", 1)[1] if "/" in tail else ""
            if donor in a_cores and dcore in a_cores[donor]:
                shared_shape.append({"b_construct": c, "donor": donor, "donor_core": dcore})

    n_other = sum(1 for v in per.values() if v["has_other_role"])
    doc = {
        "what": "every construct's B subtypes and counts, and whether B's 'other role' part "
                "reuses the donor's A prompts or shapes",
        "definition": {
            "b_set": "the construct's token out of its role",
            "other_role": "subtype 'other_role:sibling:<donor>': the token IS the keyword, in "
                          "a construct that shares it",
            "shape_shared": "the sibling core id names a donor core; shared if the donor's A "
                            "set uses a core of that id",
            "text_shared": "the rendered prompt string appears in both",
            "counts": "over both twins"},
        "fixed_before_computing": True,
        "n_constructs": len(constructs), "n_twins": len(TWINS),
        "n_with_other_role": n_other,
        "n_without_other_role": len(constructs) - n_other,
        "subtype_totals": dict(sub_total.most_common()),
        "n_distinct_subtypes": len(sub_total),
        "shared_text": {"n_pairs": len(shared_text), "pairs": shared_text[:40]},
        "shared_shape": {"n_pairs": len(shared_shape),
                         "n_distinct_donor_cores": len({(s["donor"], s["donor_core"])
                                                        for s in shared_shape}),
                         "pairs": shared_shape[:40]},
        "per_construct": per,
        "minutes": (time.time() - t0) / 60,
        "provenance": provenance(["phase_battery/twin{1,2}/*_{A,B}.jsonl.gz"])}
    (OUT / "B_SUBTYPES.json").write_text(json.dumps(doc, indent=1, default=float))

    fam = collections.Counter()
    for st in sub_total:
        fam["another role" if SIB.match(st) else "out of role entirely"] += sub_total[st]
    md = [
        "# What set B is made of, for all 188 constructs", "",
        "Set B holds the construct's token **out of its role**. Counts are over both twins.",
        "",
        f"**{len(constructs)} constructs, {len(sub_total)} distinct subtypes.** "
        f"**{n_other} of {len(constructs)}** have an *another role* subtype "
        f"(`other_role:sibling:<donor>`), in which the token is still the keyword but is "
        f"doing a different grammatical job; the remaining {len(constructs) - n_other} "
        f"contrast only against the token out of role entirely (strings, comments, "
        f"docstrings, identifiers).", "",
        f"Prompts by kind: **{fam['another role']:,}** in another role, "
        f"**{fam['out of role entirely']:,}** out of role entirely.", "",
        "## Does B reuse the donor's A set?", "",
        f"**Shapes: yes, by construction** — {doc['shared_shape']['n_pairs']} sibling cores "
        f"name a core the donor's own A set also uses "
        f"({doc['shared_shape']['n_distinct_donor_cores']} distinct donor cores). "
        f"`families.sibling_b_cores` builds them from the donor's A grammar, so this is the "
        f"design, not a leak.", "",
        (f"**Prompts: no** — no rendered prompt string appears both in a construct's A set "
         f"and in another construct's 'other role' B part. Checked every "
         f"(B construct, donor) pair."
         if not shared_text else
         f"**Prompts: {len(shared_text)} (B construct, donor) pairs SHARE rendered text.** "
         f"This needs resolving: a prompt in both an A set and another construct's B set is "
         f"counted on both sides of a contrast."), "",
        "## Subtype totals", "", "| subtype | kind | prompts |", "|---|---|---:|"]
    for st, n in sub_total.most_common():
        md.append(f"| `{st}` | {'another role' if SIB.match(st) else 'out of role'} | {n:,} |")
    md += ["", "## Per construct", "",
           "| construct | B prompts | subtypes | another role? | donors | in-another-role "
           "prompts |", "|---|---:|---:|---|---|---:|"]
    for c in constructs:
        v = per[c]
        md.append(f"| `{c}` | {v['n_b']:,} | {v['n_subtypes']} | "
                  f"{'yes' if v['has_other_role'] else 'no'} | "
                  f"{', '.join(v['donors']) or '—'} | {v['n_other_role_prompts']:,} |")
    (OUT / "B_SUBTYPES.md").write_text("\n".join(md) + "\n")

    print(f"  {len(constructs)} constructs, {len(sub_total)} distinct B subtypes")
    print(f"  with an 'another role' subtype: {n_other}/{len(constructs)}")
    print(f"  prompts: {fam['another role']:,} another role, "
          f"{fam['out of role entirely']:,} out of role entirely")
    print(f"  shapes shared with the donor's A set: {doc['shared_shape']['n_pairs']} sibling "
          f"cores, {doc['shared_shape']['n_distinct_donor_cores']} distinct donor cores")
    print(f"  rendered prompts shared: {len(shared_text)} pair(s)"
          + ("" if not shared_text else f" -- FIRST: {shared_text[0]}"))
    print(f"-> {OUT / 'B_SUBTYPES.md'}, B_SUBTYPES.json")


if __name__ == "__main__":
    main()
