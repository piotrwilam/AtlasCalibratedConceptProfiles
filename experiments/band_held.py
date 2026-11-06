r"""The length index with the position band held at padding level 0.

Check 3 compares a construct's profile across three padding levels. Padding is added ahead of
the construct, so the read site moves to a later fifth of the prompt: for `For` the band changes
in 95% of matched prompts, from band 1 at pad 0 to band 3 at pad 2 in the clearest case. Each
level is then standardised against a different cell's (mu0, sigma0), so part of what check 3
reports as length sensitivity is the yardstick moving under the construct rather than the model
responding to length.

This recomputes the index with every prompt assigned the band its own pad-0 twin had -- the
prompts are matched across levels by (core, draw, domain, axes) -- and reports both ways. What
survives is the part of the effect that is not the band.

    python experiments/band_held.py
"""
from __future__ import annotations

import collections
import gzip
import io
import json

import numpy as np

from experiments.atlas import LAST, OUT, REPO, dataset_record, provenance, token

FLAGGED = ("With", "For", "If", "While", "ImportFrom", "Assert", "Delete", "Return")


def key(r):
    return (r["core_id"], r.get("draw"), r.get("domain"),
            json.dumps(r.get("axes"), sort_keys=True))


def main() -> None:
    from huggingface_hub import HfFileSystem, hf_hub_download
    from profilebattery.profiles.cells import band
    from profilebattery.profiles.profile import build, recorder_order
    from profilebattery.profiles.yardstick import Yardstick
    fs = HfFileSystem(token=token())
    ys = Yardstick.from_npz(dataset_record(LAST, "twin1__C_cells.npz"))
    out = {}
    print(f"{'construct':12s} {'band chg':>8s} | {'as measured':>24s} | {'band held at pad 0':>24s}")
    print(f"{'':12s} {'':8s} | {'index':>7s} {'flagged':>8s} {'min cos':>7s} | "
          f"{'index':>7s} {'flagged':>8s} {'min cos':>7s}")
    for c in FLAGGED:
        try:
            p = hf_hub_download(REPO, f"phase_battery/twin1/{c}_A.jsonl.gz",
                                repo_type="dataset", token=token())
            rs = recorder_order([json.loads(l) for l in gzip.open(p, "rt")])
            with fs.open(f"datasets/{REPO}/record/{LAST}/twin1__{c}_A.npz", "rb") as f:
                prim = np.asarray(np.load(io.BytesIO(f.read()))["primary"])
        except Exception as exc:                                       # noqa: BLE001
            print(f"{c:12s} SKIP {type(exc).__name__}")
            continue
        bp = collections.defaultdict(dict)
        for r in rs:
            bp[int(r.get("padding", 0))][key(r)] = r
        pads = sorted(bp)
        common = set.intersection(*[set(bp[q]) for q in pads]) if len(pads) > 1 else set()
        chg = (sum(1 for k in common
                   if len({band(bp[q][k]["site_construct_token"], bp[q][k]["n_tokens"])
                           for q in pads}) > 1) / len(common)) if common else float("nan")

        def measure(fix: bool):
            prof = {}
            for pad in pads:
                ix = [i for i, r in enumerate(rs) if int(r.get("padding", 0)) == pad]
                rr = []
                for i in ix:
                    r = dict(rs[i])
                    if fix:
                        z = bp[pads[0]].get(key(rs[i]))
                        if z:
                            r["site_construct_token"] = z["site_construct_token"]
                            r["n_tokens"] = z["n_tokens"]
                    rr.append(r)
                prof[pad] = build(prim[ix], rr, ys, c, "A")
            mid = prof[pads[0]].mean.shape[0] // 2
            st = np.stack([prof[q].mean[mid] for q in pads])
            se = np.stack([prof[q].se[mid] for q in pads]).mean(0)
            idx = st.std(axis=0, ddof=1) / np.maximum(se, 1e-12)
            cs = [float(prof[pads[i]].mean[mid] @ prof[pads[i + 1]].mean[mid] /
                        (np.linalg.norm(prof[pads[i]].mean[mid])
                         * np.linalg.norm(prof[pads[i + 1]].mean[mid])))
                  for i in range(len(pads) - 1)]
            return float(np.median(idx)), float((idx > 3).mean()), (min(cs) if cs else None)

        a, b = measure(False), measure(True)
        out[c] = {"band_change_share": chg,
                  "as_measured": {"index_median": a[0], "flagged_share": a[1], "min_cos": a[2]},
                  "band_held": {"index_median": b[0], "flagged_share": b[1], "min_cos": b[2]},
                  "index_fall": (a[0] - b[0]) / a[0] if a[0] else None,
                  "flagged_fall": (a[1] - b[1]) / a[1] if a[1] else None}
        print(f"{c:12s} {chg:8.1%} | {a[0]:7.3f} {a[1]:8.1%} {a[2]:7.4f} | "
              f"{b[0]:7.3f} {b[1]:8.1%} {b[2]:7.4f}", flush=True)
    fi = [v["index_fall"] for v in out.values() if v["index_fall"] is not None]
    ff = [v["flagged_fall"] for v in out.values() if v["flagged_fall"] is not None]
    # The loop SKIPs a construct whose per-prompt arrays it cannot fetch, and the release
    # carries the record for `for` alone -- so on a clean machine every construct skipped,
    # the medians came out NaN, and this wrote an artefact that looked like the real one with
    # NaN inside, over the downloaded copy (found by the clean-environment run, 5 Nov). The
    # eight are the population the appendix reports, so anything less is a refusal.
    if len(fi) < len(FLAGGED) or len(ff) < len(FLAGGED):
        raise SystemExit(
            f"band_held: read {len(fi)} of {len(FLAGGED)} constructs. This needs the "
            f"per-prompt record, record/{LAST}/twin1__<c>_A.npz, which the public dataset "
            f"does not carry -- it holds one worked example. Nothing written.")
    print(f"\nhow much of check 3 is the band moving: index falls by a median {np.median(fi):.1%}, "
          f"flagged units by {np.median(ff):.1%} (n={len(fi)})")
    still = [c for c, v in out.items() if v["band_held"]["flagged_share"] > 0.05]
    print(f"still above 5% flagged with the band held: {len(still)} of {len(out)}  {still}")
    doc = {"what": "check 3's length index with the position band held at padding level 0",
           "reference": LAST, "twin": "twin1", "constructs": out,
           "matched_by": "(core_id, draw, domain, axes), which identifies the same prompt at "
                         "each padding level",
           "summary": {"median_index_fall": float(np.median(fi)),
                       "median_flagged_fall": float(np.median(ff)),
                       "still_flagged_with_band_held": still},
           "provenance": provenance([f"record/{LAST}/twin1__<c>_A.npz", "phase_battery/twin1/*"],
                                    reference=LAST)}
    (OUT / "LENGTH_BAND_HELD.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"-> {OUT / 'LENGTH_BAND_HELD.json'}")


if __name__ == "__main__":
    main()
