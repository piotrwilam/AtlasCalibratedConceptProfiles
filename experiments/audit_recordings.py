r"""Which recording each artefact read, confirmed by content, and the headline numbers re-derived.

Two halves, deliberately independent.

**Part 1** takes each artefact and recomputes one number it reports **through the dataset-only
resolver**, then compares. A match says the artefact read the dataset whatever its code does; a
difference says it read something else. This does not depend on me having enumerated the read
paths correctly, which is the weaker half of the audit -- three times now the enumeration has
missed a local tree.

**Part 2** re-derives the papers' headline numbers through the resolver alone and compares them
with the figures in the drafts.

    python experiments/audit_recordings.py
"""
from __future__ import annotations

import io
import json
import time

import numpy as np

from experiments.atlas import D, K, OUT, REPO, provenance, token
from profilebattery.profiles import resolve
from profilebattery.paths import data_root, dataset_repo


def acc(ref: str, name: str):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token())
    with fs.open(f"datasets/{REPO}/record/{ref}/{name}", "rb") as f:
        return np.load(io.BytesIO(f.read()))


def car(v, k: int = K):
    f = np.asarray(v, np.float64).ravel() ** 2
    return np.sort(np.argsort(-f, kind="stable")[:k])


def report(rows):
    print(f"{'artefact / number':46s} {'stored':>12s} {'via resolver':>13s} {'verdict':>10s}")
    for name, stored, got, tol in rows:
        if stored is None or got is None:
            print(f"{name:46s} {'--':>12} {'--':>13} {'no data':>10}")
            continue
        d = abs(got - stored)
        ok = d <= tol
        print(f"{name:46s} {stored:12.4f} {got:13.4f} "
              f"{'match' if ok else f'differs {d:+.4f}':>10}")


def main() -> None:
    import os, pathlib
    from experiments.atlas import p1_set, LAST
    t0 = time.time()
    rows, notes = [], {}

    # --- structural fact that settles every record-based artefact at once ------------------
    rec_local = data_root() / "record"
    notes["local_record_tree_exists"] = rec_local.exists()

    # --- 1. ablation masks: the `twenty` mask against a resolver-derived carrier set --------
    from experiments.ablation_splits import OUTDIR
    m = OUTDIR / "pretrain_iter_0001000__For__twenty.npy"
    if m.exists():
        mask = np.load(m)
        got = float(np.intersect1d(mask, car(resolve.concept("pretrain_iter_0001000",
                                                             "twin1", "For"))).size / K)
        rows.append(("ABLATION masks: For@0001000 twenty vs resolver", 1.0, got, 0.01))

    # --- 2. the (v)-vs-20-cell table: its stored 20-cell twin ceiling ----------------------
    p = D / "validation/paper1/b10v" / LAST / "For.json"
    if p.exists():
        stored = json.loads(p.read_text())["V"].get("stored_20cell_twin_ceiling")
        c1 = car(resolve.concept(LAST, "twin1", "For"))
        c2 = car(resolve.concept(LAST, "twin2", "For"))
        rows.append(("(v) table: For@0484865 20-cell twin ceiling", stored,
                     float(np.intersect1d(c1, c2).size / K), 0.005))

    # --- 3. MU0_RESIDUE: the profile's size on its own carriers ----------------------------
    p = OUT / "MU0_RESIDUE.json"
    if p.exists():
        e = json.loads(p.read_text())["rows"].get(f"{LAST}|For|twenty")
        if e:
            v = resolve.concept(LAST, "twin1", "For").ravel()
            cs = car(v)
            rows.append(("MU0_RESIDUE: For@0484865 profile rms on carriers",
                         e["profile_rms_on_carriers"],
                         float(np.sqrt((v[cs] ** 2).mean())), 0.02))

    # --- 4. headline: twin ceiling, plug-in, median over P1 --------------------------------
    names = sorted(p1_set())
    for ref, want in (("untrained", 0.28), ("pretrain_iter_0001000", 0.84)):
        vals = []
        for c in names[:40]:
            try:
                a = car(resolve.concept(ref, "twin1", c))
                b = car(resolve.concept(ref, "twin2", c))
            except Exception:                                          # noqa: BLE001
                continue
            vals.append(np.intersect1d(a, b).size / K)
        if vals:
            rows.append((f"twin ceiling plug-in, median, {ref} (40 constructs)",
                         want, float(np.median(vals)), 0.02))

    # --- 5. headline: section 4.8 medians from the FULL artefact ---------------------------
    from experiments.atlas import fetch
    for ref, want in (("pretrain_iter_0001000", 1.00), ("pretrain_iter_0020000", 3.26),
                      ("pretrain_iter_0484865", 5.16)):
        d = json.loads(fetch(f"validation/ablation/FULL_{ref}.json").read_text())["constructs"]
        v = [x["current_over_random_in_sd"] for x in d.values()
             if isinstance(x, dict) and x.get("current_over_random_in_sd") is not None]
        rows.append((f"section 4.8 median sd, {ref[-7:]}", want,
                     float(np.median(v)) if v else None, 0.01))

    # --- 6. headline: the pre-registered test at 0001000 -----------------------------------
    p = OUTDIR / "ABLATION_pretrain_iter_0001000.json"
    if p.exists():
        cs_ = json.loads(p.read_text())["constructs"]
        d = [v["v"]["over_random_in_sd"] - v["one"]["over_random_in_sd"]
             for v in cs_.values() if isinstance(v.get("v"), dict) and "nats" in v["v"]]
        rows.append(("pre-registered (v)-one median, 0001000", -0.55,
                     float(np.median(d)) if d else None, 0.01))

    # --- 7. headline: For's indented vs module overlap -------------------------------------
    p = OUT / "SCOPE_OVERLAP.json"
    if p.exists():
        e = json.loads(p.read_text())["rows"].get(f"{LAST}|For")
        if e:
            rows.append(("For indented/module overlap @0484865", 0.494,
                         e["between_scope_within_twin"]["twin1"]["overlap_share_of_K"], 0.001))

    report(rows)
    doc = {"what": "which recording each artefact read, confirmed by recomputing a number it "
                   "reports through the dataset-only resolver, and the headline numbers "
                   "re-derived the same way",
           "local_record_tree_exists": notes["local_record_tree_exists"],
           "note": "the local record/ tree was retired on 20 Oct (A51), so every accumulator "
                   "and per-prompt read after that date necessarily came from the dataset; the "
                   "checks below confirm that by content rather than by date",
           "checks": [{"name": n, "stored": s, "via_resolver": g,
                       "match": (None if s is None or g is None else bool(abs(g - s) <= t))}
                      for n, s, g, t in rows],
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["dataset only, via profilebattery.profiles.resolve"])}
    (OUT / "AUDIT_RECORDINGS.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"\n-> {OUT / 'AUDIT_RECORDINGS.json'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
