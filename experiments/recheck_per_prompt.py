r"""Re-run the four licence checks that pair per-prompt vectors with prompt rows.

Checks 2 (convergence), 3 (length), 7a (null scrambled) and 7b (null split half) are the only
ones that read the per-prompt arrays; every other check reads the per-core, per-cell
accumulators and is order-independent. All four indexed `rows[i]` against `primary[i]`, and the
arrays are stored length-sorted while the jsonl is not, so each vector carried another prompt's
core, cell and padding level. See `profilebattery.profiles.profile.recorder_order`.

Two things this runner does that `run_validation` does not, both deliberate:

**It streams the record from the dataset.** `PB_RECORD_DIR` defaults to the local tree, and three
of the four per-prompt references -- `untrained`, `0001000`, `0020000` -- are among the eight
where that tree holds the stale S1-era recording (A50). Reading them locally would re-run the
checks against the wrong record.

**It runs only the four affected checks, on the 31 constructs that have hidden per-prompt
vectors**, rather than the whole suite, because the other checks were never affected and
re-running them would read 300 GB to reproduce numbers already in `SUITE.json`.

    python experiments/recheck_per_prompt.py
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import time

import numpy as np

from experiments.atlas import D, OUT, REPO, provenance, token
from profilebattery.profiles.profile import recorder_order
from profilebattery.profiles.yardstick import Yardstick
from profilebattery.validation.checks import (convergence, length_sensitivity, null_scrambled,
                                              null_split_half)

REFS = ("untrained", "pretrain_iter_0001000", "pretrain_iter_0020000", "pretrain_iter_0484865")
_FS = None


def stream(path: str):
    global _FS
    if _FS is None:
        from huggingface_hub import HfFileSystem
        _FS = HfFileSystem(token=token())
    with _FS.open(f"datasets/{REPO}/{path}", "rb") as f:
        return io.BytesIO(f.read())


def rows_of(twin: str, c: str, kind: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, f"phase_battery/{twin}/{c}_{kind}.jsonl.gz", repo_type="dataset",
                        token=token())
    return recorder_order([json.loads(l) for l in gzip.open(p, "rt")])


def hidden_31() -> list[str]:
    sub = json.loads((D / "validation/battery/PER_PROMPT_SUBSET.json").read_text())["subset"]
    from profilebattery.profiles.names import tree_name
    return sorted({tree_name(x) for x in sub})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    refs = [args.reference] if args.reference else list(REFS)
    names = hidden_31()
    if args.limit:
        names = names[:args.limit]
    out, t0 = {}, time.time()
    for ref in refs:
        ys = Yardstick.from_npz(stream(f"record/{ref}/twin1__C_cells.npz"))
        for c in names:
            try:
                pa = np.asarray(np.load(stream(f"record/{ref}/twin1__{c}_A.npz"))["primary"])
                pb = np.asarray(np.load(stream(f"record/{ref}/twin1__{c}_B.npz"))["primary"])
            except Exception as exc:                                   # noqa: BLE001
                print(f"   {ref} {c:24s} SKIP {type(exc).__name__}", flush=True)
                continue
            if pa.shape[0] == 0 or pb.shape[0] == 0:
                print(f"   {ref} {c:24s} SKIP no hidden vectors", flush=True)
                continue
            ra, rb = rows_of("twin1", c, "A"), rows_of("twin1", c, "B")
            e = {}
            for name, fn in (("2 convergence", lambda: convergence(pa, ra, ys, c)),
                             ("3 length", lambda: length_sensitivity(pa, ra, ys, c)),
                             ("7a null_scrambled",
                              lambda: null_scrambled(pa, ra, pb, rb, ys, c)),
                             ("7b null_split_half", lambda: null_split_half(pa, ra, ys, c))):
                try:
                    e[name] = fn()
                except Exception as exc:                               # noqa: BLE001
                    e[name] = {"error": f"{type(exc).__name__}: {exc}"}
            out[f"{ref}|{c}"] = e
            print(f"   {ref[-8:]:>8s} {c:24s} ok   ({(time.time()-t0)/60:.1f} min)", flush=True)
    doc = {"what": "the four licence checks that pair per-prompt vectors with prompt rows, "
                   "re-run with the rows in the recorder's length-sorted order",
           "why": "all four indexed rows[i] against primary[i]; the arrays are stored "
                  "length-sorted and the jsonl is not, so every vector carried another "
                  "prompt's core, cell and padding level",
           "record_read_from": "the dataset, not the local tree, three of the four references "
                               "being among the eight where the local tree is stale (A50)",
           "population": f"{len(names)} constructs with hidden per-prompt vectors",
           "references": refs, "rows": out, "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["record/<ref>/twin1__<c>_{A,B}.npz", "phase_battery/*"])}
    (OUT / "RECHECK_PER_PROMPT.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"\n{len(out)} construct-references in {(time.time()-t0)/60:.1f} min")
    print(f"-> {OUT / 'RECHECK_PER_PROMPT.json'}")


if __name__ == "__main__":
    main()
