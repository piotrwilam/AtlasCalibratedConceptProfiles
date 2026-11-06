r"""A local store of the `concept` arrays, so the 19x CDN penalty is paid once.

Measured on the wire: a member read of a profile npz costs **11.4 MB** however small the member
is, because the dataset sits behind a chunk-serving CDN -- `concept` is 0.59 MB of that. Three
members cost 22.3 MB and the whole 99 MB file costs 49.6 MB. Every pass that needs entry values
has therefore been paying about **135 GB** (169 constructs x 35 references x 2 twins) to obtain
**7.0 GB** of array content, and paying it again next time.

This stores each `concept` as its own .npy under `concept_cache/<ref>__<twin>/<construct>.npy`
and never collapses or deletes them, so the second reader is free. Nothing else about the read
changes: same resolver, same dataset, same arrays.

    from experiments.concept_store import concept_of, warm
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from experiments.atlas import D

STORE = D / "validation/paper2/concept_cache"
N_UNITS = 24 * 6144


def path(ref: str, twin: str, construct: str):
    return STORE / f"{ref}__{twin}" / f"{construct}.npy"


def concept_of(ref: str, twin: str, construct: str, attempts: int = 20) -> np.ndarray:
    """One construct's concept vector, from the store or fetched into it."""
    p = path(ref, twin, construct)
    if p.exists():
        try:
            return np.load(p)
        except Exception:                                           # noqa: BLE001
            p.unlink(missing_ok=True)                               # truncated: refetch
    from experiments.atlas import _tree
    last = None
    for attempt in range(attempts):
        try:
            z = _tree(ref, twin, construct, ("concept",))
            v = z.get("concept")
            if v is None:
                z = _tree(ref, twin, construct, ("A_mean", "B_mean"))
                v = z["A_mean"] - z["B_mean"]
            arr = np.asarray(v, np.float32).ravel()
            p.parent.mkdir(parents=True, exist_ok=True)
            # the temp name must END in .npy: np.save appends the suffix otherwise and the
            # rename then targets a file that was never written
            t = p.parent / f".{construct}.tmp.npy"
            np.save(t, arr)
            t.replace(p)
            return arr
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            time.sleep(min(30, 2 * (attempt + 1)))
    raise last


def warm(refs, twins, names, workers: int = 8, label: str = "") -> dict:
    """Fetch everything missing, 8 threads, reporting as it goes. Returns a small summary."""
    want = [(r, t, c) for r in refs for t in twins for c in names]
    missing = [x for x in want if not path(*x).exists()]
    t0 = time.time()
    print(f"  {label}store: {len(want)-len(missing):,} of {len(want):,} present, "
          f"{len(missing):,} to fetch (~{len(missing)*11.4/1000:.1f} GB)", flush=True)
    done = [0]
    err = []

    def one(x):
        try:
            concept_of(*x)
        except Exception as exc:                                    # noqa: BLE001
            err.append((x, type(exc).__name__))
        done[0] += 1
        if done[0] % 500 == 0:
            el = time.time() - t0
            print(f"    {done[0]:,}/{len(missing):,} in {el/60:.0f} min "
                  f"(eta {el/done[0]*(len(missing)-done[0])/60:.0f} min)", flush=True)

    if missing:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, missing))
    return {"requested": len(want), "fetched": len(missing), "errors": err[:20],
            "n_errors": len(err), "minutes": (time.time() - t0) / 60}
