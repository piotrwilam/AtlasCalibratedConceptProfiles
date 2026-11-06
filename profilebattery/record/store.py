"""Storage for the sweep: per-prompt at the primary site, accumulators elsewhere.

The rule the §4 budget rests on. Per-prompt vectors are kept only at the construct-token
site, in fp16, because only two of the nine validation checks need per-prompt resolution
(convergence and the median twin) and both read the primary site. Everything else is a
running (mean, M2) accumulator — and crucially they are kept in **raw** units, so any later
re-standardisation against a different C is arithmetic rather than a new sweep.

Accumulators are per *core*, not per set: the bootstrap resamples cores, so per-set
accumulators would forfeit it, and per-core costs eight times nothing.
"""

from __future__ import annotations

import gzip
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Welford:
    """Running mean and M2, vectorised over units. Raw units, never standardised."""

    n: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    active: np.ndarray | None = None        # count of |x| > eps, per unit
    nonzero_eps: float = 1e-6
    #: Keep the spread fields. **False on A and B sets**, which is what D1a §286 and
    #: amendment A21 license: `from_accumulators` reads `|n` and `|mean` and nothing else,
    #: and `m2`/`active` belong to the C yardstick, which is kept whole and accumulated by
    #: `sweep_c` with `spread=True`. Dropping them here is two thirds of the stored bytes
    #: and — measured — most of the loop's cost, since each costs a full pass over the
    #: batch and `m2` needs the float64 widening that the mean alone does not.
    spread: bool = True

    def update(self, x: np.ndarray) -> None:
        """Merge a batch of observations. x: [n_obs, width].

        Chan's parallel form rather than a row loop. The row loop cost 62 ms per prompt
        against 32 ms of forward pass — 576 accumulators per prompt, each doing three numpy
        operations on 6,144 floats through the Python interpreter. The batch form is O(1)
        numpy calls per update regardless of how many rows arrive.

With `spread=False` only the two spread fields are skipped. **The mean's arithmetic is
        the same to the last bit**, the float64 widening included, and that is deliberate: the
        licence pass was recorded in the full form and Stage 4 records the trimmed one, and C1
        compares profiles *across* references, so no reading may be able to tell which form a
        reference happened to be recorded in.

        The cheaper form — `x.mean(axis=0, dtype=np.float64)` straight off the float32 slice,
        skipping the copy — is in fact also bit-identical on real activations, and 5% faster.
        The widening is kept anyway because it makes the two paths identical *by construction*
        rather than by measurement on one set, and 5% of an operation that is 1.7 ms per 90
        updates is not worth a guarantee.

        Two traps, recorded because together they cost an hour and made an identical record
        look broken on 1,695 of 3,432 keys. The two forms store the same float64 mean at
        different precisions, so the files cannot be compared bit-for-bit at all; and the
        obvious repairs are both wrong. Casting the fp32 file down to fp16 **double-rounds**
        (float64 -> float32 -> float16 against the record's single step), which moves about one
        unit in 18,000. And an fp16 ULP computed as `nextafter(x, +inf) - x` is **half the true
        spacing for a negative value at a binade boundary**, since for negatives that step moves
        toward zero — so a legitimate half-ULP rounding reads as 0.8 ULP. Compare the float64
        means in process, which is what `tests/test_record.py` does against a real model.
        """
        if not self.spread:
            return self._update_mean(x)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        k = x.shape[0]
        if not k:
            return
        if self.mean is None:
            w = x.shape[-1]
            self.mean = np.zeros(w, dtype=np.float64)
            self.m2 = np.zeros(w, dtype=np.float64)
            self.active = np.zeros(w, dtype=np.int64)
        bmean = x.mean(axis=0)
        bm2 = ((x - bmean) ** 2).sum(axis=0)
        n = self.n
        delta = bmean - self.mean
        total = n + k
        self.mean += delta * (k / total)
        self.m2 += bm2 + (delta ** 2) * (n * k / total)
        self.n = total
        self.active += (np.abs(x) > self.nonzero_eps).sum(axis=0)

    def _update_mean(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        k = x.shape[0]
        if not k:
            return
        if self.mean is None:
            self.mean = np.zeros(x.shape[-1], dtype=np.float64)
        bmean = x.mean(axis=0)
        total = self.n + k
        self.mean += (bmean - self.mean) * (k / total)
        self.n = total

    def var(self) -> np.ndarray:
        if self.m2 is None:
            raise ValueError("this accumulator kept no spread (spread=False); "
                             "variance is the C yardstick's, not an A/B set's")
        return self.m2 / max(self.n - 1, 1)

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "m2": self.m2, "active": self.active}


class ShardWriter:
    """Per-prompt fp16 vectors, one shard per (reference, set), plus an index.

    Sharded per set so an interruption in hour three costs one set rather than the run — the
    lesson from the timing probe, where a multi-hour MPS job was the part least to be trusted.
    """

    def __init__(self, root: pathlib.Path, reference: str):
        self.root = pathlib.Path(root) / reference
        self.root.mkdir(parents=True, exist_ok=True)
        self.reference = reference

    def path(self, name: str) -> pathlib.Path:
        return self.root / f"{name}.npz"

    def done(self, name: str) -> bool:
        return self.path(name).exists()

    def write(self, name: str, primary: np.ndarray, meta: dict,
              resid: np.ndarray | None = None) -> dict:
        """primary: [n_prompts, n_layers, width] float16.

        `resid` is A21 decision 3's per-prompt residual arm, kept for all of P1 at D1's four
        checkpoints and written into the same shard rather than beside it, so `done()` still
        answers for the whole set and an interruption cannot leave one stream of a set
        present and the other missing.

        Uncompressed, as before: these are fp16 activations, zlib buys a few per cent on them
        and costs seconds per set — measured at 4.5 s against 0.4 s on the accumulators, which
        is why `save_accumulators` is the one that got smaller rather than slower.
        """
        p = self.path(name)
        arrays = {"primary": primary.astype(np.float16),
                  "meta": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)}
        if resid is not None:
            arrays["primary_resid"] = resid.astype(np.float16)
        np.savez(p, **arrays)
        return {"path": str(p), "shape": list(primary.shape),
                "resid_shape": list(resid.shape) if resid is not None else None,
                "bytes": p.stat().st_size}


def save_accumulators(path: pathlib.Path, acc: dict, half: bool = False) -> dict:
    """Named Welford accumulators to one compressed npz, in raw units.

    `half` writes the mean in fp16, which is what D1a §284 specifies for the record ("means
    in fp16") and what A21's 363 GB is arithmetic on. It is set on A and B sets and left off
    for the C yardstick, which §268 keeps whole. The pilot measured **zero fp16 overflow at
    every reference, layer and site**, and every statistic in this study is rank- or
    cosine-based, so the narrowing is not in any reported number's path.

    An accumulator built with `spread=False` has no `m2` and no `active`; those keys are then
    absent from the file rather than written as zeros, because a zero variance would read as
    a measurement and not as an omission.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for key, w in acc.items():
        if w.mean is None:
            continue
        flat[f"{key}|n"] = np.array([w.n])
        flat[f"{key}|mean"] = w.mean.astype(np.float16 if half else np.float32)
        if w.m2 is not None:
            flat[f"{key}|m2"] = w.m2.astype(np.float32)
        if w.active is not None:
            flat[f"{key}|active"] = w.active.astype(np.int32)
    np.savez_compressed(path, **flat)
    return {"path": str(path), "keys": len(acc), "bytes": path.stat().st_size,
            "mean_dtype": "float16" if half else "float32",
            "spread_kept": any(w.m2 is not None for w in acc.values())}


def read_battery(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in gzip.open(path, "rt")]


RETIRED_RECORD_DIR = "$ATLAS_DATA/record"


def refuse_retired_record_dir(path) -> "pathlib.Path":
    """For the **recorder**, which creates its tree: refuse the retired path, do not require it.

    `require_record_dir` is for readers and insists the tree is there. A writer must not be held
    to that -- a fresh recording legitimately starts with nothing -- but it must also not be
    allowed to recreate a record tree at the path that caused A50, where a later reader would
    take whatever it finds for the phase recording.
    """
    import os as _o
    import pathlib as _p
    q = _p.Path(_o.path.expanduser(str(path)))
    if q == _p.Path(_o.path.expanduser(RETIRED_RECORD_DIR)):
        raise SystemExit(
            f"refusing to use {q} as a record tree.\n"
            "  That path held the PILOT recording at the PHASE recording's path (A50) and was "
            "retired on 20 Oct. Writing a new recording there would recreate the confusion.\n"
            "  Set PB_RECORD_DIR to an explicit location.")
    return q


def require_record_dir(path) -> "pathlib.Path":
    """Fail with a sentence rather than fall back to a record tree that is no longer there.

    `$ATLAS_DATA/record` held the **pilot** recording at the **phase** recording's path
    (A50) -- which is how A48 came to compare two different recordings of the same checkpoints.
    On 20 Oct it was verified file by file against the dataset's `pilot/record` (3,400 of 3,400
    byte-identical: 2,116 by LFS SHA-256, 1,284 by git blob hash) and deleted, keeping only
    `untrained` at `$ATLAS_DATA/pilot_record/untrained`.

    A pass that silently found nothing there, or found the pilot and took it for the phase
    record, is the failure this replaces.
    """
    import pathlib as _p
    q = refuse_retired_record_dir(path)
    if q.is_dir():
        return q
    raise SystemExit(
        f"no record tree at {q}.\n"
        "  The local copy that used to sit at $ATLAS_DATA/record was the PILOT "
        "recording at the PHASE recording's path; it was verified against the dataset's "
        "pilot/record and deleted on 20 Oct, keeping only `untrained` at "
        "$ATLAS_DATA/pilot_record/untrained.\n"
        "  Set PB_RECORD_DIR to a real record tree, or read from the dataset "
        "(experiments/atlas.py:dataset_record streams it and ignores any local copy).")
