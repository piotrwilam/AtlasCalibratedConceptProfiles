"""Compute profiles and concept profiles for one reference — §5, as amended by S15.

    python experiments/compute_profiles.py --reference untrained --repo $ATLAS_DATASET

**What S15 added, and why in one pass over both twins.** Until 9 Oct this wrote twelve arrays
for one twin: the pooled-over-scope profile, its SE, median, raw mean, bootstrap and core
means, for A and B, plus the concept profile. Three things were missing and all three are
recoverable from the record without re-recording:

* **the scope-restricted profiles.** The recorder wrote a second accumulator family keyed by
  core *and* scope at every construct of every reference. For the 24 constructs whose
  construct-token piece differs between module and indented scope, the *primary* profile is
  the indented scopes pooled (Cowork, read-site rule, NOTES §3a item 2 option iii) and the
  module profile is a separate TOKEN reading -- measured on `For` at `pretrain_iter_0484865`,
  cosine 0.6688 between them at layer 12. The pooled profile this script used to write is
  neither of those objects.
* **twin 2.** Stored as the concept profile only, not a second full tree: R1's turnover curve
  compares twin 1's carrier set at t with twin 2's at the checkpoint nearest 1.5t, so the set
  must be in the tree, but nothing reads twin 2's cores, means, SEs or bootstrap. Everything
  else any pre-registered reading takes from twin 2 is a scalar, and those go in `TWINS.json`.
* **float32 on the way out** (Piotr, 9 Oct) -- see `narrow`.

Both twins in **one** pass because the accumulators are the expensive part: a reference holds
about 25 GB of them across the two twins, and two passes would fetch that twice.
"""

import argparse
import gzip
import json
import os
import pathlib
import time

import numpy as np

from profilebattery.profiles.profile import (build, concept, core_bootstrap, recorder_order,
                                             from_accumulators, pooled)
from profilebattery.profiles.scoped import GROUPS, scope_varying, scoped
from profilebattery.profiles.yardstick import EPS_GRID, Yardstick
from profilebattery.paths import data_root

REC = pathlib.Path(os.path.expanduser(
    os.environ.get("PB_RECORD_DIR", str(data_root() / "record"))))
#: **The PHASE battery, not the pilot's.** This pointed at `battery/` -- the 30-construct pilot,
#: 212 sets -- while `TODO-Settle` S1's eight references are recorded against `phase_battery/`,
#: 188 constructs and 964 sets, prompt hash 570b42db. The same defect cost `run_sweep` four
#: minutes of A100 time on 4 Oct, where the set count gave it away; here it would not have
#: aborted at all. It would have standardised sweep references against the wrong prompts and
#: written profiles that every later reading consumes. Overridable, so the pilot's own profiles
#: can still be recomputed deliberately rather than by default.
BATT = pathlib.Path(os.path.expanduser(
    os.environ.get("PB_BATTERY_DIR", str(data_root() / "phase_battery"))))
OUT = pathlib.Path(os.path.expanduser(
    os.environ.get("PB_PROFILES_DIR", str(data_root() / "profiles"))))


def load_set(ref: str, twin: str, construct: str, kind: str, store=None,
             accumulators_only: bool = False):
    """(per-prompt array or None, rows, accumulators or None).

    DECIDED 6 keeps per-prompt arrays at `0003000` only, so at the other checkpoints the
    shard is present but empty and the profile comes from the per-core, per-cell
    accumulators instead. The two routes give the same profile — cosine 1.000000,
    max|diff| 0.0, asserted by a test — so this is a storage choice, not a method one.
    """
    batt = BATT / twin / f"{construct}_{kind}.jsonl.gz"
    if not batt.exists():
        return None, None, None
    # in the order the per-prompt arrays hold them, not the jsonl's
    rows = recorder_order([json.loads(l) for l in gzip.open(batt, "rt")])
    # local first, then the dataset -- `RecordStore` is the same route `run_validation` uses,
    # so one reference's accumulators are fetched on demand rather than all 213 GB up front
    def at(name):
        if store is not None:
            return store.path(name)
        q = REC / ref / name
        return q if q.exists() else None
    # **`accumulators_only` exists because the per-prompt route is free locally and ruinous
    # over the network.** A per-prompt reference holds 78.7 GB of shards -- 820 non-empty, up to
    # 290 MB each -- and this function fetched the shard *first* and preferred it whenever
    # present. Three of S1's eight references are per-prompt, so the first run spent 90 minutes
    # downloading per-prompt data it did not need and would have added ~236 GB to a 213 GB job.
    # The two routes give the same profile -- cosine 1.000000, max|diff| 0.0, asserted by a test
    # -- so skipping the shard costs nothing but the bytes.
    acc = at(f"{twin}__{construct}_{kind}_acc.npz")
    if accumulators_only:
        return None, rows, (np.load(acc) if acc is not None else None)
    shard = at(f"{twin}__{construct}_{kind}.npz")
    primary = None
    if shard is not None:
        arr = np.load(shard)["primary"]
        primary = arr if arr.shape[0] else None
    return primary, rows, (np.load(acc) if acc is not None else None)


def narrow(payload: dict) -> dict:
    """float32 on the way out (Piotr, 9 Oct), applied to every float array wider than 4 bytes.

    The arithmetic stays float64 -- the standardisation, the count-weighted combination over
    cells, the bootstrap -- and only the stored result narrows. Three reasons that costs
    nothing: the inputs are **fp16** means, so float32 is still three orders of magnitude finer
    than the data; every statistic downstream is rank- or cosine-based; and this is **one**
    rounding from the float64 that was computed, not the double rounding item 101 ran into,
    where the original had already been narrowed once before being narrowed again.

    A function rather than an inline comprehension because `narrow_profiles.py` converts the
    already-written references with the same rule, and two copies of a precision rule is how a
    tree ends up in mixed precision -- the one state item 101 exists to keep it out of.
    Non-arrays and arrays already at or below 4 bytes pass through untouched, so fp16 means are
    not widened and the integer counts are not touched.
    """
    return {k: (v.astype(np.float32)
                if getattr(v, "dtype", None) is not None
                and v.dtype.kind == "f" and v.dtype.itemsize > 4 else v)
            for k, v in payload.items()}


def make(primary, rows, acc, ys, construct, kind):
    if primary is not None:
        return build(primary, rows, ys, construct, kind)
    if acc is not None:
        return from_accumulators(acc, ys, construct, kind, stream="h")
    return None


def twin_scalars(A, A2, B, B2, c: str, n_boot: int,
                 acca, accb, acc2a, acc2b, ys, ys2, varying: bool) -> dict:
    """What the partner twin is read for, once its profiles are no longer stored.

    `run_validation` takes `twin_dist = twins(a1, a2)["median_distance"]` -- one float -- and
    checks 8, 9 and the nu sensitivity all divide by it. Twin agreement and carrier overlap
    are the watched numbers. Check 6 takes both profiles and both bootstraps and returns
    scalars. None of that needs the partner's arrays to be in the tree; R1's carrier set does,
    and that is the concept profile, which is stored.

    For the scope-varying constructs the unit is **also** computed on the indented object,
    because that is the profile check 9's step is re-read on and a unit taken on the pooled
    profile would not be the unit of the thing being measured.
    """
    from profilebattery.validation.checks import calibration, twins
    from profilebattery.validation.watched import carrier_overlap
    t = twins(A, A2)
    d1, d2 = A.mean - B.mean, A2.mean - B2.mean
    out = {"twin_unit": t["median_distance"],
           "twin_agreement": t["median_cosine"],
           "carrier_overlap": carrier_overlap(d1, d2),
           "concept_norm": float(np.median(np.linalg.norm(d1, axis=1))),
           "calibration": calibration(A, A2, core_bootstrap(A, 200),
                                      core_bootstrap(A2, 200))}
    if varying and acca is not None and acc2a is not None:
        ai = scoped(acca, ys, c, "A", GROUPS["indented"])
        a2i = scoped(acc2a, ys2, c, "A", GROUPS["indented"])
        ti = twins(ai, a2i)
        out["indented"] = {"twin_unit": ti["median_distance"],
                           "twin_agreement": ti["median_cosine"]}
        if accb is not None and acc2b is not None:
            bi = scoped(accb, ys, c, "B", GROUPS["indented"])
            b2i = scoped(acc2b, ys2, c, "B", GROUPS["indented"])
            out["indented"]["carrier_overlap"] = carrier_overlap(
                ai.mean - bi.mean, a2i.mean - b2i.mean)
    return out


#: the seven arrays a scope grouping contributes. Means and SEs for both sides, the concept
#: profile and its SE, and A's core bootstrap -- `B_boot` is not among them because no
#: pre-registered reading takes a bootstrap interval on the B side alone.
SCOPE_ARRAYS = ("A_mean", "A_se", "A_boot", "B_mean", "B_se", "concept", "concept_se")


def scope_block(acc_a, acc_b, ys, c: str, n_boot: int, groups) -> dict:
    """The per-scope arrays for one construct, for each grouping in `groups`.

    `scoped` merges function and method into one 8-core indented profile rather than sixteen
    pseudo-cores -- see `profilebattery.profiles.scoped.ScopeView`, which exists because the
    obvious implementation of that merge silently keeps half the prompts.
    """
    out, meta = {}, {}
    for name in groups:
        scopes = GROUPS[name]
        a = scoped(acc_a, ys, c, "A", scopes)
        b = scoped(acc_b, ys, c, "B", scopes) if acc_b is not None else None
        k = concept(a, b) if b is not None else None
        got = {"A_mean": a.mean, "A_se": a.se, "A_boot": core_bootstrap(a, n_boot)}
        if b is not None:
            got |= {"B_mean": b.mean, "B_se": b.se,
                    "concept": k["delta"], "concept_se": k["se"]}
        out |= {f"{q}_{name}": got[q] for q in SCOPE_ARRAYS if q in got}
        meta[name] = {"n_a": a.n, "cores_a": len(a.core_means),
                      "n_b": b.n if b is not None else None}
        out[f"_prof_{name}"] = (a, b)        # kept for the caller, stripped before saving
    return out, meta


def core_block(block: dict, groups) -> dict:
    """Per-scope core means, for the constructs that need a bootstrap on their own scope."""
    out = {}
    for name in groups:
        a, b = block[f"_prof_{name}"]
        out[f"A_cores_{name}"] = np.stack(
            [a.core_means[q] for q in sorted(a.core_means)]).astype(np.float32)
        if b is not None:
            out[f"B_cores_{name}"] = np.stack(
                [b.core_means[q] for q in sorted(b.core_means)]).astype(np.float32)
    return out


#: worker state. Set once per process by `setup`, so a pool worker loads the two yardsticks
#: -- about 137 MB each -- once, instead of having them pickled with every construct.
_W: dict = {}


def setup(reference, twin, partner, n_boot, arrays, no_scope_cores, varying,
          ys_path, ys2_path, out, out2, repo, cache_root=None):
    """Initialise one process. The serial path calls it too, so there is one code path."""
    store = None
    if repo:
        from profilebattery.record.remote import RecordStore, quiet_hub
        quiet_hub()
        store = RecordStore(REC, reference, repo)
        if cache_root:
            store.use_cache(cache_root)
    _W.update(reference=reference, twin=twin, partner=partner, n_boot=n_boot,
              arrays=arrays, no_scope_cores=no_scope_cores, varying=set(varying),
              ys=Yardstick.from_npz(ys_path),
              ys2=Yardstick.from_npz(ys2_path) if ys2_path else None,
              out=pathlib.Path(out), out2=pathlib.Path(out2) if out2 else None,
              store=store)


def one(c: str):
    """Every array and scalar of one construct. Returns `(name, row, twin entry)`.

    **One construct is one unit of work, and they are independent** -- each writes its own two
    npz files and contributes one summary row -- which is what makes `--jobs` safe
    rather than merely fast, and `--jobs 4` was asserted to write files **byte-identical** to
    `--jobs 1` over 40 npz, with the same `SUMMARY.json` and `TWINS.json`.

    Measured: this body takes about 66s for a scope-varying construct on Colab's CPU against
    7s on the laptop, and a reference is 188 of them, so serial it is about 95 minutes a
    reference and 4.5 days for 61. Of the 7s, about 2.7s is arithmetic, 2.4s is
    `savez_compressed` -- which buys 7.8% because float arrays barely deflate, and level 1 is
    no faster since deflate still scans 100 MB -- and the rest is twin 2 and the scalar table.

    **The scaling saturates early.** On this laptop, 20 constructs: 86s at 1 worker, 57s at 2
    (1.5x), 40s at 4 (2.15x), 37s at 8 (2.3x). Its numpy is on Accelerate, which already uses
    several cores inside a call, so the serial baseline is not single-threaded and the headroom
    a process pool can add is small. **That number does not transfer**: a host whose numpy is
    effectively single-threaded has its whole core count available instead, which is why the
    worker count is measured on the host that will run it rather than assumed from here.
    """
    W = _W
    ys, ys2, store = W["ys"], W["ys2"], W["store"]
    out, out2, varying = W["out"], W["out2"], W["varying"]
    n_boot, s15 = W["n_boot"], W["arrays"] == "s15"
    ref, twin, partner = W["reference"], W["twin"], W["partner"]
    twin_entry, t_c = None, time.time()

    pa, ra, acca = load_set(ref, twin, c, "A", store, True)
    A = make(pa, ra, acca, ys, c, "A")
    if A is None:
        return c, None, None
    row = {"n_a": A.n, "skipped_a": A.skipped, "cells": A.cells,
           "source": "per-prompt" if pa is not None else "accumulators",
           "cores": len(A.core_means),
           "norm": A.norm().tolist(), "top64": A.top_share().tolist(),
           "raw_norm": np.linalg.norm(A.raw_mean, axis=1).tolist()}
    # Per-core means for both sides, float32. The go/no-go's twin-noise unit is the larger of
    # the twin difference and a core bootstrap (DECIDED 3), and only the bootstrap carries
    # core-sampling variance: this battery's twins share all eight cores, so their differences
    # cannot supply it. A and B have their own core sets, resampled independently.
    payload = {"A_mean": A.mean, "A_se": A.se, "A_median": A.median,
               "A_raw": A.raw_mean,
               "A_boot": core_bootstrap(A, n_boot),
               "A_cores": np.stack([A.core_means[k]
                                    for k in sorted(A.core_means)]).astype(np.float32)}
    pb, rb, accb = load_set(ref, twin, c, "B", store, True)
    B = make(pb, rb, accb, ys, c, "B")
    if B is not None:
        k = concept(A, B)
        payload |= {"B_mean": B.mean, "B_se": B.se, "B_raw": B.raw_mean,
                    "B_cores": np.stack([B.core_means[q]
                                         for q in sorted(B.core_means)]).astype(np.float32),
                    "concept": k["delta"], "concept_se": k["se"]}
        row |= {"n_b": B.n,
                "concept_norm": np.linalg.norm(k["delta"], axis=1).tolist(),
                "beyond_3se": [int((np.abs(k["delta"][l]) > 3 * k["se"][l]).sum())
                               for l in range(k["delta"].shape[0])]}
    # the scope block. Every construct gets the three scopes -- register row C3 asks the
    # role-or-context question of all of them -- and the 24 whose token differs by scope also
    # get `indented`, their primary profile, with per-scope cores so the bootstrap interval can
    # be taken on that object rather than on the pooled one.
    if s15 and acca is not None:
        groups = ["module", "function", "method"] + (["indented"] if c in varying else [])
        block, meta = scope_block(acca, accb, ys, c, n_boot, groups)
        if c in varying and not W["no_scope_cores"]:
            block |= core_block(block, groups)
        payload |= {q: v for q, v in block.items() if not q.startswith("_prof_")}
        row["scopes"] = meta
    np.savez_compressed(out / f"{c}.npz", **narrow(payload))

    # the partner twin: the concept profile, and the indented one where it is primary, plus the
    # scalars every other pre-registered reading takes from it.
    if ys2 is not None:
        p2a, r2a, acc2a = load_set(ref, partner, c, "A", store, True)
        p2b, r2b, acc2b = load_set(ref, partner, c, "B", store, True)
        A2 = make(p2a, r2a, acc2a, ys2, c, "A")
        B2 = make(p2b, r2b, acc2b, ys2, c, "B")
        if A2 is not None and B2 is not None:
            k2 = concept(A2, B2)
            pay2 = {"concept": k2["delta"], "concept_se": k2["se"]}
            if s15 and acc2a is not None and c in varying:
                a2i = scoped(acc2a, ys2, c, "A", GROUPS["indented"])
                b2i = scoped(acc2b, ys2, c, "B", GROUPS["indented"])
                pay2["concept_indented"] = concept(a2i, b2i)["delta"]
            np.savez_compressed(out2 / f"{c}.npz", **narrow(pay2))
            twin_entry = twin_scalars(A, A2, B, B2, c, n_boot,
                                      acca, accb, acc2a, acc2b, ys, ys2,
                                      c in varying and s15)
    # eps-stability (check 8): the profile recomputed at three floors
    for eps in EPS_GRID:
        p2 = make(pa, ra, acca, Yardstick(ys.cells, eps=eps), c, "A")
        row[f"top64_eps{eps}"] = p2.top_share().tolist()
    row["seconds"] = round(time.time() - t_c, 2)
    return c, row, twin_entry


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--twin", default="twin1",
                    help="the twin that gets the full payload; its partner gets the "
                         "concept profile only")
    ap.add_argument("--partner", default="twin2")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--repo", default=None,
                    help="fetch what is not local from this dataset, e.g. $ATLAS_DATASET")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N constructs only, for a timed trial before committing to 61 "
                         "references. Writes to a `_trial` suffix so a trial can never be "
                         "mistaken for a reference -- S1 taught this one.")
    ap.add_argument("--accumulators-only", action="store_true",
                    help="never fetch the per-prompt shards: the same profile, without the "
                         "78.7 GB a per-prompt reference carries. What TODO-Settle S1 wants.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes over constructs; 0 means one per cpu. Constructs are "
                         "independent and each writes its own files. Serial is about 95 min a "
                         "reference on Colab's cpu and 4.5 days for 61, which is why this "
                         "exists. Needs --accumulators-only.")
    ap.add_argument("--only", default=None,
                    help="comma-separated constructs, for a trial that deliberately includes "
                         "the scope-varying ones -- a trial over the first N alphabetically "
                         "exercises neither the indented profile nor the per-scope cores, and "
                         "so under-reports the size per reference")
    ap.add_argument("--no-scope-cores", action="store_true",
                    help="skip the per-scope core means. They are 63.5%% of a scope-varying "
                         "construct's bytes and 75 GB over 61 references, and the "
                         "bootstrap interval they were wanted for is already stored as "
                         "`A_boot_<scope>`; they only permit a *different* resampling later, "
                         "and stay recoverable from the record. Open decision, so a flag.")
    ap.add_argument("--arrays", choices=("twelve", "s15"), default="s15",
                    help="`twelve` reproduces the pre-S15 payload, for comparison only")
    ap.add_argument("--no-partner", action="store_true",
                    help="skip the partner twin entirely (it carries R1's turnover curve, so "
                         "this is for trials, not for a reference)")
    args = ap.parse_args()
    from profilebattery.record.store import require_record_dir
    require_record_dir(REC)   # after parsing, so --help still works (A50/A51)

    store = None
    if args.repo:
        from profilebattery.record.remote import RecordStore, quiet_hub
        quiet_hub()
        store = RecordStore(REC, args.reference, args.repo)

    def yardstick_path(twin):
        """The path, not the object: a pool worker loads its own copy once (see `setup`)."""
        q = (store.path(f"{twin}__C_cells.npz") if store is not None
             else (REC / args.reference / f"{twin}__C_cells.npz"))
        if q is None or not pathlib.Path(q).exists():
            raise SystemExit(f"no yardstick for {args.reference}/{twin}: "
                             f"{twin}__C_cells.npz is neither local nor in "
                             f"{args.repo or 'any repo'}")
        return str(q)

    # **prefetch, because one file at a time is the wrong shape for this record.** The loop
    # below asks for four accumulators per construct, and `hf_hub_download` is serial: measured
    # 5.7 MB/s a file from this machine, so a reference's ~25 GB would take about 70 minutes of
    # pure transfer and 61 references about 70 hours. `RecordStore.prefetch` fetches them in
    # parallel into the same cache `path()` reads, so nothing else changes.
    if store is not None:
        names = [f"{t}__C_cells.npz" for t in
                 ([args.twin] if args.no_partner else [args.twin, args.partner])]
        want_c = sorted(q.name[:-len("_A.jsonl.gz")]
                        for q in (BATT / args.twin).glob("*_A.jsonl.gz"))
        if args.only:
            want_c = [q.strip() for q in args.only.split(",") if q.strip()]
        elif args.limit:
            want_c = want_c[:args.limit]
        for t in ([args.twin] if args.no_partner else [args.twin, args.partner]):
            for c in want_c:
                for kind in ("A", "B"):
                    names.append(f"{t}__{c}_{kind}_acc.npz")
        store.prefetch(names)

    ys_path = yardstick_path(args.twin)
    ys2_path = None if args.no_partner else yardstick_path(args.partner)

    suffix = "_trial" if (args.limit or args.only) else ""
    out = OUT / (args.reference + suffix) / args.twin
    out.mkdir(parents=True, exist_ok=True)
    out2 = OUT / (args.reference + suffix) / args.partner
    if ys2_path is not None:
        out2.mkdir(parents=True, exist_ok=True)

    # **from the battery, not from the per-prompt shards.** This globbed
    # `{twin}__*_A.npz` -- the per-prompt files -- and A21 keeps those at four references only,
    # as present-but-empty placeholders elsewhere. On five of S1's eight references that glob
    # returns almost nothing, so the run would have written almost no profiles and said nothing
    # about it. The battery is the authority on which constructs exist.
    constructs = sorted(q.name[:-len("_A.jsonl.gz")]
                        for q in (BATT / args.twin).glob("*_A.jsonl.gz"))
    varying = set(scope_varying()) if args.arrays == "s15" else set()
    print(f"reference {args.reference} / {args.twin}"
          + (f" + {args.partner} (concept only)" if ys2_path is not None else "")
          + f": {len(constructs)} constructs from {BATT.name}"
          + (f", fetching from {args.repo}" if args.repo else ", local only")
          + (" (accumulators only)" if args.accumulators_only else ""), flush=True)
    print(f"arrays {args.arrays}; {len(varying)} scope-varying constructs get the indented "
          f"primary profile"
          + ("" if args.no_scope_cores else " and per-scope cores"))
    if args.only:
        want = [q.strip() for q in args.only.split(",") if q.strip()]
        missing = [q for q in want if q not in constructs]
        if missing:
            raise SystemExit(f"--only names constructs the battery does not have: {missing}")
        constructs = want
        print(f"--only: {len(constructs)} named constructs, "
              f"{len([q for q in constructs if q in varying])} of them scope-varying\n")
    elif args.limit:
        constructs = constructs[:args.limit]
        print(f"--limit {args.limit}: a trial, written to {out.parent.name}\n")

    cache_root = str(store._cache_root) if (store is not None
                                            and store._cache_root is not None) else None
    setup(args.reference, args.twin, args.partner, args.n_boot, args.arrays,
          args.no_scope_cores, varying, ys_path, ys2_path, out, out2, args.repo, cache_root)
    ys = _W["ys"]
    print(f"yardstick {ys.summary()['layers']} layers x {ys.summary()['cells']} cells")

    jobs = args.jobs or (os.cpu_count() or 1)
    if jobs > 1 and not args.accumulators_only:
        raise SystemExit("--jobs needs --accumulators-only: the per-prompt route would have "
                         "every worker fetch its own shards.")
    print(f"{jobs} worker(s) over {os.cpu_count()} cpu(s)\n", flush=True)

    summary, twin_table, t0 = {}, {}, time.time()

    def record(got):
        c, row, twin_entry = got
        if row is None:
            return
        summary[c] = row
        if twin_entry is not None:
            twin_table[c] = twin_entry
        print(f"  {c:16} n={row['n_a']:4} cores={row['cores']} "
              f"norm@L12={row['norm'][12]:7.2f} top64@L12={row['top64'][12]:.3f}"
              + (f" concept@L12={row['concept_norm'][12]:6.2f}"
                 if "concept_norm" in row else "")
              + f" {row['seconds']:5.1f}s"
              + f"   [{len(summary)}/{len(constructs)}]", flush=True)

    if jobs == 1:
        for c in constructs:
            record(one(c))
    else:
        # **a process pool, not threads**: this is numpy-bound and the GIL would serialise it.
        # Each worker loads the yardsticks once in `setup`; constructs are independent and each
        # writes its own files, so there is nothing to merge but the summary rows.
        from concurrent.futures import ProcessPoolExecutor
        initargs = (args.reference, args.twin, args.partner, args.n_boot, args.arrays,
                    args.no_scope_cores, sorted(varying), ys_path, ys2_path,
                    str(out), str(out2) if out2 is not None else None, args.repo, cache_root)
        with ProcessPoolExecutor(max_workers=jobs, initializer=setup,
                                 initargs=initargs) as ex:
            for got in ex.map(one, constructs, chunksize=1):
                record(got)
    (out / "SUMMARY.json").write_text(json.dumps(
        {"reference": args.reference, "twin": args.twin, "arrays": args.arrays,
         "partner": None if ys2_path is None else args.partner,
         "scope_varying": sorted(varying),
         "yardstick": ys.summary(), "eps_grid": list(EPS_GRID),
         "seconds": round(time.time() - t0, 1), "constructs": summary}, indent=2))
    if twin_table:
        (out.parent / "TWINS.json").write_text(json.dumps(
            {"reference": args.reference, "twin": args.twin, "partner": args.partner,
             "note": "the scalars every pre-registered reading takes from the partner twin; "
                     "`twin_unit` is what checks 8 and 9 divide by",
             "constructs": twin_table}, indent=2))
    print(f"\n{len(summary)} constructs in {time.time()-t0:.0f}s -> {out}")
    if args.limit or args.only:
        # **weighted by the real mix, not by the trial's.** A trial over named constructs is
        # deliberately rich in the 24 scope-varying ones -- they carry the indented block and
        # the per-scope cores, and their files are three times the size -- so multiplying the
        # trial's mean by 188 overstates a reference by half. A trial over the first N
        # alphabetically contains none of them and understates it by as much. This is the same
        # error as extrapolating an ETA from the latest window, which cost four wrong estimates
        # on 7 Oct, so the estimate is built from the two classes and their true counts.
        import collections
        sizes = collections.defaultdict(list)
        for twin, d in (("twin1", out), ("twin2", out2 if ys2_path is not None else None)):
            if d is None:
                continue
            for q in d.glob("*.npz"):
                sizes[(twin, q.stem in varying)].append(q.stat().st_size)
        n_var, n_plain = len(varying), len(constructs_all := sorted(
            q.name[:-len("_A.jsonl.gz")] for q in (BATT / args.twin).glob("*_A.jsonl.gz")))
        n_plain -= n_var
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None   # noqa: E731
        total, missing = 0, []
        for twin in ("twin1", "twin2"):
            if twin == "twin2" and ys2_path is None:
                continue
            for is_var, count in ((True, n_var), (False, n_plain)):
                m = mean(sizes[(twin, is_var)])
                if m is None:
                    missing.append(f"{twin}/{'scope-varying' if is_var else 'plain'}")
                    continue
                total += m * count
        print(f"\nsize per reference, weighted by the real mix "
              f"({n_var} scope-varying, {n_plain} plain):")
        for (twin, is_var), xs in sorted(sizes.items()):
            print(f"  {twin}/{'scope-varying' if is_var else 'plain':14} "
                  f"n={len(xs)} measured, mean {mean(xs)/2**20:7.2f} MB")
        if missing:
            print(f"  NOT MEASURED: {', '.join(missing)} -- the estimate below is a floor, "
                  f"not an estimate. Re-run the trial with --only naming both classes.")
        print(f"  -> {total/2**30:6.2f} GiB per reference, {total*61/2**30:6.0f} GiB "
              f"({total*61/1e9:.0f} GB) for 61 references")
        print(f"  -> account would reach about {2422.7 + total*61/1e9:.0f} GB "
              f"against the 3,000 GB cap")
        print("Compare this with the approved figure BEFORE starting the full run.")


if __name__ == "__main__":
    main()
