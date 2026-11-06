"""The twin-2 pass: the intervals for curves 1 and 5, which the derived tree cannot give.

**Why it exists.** The derived tree carries per-core arrays for **twin 1 only** -- a twin-2 npz
holds `concept`, `concept_se` and `concept_indented` and nothing else, which is the array set as
approved. So the first bootstrap pass could give curves 2, 3 and 6 their intervals and could not
give the **twin ceiling** or the **twin agreement** theirs, because both are
between-battery quantities. Those come from twin 2's **accumulators in the record**:
`record/<ref>/twin2__<c>_{A,B}_acc.npz`, with `record/<ref>/twin2__C_cells.npz` as the yardstick.

**Per-core profiles are computed in memory and never stored**, which is the Settle chat's ruling
of 13 Oct: only interval summaries are written. `from_accumulators` is the existing, validated
route -- the same one `compute_profiles` used to build twin 1 for S15, at `stream="h"` -- so the
twin-2 profiles are built by the same code that built the twin-1 ones and not by a second
implementation of the same aggregation.

**C0.5 across twins and references.** One core-index set per draw, drawn once over the construct's
**core count** and applied to twin 1 and twin 2 at every reference. The twin-1 cores come from
the derived tree's `A_cores`/`B_cores`; the twin-2 cores from the accumulators. The two twins
share all eight A cores on this battery and none of the 720 prompts, which is why the raw
quantity is the twin unit (item 108), and it is also why one index set is meaningful across them.

**What it measures, per draw.**

* **the twin ceiling** -- the carrier overlap between twin 1's contrast and twin 2's at the same
  checkpoint, which is what curve 1 divides by;
* **the twin agreement** -- the cosine of the two contrasts in the product space.

From their draw distributions come the intervals curve 1 and curve 5 are missing.

**A bias these intervals carry, measured before the pass was run, and it bears on how they may
be used.** Both quantities are **nonlinear in the resampled means**, so their bootstrap
distributions sit *below* the plug-in value rather than around it. Checked on `For` at
`pretrain_iter_0484865`:

| | plug-in (carrier index and `TWINS.json` agree exactly) | bootstrap |
|---|---|---|
| twin ceiling | **0.8717** | mean 0.8436, SE 0.0314, **[0.7716, 0.8935]** -- covers it |
| twin agreement | **0.9950** | mean 0.9743, SE 0.0109, **[0.9467, 0.9891]** -- **does not** |

Resampling cores with replacement makes each draw's contrast a noisier estimate, and the overlap
or cosine of two noisier vectors is systematically smaller: the agreement's whole interval falls
below its plug-in value. **So these are intervals on the resampled quantity, not confidence
intervals for the plug-in one**, and substituting the bootstrap mean for the plug-in ceiling in
curve 1's denominator would raise every ratio. Curve 1 therefore keeps the **plug-in ceiling** as
its denominator, as it already does, and takes from this pass only the **interval width**.
Whether a bias correction belongs in the contract is the Settle chat's to rule; the draw
distributions are stored so any correction can be applied without re-running the pass.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time

import numpy as np
from profilebattery.paths import data_root, dataset_repo

REPO = dataset_repo()
D = data_root()
PREFIX = "validation/bootstraps"
DRAWS = 1_000
SEED = 20261013
K = 0.05


def weights(idx_pairs, n: int, which: int) -> np.ndarray:
    """`(draws, n)` resampling weights -- a draw's mean is a weighted sum of the cores."""
    w = np.zeros((len(idx_pairs), n), dtype=np.float64)
    for d, pair in enumerate(idx_pairs):
        np.add.at(w[d], np.asarray(pair[which]) % n, 1.0)
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1.0)


def draw_indices(n_a: int, n_b: int, draws: int = DRAWS, seed: int = SEED):
    """One core-index set per draw, reused across twins and references (C0.5)."""
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, n_a, n_a), rng.integers(0, n_b, n_b)) for _ in range(draws)]


def contrasts(a: np.ndarray, b: np.ndarray, idx_pairs, lo: int, hi: int) -> np.ndarray:
    """`(chunk, units)` -- one contrast profile per draw, as a matrix product."""
    af = np.asarray(a, np.float64).reshape(a.shape[0], -1)
    bf = np.asarray(b, np.float64).reshape(b.shape[0], -1)
    wa = weights(idx_pairs[lo:hi], af.shape[0], 0)
    wb = weights(idx_pairs[lo:hi], bf.shape[0], 1)
    return wa @ af - wb @ bf


def ceiling_and_agreement(v1: np.ndarray, v2: np.ndarray, k: float = K):
    """`(overlap, cosine)` per row: the carrier overlap and the product-space cosine.

    The overlap is computed from **boolean masks of the top-k mass**, not from sorted index
    arrays: a mask-and-count is one `&` and one `sum` per row, where `intersect1d` per draw is a
    sort per draw. The two agree exactly -- the mask is the same selection -- and the mask form
    is what makes 1,000 draws affordable.
    """
    kk = max(1, int(round(k * v1.shape[1])))
    def mask(v):
        f = v * v
        cut = np.partition(f, f.shape[1] - kk, axis=1)[:, f.shape[1] - kk][:, None]
        m = f >= cut
        # a tie at the cut can select more than kk; trim deterministically by index order so
        # two references' masks are comparable
        over = m.sum(axis=1) - kk
        if np.any(over > 0):
            for i in np.nonzero(over > 0)[0]:
                idx = np.nonzero(m[i])[0]
                m[i, idx[kk:]] = False
        return m
    m1, m2 = mask(v1), mask(v2)
    inter = (m1 & m2).sum(axis=1).astype(float)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    dot = np.einsum("ij,ij->i", v1, v2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where((n1 > 0) & (n2 > 0), dot / (n1 * n2), np.nan)
    return inter / kk, cos


def interval(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    good = x[np.isfinite(x)]
    if good.size < 2:
        return {"mean": None, "se": None, "lo": None, "hi": None, "n_finite": int(good.size)}
    return {"mean": float(good.mean()), "se": float(good.std(ddof=1)),
            "lo": float(np.quantile(good, 0.025)), "hi": float(np.quantile(good, 0.975)),
            "n_finite": int(good.size)}


def one_construct(t1a, t1b, t2a, t2b, draws: int = DRAWS, chunk: int = 50) -> dict:
    """The ceiling and agreement draw distributions for one (reference, construct)."""
    pairs = draw_indices(t1a.shape[0], max(t1b.shape[0], t2b.shape[0]), draws)
    ceil, agree = [], []
    for s0 in range(0, draws, chunk):
        s1 = min(s0 + chunk, draws)
        v1 = contrasts(t1a, t1b, pairs, s0, s1)
        v2 = contrasts(t2a, t2b, pairs, s0, s1)
        c, g = ceiling_and_agreement(v1, v2)
        ceil.append(c)
        agree.append(g)
    return {"ceiling": interval(np.concatenate(ceil)),
            "agreement": interval(np.concatenate(agree)),
            "n_cores": {"A_twin1": int(t1a.shape[0]), "B_twin1": int(t1b.shape[0]),
                        "A_twin2": int(t2a.shape[0]), "B_twin2": int(t2b.shape[0])}}


def read_twin2(store_dir: pathlib.Path, ref: str, c: str, ys):
    """Twin 2's per-core A and B profiles, built in memory and not stored."""
    from profilebattery.profiles.profile import from_accumulators
    out = []
    for kind in ("A", "B"):
        with np.load(store_dir / f"twin2__{c}_{kind}_acc.npz") as z:
            p = from_accumulators(z, ys, c, kind, stream="h")
        out.append(np.stack([p.core_means[k] for k in sorted(p.core_means)]))
    return out


def read_twin1(tree_dir: pathlib.Path, ref: str, c: str):
    with np.load(tree_dir / ref / "twin1" / f"{c}.npz") as z:
        return np.asarray(z["A_cores"], np.float64), np.asarray(z["B_cores"], np.float64)


def fetch(paths, token, dest: pathlib.Path, workers: int = 8, attempts: int = 4,
          log=print) -> int:
    """Fetch named files by **exact path**, pooled. Returns the bytes written.

    **Why not `snapshot_download`.** `snapshot_download(allow_patterns=...)` does not ask the Hub
    for the files it wants: it walks the **whole repository tree** in 1,000-file pages and filters
    client-side. On this dataset that is **146,266 files, 147 pages**, measured at **60.8 s**
    uncontended -- per reference, per session. With three sessions it produced a stream of
    HTTP 504 and 429 on the tree endpoint, visible in the 13 Oct logs, where the decoded
    pagination cursors sat in `pilot/record/...` and `profiles/.../twin1/...` -- prefixes this
    pass never asks for.

    `hf_hub_download` resolves one path with no listing at all. This is the replacement named as
    owed during S15 ("pooled `hf_hub_download` to stop the whole-repo listing and its 504s,
    deferred until no run depends on that path"), and it came due twice before being written.

    Each file is retried, and a file that never arrives raises -- a silently missing accumulator
    would make one construct's profile wrong rather than absent.
    """
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download

    def one(rel):
        for i in range(attempts):
            try:
                return hf_hub_download(REPO, rel, repo_type="dataset", token=token,
                                       local_dir=str(dest))
            except Exception:                                # noqa: BLE001
                if i == attempts - 1:
                    raise
                time.sleep(1.5 * (i + 1))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(one, paths))
    return sum(os.path.getsize(g) for g in got if g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(D / "validation/bootstraps_twin2"))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--only", default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--scratch", default="/content/_twin2")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    from experiments.run_s15 import references
    from profilebattery.profiles.yardstick import Yardstick
    from profilebattery.profiles.names import tree_name
    from profilebattery.record.remote import quiet_hub
    # **Not cosmetic.** The Hub draws a progress bar per file and this pass fetches 508 a
    # reference. Colab truncates a cell to its last 5,000 lines, so the bars push out the
    # per-construct counters and the upload confirmations -- the lines this run is monitored by.
    # `quiet_hub` exists for exactly this and its docstring records a fetch that twice looked
    # stalled on 4 Oct for the same reason. Omitting the call was my defect, not the Hub's.
    quiet_hub()
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    p1 = sorted({tree_name(c) for c in split["p1"]})
    refs = a.only.split(",") if a.only else references(token)
    if a.reverse:
        refs = list(reversed(refs))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(a.scratch)
    print(f"{len(p1)} P1 constructs, {len(refs)} references, {a.draws} draws", flush=True)

    def upstream(ref):
        try:
            return any(f.path == f"{PREFIX}/{ref}_twin2.json" for f in api.list_repo_tree(
                REPO, path_in_repo=PREFIX, repo_type="dataset", recursive=True))
        except Exception:                                    # noqa: BLE001
            return False

    for ref in refs:
        p = out / f"{ref}_twin2.json"
        if p.exists():
            # **Local but not upstream means a stranded upload, not a finished reference.**
            # The per-reference upload can fail -- it prints `upload deferred` -- and there is
            # no final sweep, so before this fix the file sat in /content forever and
            # `already done` skipped it on every restart. That is how `anneal_iter_0018000`
            # went missing while the session that computed it moved on, and it is the same
            # shape as the three local-only references the first bootstrap pass stranded.
            if a.upload and not upstream(ref):
                try:
                    api.upload_file(path_or_fileobj=str(p),
                                    path_in_repo=f"{PREFIX}/{ref}_twin2.json",
                                    repo_id=REPO, repo_type="dataset",
                                    commit_message=f"R2 twin-2 ceiling and agreement: {ref}")
                    print(f"  {ref}: computed earlier but not upstream -- uploaded now",
                          flush=True)
                except Exception as e:                       # noqa: BLE001
                    print(f"  {ref}: computed earlier, upload failed again "
                          f"({type(e).__name__})", flush=True)
            else:
                print(f"  {ref}: already done", flush=True)
            continue
        if a.upload and upstream(ref):
            print(f"  {ref}: already uploaded by another session", flush=True)
            continue
        shutil.rmtree(scratch, ignore_errors=True)
        t0 = time.time()
        pats = ([f"record/{ref}/twin2__{c}_{k}_acc.npz" for c in p1 for k in ("A", "B")]
                + [f"record/{ref}/twin2__C_cells.npz"]
                + [f"profiles/{ref}/twin1/{c}.npz" for c in p1])
        nb = fetch(pats, token, scratch)
        rec, tree = scratch / "record" / ref, scratch / "profiles"
        print(f"  {ref}: prefetched {len(pats)} files, {nb / 2**30:.1f} GiB in "
              f"{time.time() - t0:.0f}s (by path, no repo listing)", flush=True)
        ys = Yardstick.from_npz(str(rec / "twin2__C_cells.npz"))

        res, t1 = {}, time.time()
        for i, c in enumerate(p1, 1):
            try:
                t2a, t2b = read_twin2(rec, ref, c, ys)
                a1, b1 = read_twin1(tree, ref, c)
                res[c] = one_construct(a1, b1, t2a, t2b, a.draws)
            except Exception as e:                           # noqa: BLE001
                res[c] = {"error": f"{type(e).__name__}: {e}"}
            if i % 25 == 0:
                print(f"    {ref}: {i}/{len(p1)} constructs, {time.time() - t1:.0f}s",
                      flush=True)
        bad = sum("error" in v for v in res.values())
        print(f"  {ref}: {len(res)} constructs in {time.time() - t1:.0f}s ({bad} failed)",
              flush=True)
        p.write_text(json.dumps(res, indent=1, default=float))
        if a.upload:
            try:
                api.upload_file(path_or_fileobj=str(p),
                                path_in_repo=f"{PREFIX}/{ref}_twin2.json",
                                repo_id=REPO, repo_type="dataset",
                                commit_message=f"R2 twin-2 ceiling and agreement: {ref}")
                print(f"    uploaded {PREFIX}/{ref}_twin2.json", flush=True)
            except Exception as e:                           # noqa: BLE001
                print(f"    upload deferred ({type(e).__name__})", flush=True)
        shutil.rmtree(scratch, ignore_errors=True)
        # **No cache to purge, by design.** `fetch` passes `local_dir`, so files are written as
        # real copies into the scratch directory and the shared blob cache is never populated --
        # the `rmtree` above is the whole of the cleanup. The purge is attempted only if a cache
        # directory exists at all, because otherwise it reports `rc=1 Cache directory not found`
        # three times and a freed count of 0, which reads as a failure and is not one. A side
        # effect worth having: with no cache, nothing can be left orphaned in it either, which
        # is the failure mode that halted an S15 run at reference 1.
        cache = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface"))
        hub = cache / "hub" if cache.name != "hub" else cache
        if hub.exists():
            try:
                from profilebattery.record.remote import purge_repo_cache
                freed, how = purge_repo_cache(f"dataset/{dataset_repo()}")
                print(f"    purged the cache: {freed} ({how})", flush=True)
            except Exception as e:                           # noqa: BLE001
                print(f"    cache purge skipped: {type(e).__name__}", flush=True)
        print(f"    free {shutil.disk_usage('/').free / 2**30:.0f} GiB "
              f"(no blob cache: files are fetched to a scratch directory and deleted)",
              flush=True)

    # **A closing sweep, because a deferred upload had no second chance.** Any local result that
    # is not upstream is uploaded here, so a transient failure costs a retry and not a whole
    # reference. Cheap: one scoped listing and an upload only for what is genuinely missing.
    if a.upload:
        stranded = [q for q in sorted(out.glob("*_twin2.json"))
                    if not upstream(q.name.replace("_twin2.json", ""))]
        if stranded:
            print(f"\nclosing sweep: {len(stranded)} local result(s) are not upstream",
                  flush=True)
            for q in stranded:
                ref = q.name.replace("_twin2.json", "")
                try:
                    api.upload_file(path_or_fileobj=str(q),
                                    path_in_repo=f"{PREFIX}/{ref}_twin2.json",
                                    repo_id=REPO, repo_type="dataset",
                                    commit_message=f"R2 twin-2 ceiling and agreement: {ref}")
                    print(f"  uploaded {ref}", flush=True)
                except Exception as e:                       # noqa: BLE001
                    print(f"  {ref}: upload failed ({type(e).__name__}) -- it is still only "
                          f"in {out}", flush=True)
        else:
            print("\nclosing sweep: every local result is upstream", flush=True)


if __name__ == "__main__":
    main()
