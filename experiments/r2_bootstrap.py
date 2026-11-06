"""R2's Colab pass: curves 2 and 3, every core-bootstrap interval, and the verdict-layer tie-break.

**Why this cannot run locally and cannot be split.** The per-core arrays `A_cores` (8 x 24 x 6,144)
and `B_cores` (21 x 24 x 6,144) are about 17 MB per construct per twin per reference -- roughly
**390 GB** across 61 references, where the carrier index was 14 GB of ranged reads. And C0.5
requires **one core-index set per draw, applied to both twins and both references**, so the draw
cannot be made on one host for one reference and another host for the next: the resampling has to
see every reference at once. What comes back is interval summaries only, a few hundred megabytes,
never the per-core arrays.

**What it computes, per construct.**

**One limit found on implementing it, and it is structural.** The derived tree carries per-core
arrays for **twin 1 only** -- a twin-2 npz holds `concept`, `concept_se`, `concept_indented` and
nothing else, which is the array set as approved. So curves 2, 3 and 6 and the verdict-layer
tie-break bootstrap fully here, and the **twin ceiling** and **twin agreement** -- hence the
intervals for **curves 1 and 5** -- do not: they need twin 2's per-core arrays, which exist only
in the record's accumulators, about 1.7 TB of reads. See `per_draw`.

* **Curve 2, norm** -- the concept profile's norm in the product space as a share of its value at
  the last checkpoint of the branch. Raw form primary (unstandardised means), standardised beside
  it, because the pilot's Fact 3 was a growth that existed in the standardised form only.
* **Curve 3, depth** -- the centre of mass of the per-layer concept norms over **layers 1 to 22**.
  The plan's range, which already excludes layer 0 and layer 23; the last layer feeds the
  unembedding, and the Settle chat's R1 ruling asks for every verdict-layer figure both as the
  rule stands and with the last layer excluded, which this range already satisfies for curve 3.
* **The core bootstrap** -- `DRAWS` draws; each draw fixes one core-index set and applies it to
  twin 1 and twin 2 at **every** reference before any curve value is computed, so a draw is a
  resampling of cores and not of checkpoints. The returned interval is the 2.5/97.5 quantile of
  the draw distribution, per curve per checkpoint, plus the SE of the difference that A30's
  dating rule divides by.
* **The verdict-layer tie-break** -- the share of draws in which each layer is the argmax of the
  per-layer norm at `pretrain_iter_0484865`. A layer winning at least 5% of draws is reported
  beside the modal layer, which is `PLAN-Analysis` part 3's clause and the R1 ruling's condition
  for quoting any verdict-layer figure.

**Written per reference, resumable.** A reference whose output exists is skipped, as `run_s15`
does, because a Colab session can be reclaimed mid-pass.
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
DEPTH_LAYERS = slice(1, 23)         # the plan's layers 1 to 22
VERDICT_REF = "pretrain_iter_0484865"
K = 0.05


def core_contrast(a_cores: np.ndarray, b_cores: np.ndarray, idx_a, idx_b) -> np.ndarray:
    """One draw's contrast profile: the resampled A cores' mean minus the resampled B cores'.

    A and B have different core counts (8 and 21 on the phase battery), so a draw resamples each
    side over its own cores with the **same** generator state -- the index sets are drawn once per
    draw in `draw_indices` and handed in here, so both twins and every reference see the same
    draw, which is what C0.5 requires.
    """
    return a_cores[idx_a].mean(axis=0) - b_cores[idx_b].mean(axis=0)


def draw_indices(n_a: int, n_b: int, draws: int, seed: int = SEED):
    """`[(idx_a, idx_b)]` -- one core-index set per draw, reused across twins and references."""
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, n_a, n_a), rng.integers(0, n_b, n_b)) for _ in range(draws)]


def layer_norms(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(v, dtype=np.float64), axis=1)


def centre_of_mass(norms: np.ndarray, layers: slice = DEPTH_LAYERS) -> float:
    """Curve 3: the centre of mass of the per-layer norms over the plan's layer range."""
    n = np.asarray(norms, float)[layers]
    w = n.sum()
    if not np.isfinite(w) or w <= 0:
        return float("nan")
    return float((np.arange(layers.start, layers.start + n.size) * n).sum() / w)


def sparsity(v: np.ndarray, k: float = K) -> float:
    f = np.abs(np.asarray(v, dtype=np.float64)).ravel() ** 2
    tot = f.sum()
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    kk = max(1, int(round(k * f.size)))
    return float(np.sort(f)[::-1][:kk].sum() / tot)


def carriers(v: np.ndarray, k: float = K) -> np.ndarray:
    f = np.abs(np.asarray(v, dtype=np.float64)).ravel() ** 2
    if not np.isfinite(f.sum()) or f.sum() <= 0:
        return np.zeros(0, dtype=np.uint32)
    kk = max(1, int(round(k * f.size)))
    return np.sort(np.argsort(-f, kind="stable")[:kk]).astype(np.uint32)


def per_draw(a, b, idx_pairs) -> dict:
    """Every per-draw quantity for one (reference, construct) on **twin 1**.

    **Why twin 1 only, and what that costs.** The derived tree carries the per-core arrays for
    **twin 1 alone**: a twin-2 npz holds `concept`, `concept_se` and `concept_indented` and
    nothing else, which is the array set as approved -- concept-only, to keep the tree at 427 GB.
    So a per-core draw exists for twin 1 and does not exist for twin 2 anywhere in the tree.

    That splits R2's intervals in two:

    * **Single-twin quantities are fully computable here** -- curve 2's norm, curve 3's depth and
      curve 6's sparsity are read on twin 1, the primary battery, so a core bootstrap over twin
      1's cores is the interval the contract asks for. Same for the verdict-layer argmax.
    * **Quantities that use both twins are not** -- the **twin ceiling** (twin 1's carriers
      against twin 2's) and the **twin agreement**, and therefore the intervals for **curve 1**
      and **curve 5**, need twin 2's per-core arrays. Those exist only in the **record's
      accumulators**, not in the derived tree, so they need a separate pass over
      `record/<ref>/twin2__<c>_{A,B}_acc.npz` -- about 1.7 TB of reads against this pass's 390 GB.
      That is a decision, not something to approximate: substituting twin 1's own cores for twin
      2's would make the ceiling a within-battery quantity and it is a between-battery one.

    C0.5's "one core-index set per draw applied to both twins and both references" is therefore
    met for the twin it can be met for, and the index sets are still drawn once per draw and
    reused across every reference, so the twin-2 pass can reuse the identical draws when it runs.
    """
    return _per_draw_vectorised(a, b, idx_pairs)


def _weights(idx_pairs, n, which: int) -> np.ndarray:
    """`(draws, n)` -- each row the resampling weights of one draw: core counts over `n`.

    A draw's resampled mean is a **weighted sum of the cores**, the weight of core `k` being how
    many times `k` was drawn divided by the draw size. Writing it that way is what makes the
    bootstrap a matrix product instead of a loop.
    """
    w = np.zeros((len(idx_pairs), n), dtype=np.float64)
    for d, pair in enumerate(idx_pairs):
        idx = np.asarray(pair[which]) % n
        np.add.at(w[d], idx, 1.0)
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1.0)


def _per_draw_vectorised(a, b, idx_pairs, chunk: int = 100) -> dict:
    """The same quantities as the loop, computed as a matrix product over draws.

    **Why this is not a micro-optimisation.** Measured on real arrays, the per-draw loop took
    **15 s for 2 constructs x 50 draws** including the reads -- about 30 ms a draw -- which over
    169 constructs x 1,000 draws x 61 references is **86 hours**. Handing that to a Colab session
    would have been handing over a pass that cannot finish. As a matrix product the draws cost one
    `(draws x n_cores) @ (n_cores x 147,456)` multiply per chunk, and `np.partition` replaces the
    full `argsort` that `sparsity` was doing per draw.

    Chunked so the intermediate stays bounded: 100 draws x 147,456 float64 is 118 MB.
    """
    n_a, n_b = a.shape[0], b.shape[0]
    af = np.asarray(a, np.float64).reshape(n_a, -1)
    bf = np.asarray(b, np.float64).reshape(n_b, -1)
    nlay, nunit = a.shape[1], a.shape[2]
    kk = max(1, int(round(K * af.shape[1])))
    wa, wb = _weights(idx_pairs, n_a, 0), _weights(idx_pairs, n_b, 1)
    norms, depth, sp, lay = [], [], [], []
    for s0 in range(0, len(idx_pairs), chunk):
        s1 = min(s0 + chunk, len(idx_pairs))
        v = wa[s0:s1] @ af - wb[s0:s1] @ bf                  # (chunk, layers*units)
        norms.append(np.linalg.norm(v, axis=1))
        ln = np.linalg.norm(v.reshape(-1, nlay, nunit), axis=2)   # (chunk, layers)
        lay.append(np.argmax(ln, axis=1))
        seg = ln[:, DEPTH_LAYERS]
        w = seg.sum(axis=1)
        idx = np.arange(DEPTH_LAYERS.start, DEPTH_LAYERS.start + seg.shape[1], dtype=float)
        depth.append(np.where(w > 0, (seg * idx).sum(axis=1) / np.maximum(w, 1e-300), np.nan))
        f = v * v
        tot = f.sum(axis=1)
        top = np.partition(f, f.shape[1] - kk, axis=1)[:, f.shape[1] - kk:].sum(axis=1)
        sp.append(np.where(tot > 0, top / np.maximum(tot, 1e-300), np.nan))
    return {"norm_twin1": np.concatenate(norms), "depth_twin1": np.concatenate(depth),
            "sparsity_twin1": np.concatenate(sp),
            "argmax_layer_twin1": np.concatenate(lay)}


def interval(x: np.ndarray) -> dict:
    """The 2.5/97.5 quantiles, the mean and the SE of a draw distribution."""
    x = np.asarray(x, float)
    good = x[np.isfinite(x)]
    if good.size < 2:
        return {"mean": None, "se": None, "lo": None, "hi": None, "n_finite": int(good.size)}
    return {"mean": float(good.mean()), "se": float(good.std(ddof=1)),
            "lo": float(np.quantile(good, 0.025)), "hi": float(np.quantile(good, 0.975)),
            "n_finite": int(good.size)}


def tie_break(argmax_draws: np.ndarray, floor: float = 0.05) -> dict:
    """Part 3's clause: every layer winning at least `floor` of the draws, with its share."""
    a = np.asarray(argmax_draws, int)
    vals, counts = np.unique(a, return_counts=True)
    share = {int(v): float(n / a.size) for v, n in zip(vals, counts)}
    contenders = {k: v for k, v in share.items() if v >= floor}
    modal = max(share, key=share.get)
    return {"modal_layer": modal, "modal_share": share[modal], "shares": share,
            "contenders": sorted(contenders), "tied": len(contenders) > 1}


def one_reference(ref: str, constructs, read, draws: int = DRAWS, log=print) -> dict:
    """Every per-draw summary for one reference. `read(ref, twin, construct) -> npz mapping."""
    t0 = time.time()
    out = {}
    for i, c in enumerate(constructs, 1):
        if i % 25 == 0:
            print(f"    {ref}: {i}/{len(constructs)} constructs, "
                  f"{time.time() - t0:.0f}s", flush=True)
        try:
            z = read(ref, "twin1", c)
            a = np.asarray(z["A_cores"], np.float64)
            b = np.asarray(z["B_cores"], np.float64)
        except Exception as e:                                   # noqa: BLE001
            out[c] = {"error": f"{type(e).__name__}: {e}"}
            continue
        pairs = draw_indices(a.shape[0], b.shape[0], draws)
        d = per_draw(a, b, pairs)
        e = {k: interval(v) for k, v in d.items() if not k.startswith("argmax")}
        e["n_cores"] = {"A": int(a.shape[0]), "B": int(b.shape[0])}
        e["twin"] = "twin1 only; the tree carries no per-core arrays for twin 2"
        if ref == VERDICT_REF:
            e["tie_break_twin1"] = tie_break(d["argmax_layer_twin1"])
        out[c] = e
    log(f"  {ref}: {len(out)} constructs in {time.time() - t0:.0f}s "
        f"({sum('error' in v for v in out.values())} failed)")
    return out


def _reader_tree(root: pathlib.Path):
    def read(ref, twin, c):
        with np.load(root / ref / twin / f"{c}.npz") as z:
            return {k: z[k] for k in ("A_cores", "B_cores")}
    return read


def prefetch(ref: str, constructs, token, dest: pathlib.Path, log=print) -> pathlib.Path:
    """Download one reference's twin-1 files whole, and return the directory they are in.

    **Why whole files and not ranged reads.** This pass needs `A_cores` and `B_cores`, about
    17 MB of a 35-98 MB npz, so a ranged read looks like the cheaper route -- and it is, in bytes.
    It is not in time: a single ranged read into one of these files measured **~12 s**, almost all
    of it per-file latency, and the pass makes **169 x 61 = 10,309** of them. That is about
    **34 hours of latency** before any arithmetic, which is what the first Colab attempt was
    doing when it produced no output for 13 minutes.

    Whole files invert it: one reference's twin 1 is about **5.9 GiB** for 169 constructs, which
    at Colab's measured ~110 MB/s is **under a minute**, against 3.7 min of compute. The cost is
    disk, which is why the caller deletes the directory and purges the cache after each reference.
    """
    from huggingface_hub import snapshot_download
    t0 = time.time()
    pats = [f"profiles/{ref}/twin1/{c}.npz" for c in constructs]
    snapshot_download(REPO, repo_type="dataset", token=token, allow_patterns=pats,
                      local_dir=str(dest), max_workers=8)
    root = dest / "profiles"
    got = len(list((root / ref / "twin1").glob("*.npz"))) if (root / ref).exists() else 0
    nbytes = sum(f.stat().st_size for f in (root / ref).rglob("*.npz")) if got else 0
    log(f"  {ref}: prefetched {got} files, {nbytes / 2**30:.2f} GiB in "
        f"{time.time() - t0:.0f}s")
    return root


def _reader_ranged(token):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token)

    def read(ref, twin, c):
        with fs.open(f"datasets/{REPO}/profiles/{ref}/{twin}/{c}.npz", "rb") as f:
            with np.load(f) as z:
                return {k: z[k] for k in ("A_cores", "B_cores")}
    return read


def retrying(read, attempts: int = 4, log=print):
    def wrapped(ref, twin, c):
        for i in range(attempts):
            try:
                return read(ref, twin, c)
            except Exception:                                    # noqa: BLE001
                if i == attempts - 1:
                    raise
                time.sleep(1.5 * (i + 1))
    return wrapped


def _done_upstream(ref: str, token) -> bool:
    """Whether another session has already uploaded this reference.

    Checked immediately before a reference starts, not once at the beginning, so two sessions
    working from opposite ends do not both spend ten minutes on the one they meet at. Any error
    is treated as *not done*: redoing a reference costs ten minutes, skipping one that is not
    there costs the whole pass a hole.
    """
    try:
        from huggingface_hub import HfApi
        # **Scoped, not a whole-repo listing.** `list_repo_files` enumerates all 146,266 files
        # of this dataset and measured **60.8 s**; `list_repo_tree` on this one prefix measured
        # **0.2 s**. At one call per reference per session that is about an hour of wall time
        # across the pass for nothing -- and it is the same defect S15 already paid for, where
        # two whole-repo enumerations per reference cost 1,437 s on a six-construct trial. I
        # reintroduced it here and the fix is the call I had already written for it.
        return any(f.path == f"{PREFIX}/{ref}.json"
                   for f in HfApi(token=token).list_repo_tree(
                       REPO, path_in_repo=PREFIX, repo_type="dataset", recursive=True))
    except Exception:                                        # noqa: BLE001
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(D / PREFIX))
    ap.add_argument("--tree", default=None, help="a local profiles/ tree; omit to read ranged")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--only", default=None)
    ap.add_argument("--reverse", action="store_true",
                    help="work the reference list from the end, so a second session halves the "
                         "wall time: the pass already skips a reference whose output exists, so "
                         "two hosts from opposite ends meet in the middle and the split "
                         "balances itself. The same arrangement S15 used.")
    ap.add_argument("--ranged", action="store_true",
                    help="read A_cores/B_cores by ranged read instead of prefetching whole "
                         "files; measured ~34 h of pure latency over the full pass, so this is "
                         "for a one-reference check and not for the pass")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()

    from experiments.run_s15 import references
    from profilebattery.profiles.names import tree_name
    from profilebattery.record.remote import hub_token, quiet_hub
    quiet_hub()
    token = hub_token()

    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    constructs = sorted({tree_name(c) for c in split["p1"]})
    refs = a.only.split(",") if a.only else references(token)
    if a.reverse:
        refs = list(reversed(refs))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(os.environ.get("PB_SCRATCH", "/content/_r2_tree"))
    mode = "tree" if a.tree else ("ranged" if a.ranged else "prefetch")
    print(f"{len(constructs)} P1 constructs, {len(refs)} references, {a.draws} draws; "
          f"reads: {mode}", flush=True)
    if mode == "prefetch":
        print(f"  free disk {shutil.disk_usage(scratch.parent if scratch.parent.exists() else '/').free / 2**30:.0f} GiB "
              f"(a reference needs ~6 GiB, deleted after each)", flush=True)

    fixed = retrying(_reader_tree(pathlib.Path(a.tree))) if a.tree else None
    for ref in refs:
        p = out / f"{ref}.json"
        if p.exists():
            print(f"  {ref}: already done", flush=True)
            continue
        if a.upload and _done_upstream(ref, token):
            print(f"  {ref}: already uploaded by another session", flush=True)
            continue
        if mode == "prefetch":
            shutil.rmtree(scratch, ignore_errors=True)
            root = prefetch(ref, constructs, token, scratch)
            read = retrying(_reader_tree(root))
        else:
            read = fixed or retrying(_reader_ranged(token))
        res = one_reference(ref, constructs, read, a.draws)
        p.write_text(json.dumps(res, indent=1, default=float))
        if a.upload:
            # **per reference, not once at the end.** A folder upload after all 61 would mean a
            # second session working from the other end sees nothing until the first finishes,
            # so `_done_upstream` would never fire and both would redo the middle. Uploading as
            # each reference lands is what makes the split balance itself.
            try:
                from huggingface_hub import HfApi
                HfApi(token=token).upload_file(
                    path_or_fileobj=str(p), path_in_repo=f"{PREFIX}/{ref}.json",
                    repo_id=REPO, repo_type="dataset",
                    commit_message=f"R2 bootstrap: {ref}")
                print(f"    uploaded {PREFIX}/{ref}.json", flush=True)
            except Exception as e:                           # noqa: BLE001
                print(f"    upload deferred ({type(e).__name__}); the folder upload at the "
                      f"end will carry it", flush=True)
        if mode == "prefetch":
            shutil.rmtree(scratch, ignore_errors=True)
            try:
                from profilebattery.record.remote import purge_repo_cache
                freed, how = purge_repo_cache(f"dataset/{dataset_repo()}")
                print(f"    purged the cache: {freed} ({how}); "
                      f"free {shutil.disk_usage('/').free / 2**30:.0f} GiB", flush=True)
            except Exception as e:                               # noqa: BLE001
                print(f"    cache purge skipped: {type(e).__name__}", flush=True)

    if a.upload:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_folder(
            folder_path=str(out), path_in_repo=PREFIX, repo_id=REPO, repo_type="dataset",
            commit_message="R2's core-bootstrap intervals and the verdict-layer tie-break")
        print(f"uploaded to {PREFIX}/")


if __name__ == "__main__":
    main()
