r"""A timed trial of **one whole reference** for the merged B9 + B10 Colab session.

The programme's rule after A46 and A48: cost a pass from a timed trial of one whole unit of
work, never from a single read. A46 projected 0.49 h for the sharing bootstrap from a spot
benchmark and the pass took 3.19 h, because sustained throughput was 4.2 MB/s against a 15.9
MB/s single read. The unit of work here is one reference -- 687 per-prompt files, 77.86 GB --
and all four references are the same shape, so the session's cost is four times what this
measures, plus B10's C re-recording, which is a model run and is timed separately by the design
(~284 s per C set, two sets per reference).

What is timed, per construct, in the order the session would do it:

  read    the four per-prompt files (twin1/twin2 x A/B), ~1.16 GB, then deleted
  B9      two probes from the same arrays -- the 2,240-dim residual stream (primary for S7) and
          the 6,144 hidden units -- at the verdict layer and at the layer of largest norm,
          logistic regression A against B with both twins pooled, held out by core
          (leave-one-core-out over the eight cores), plus a random-label control
  B10     alternative (iv): every prompt's within-line band (tokens since the last newline:
          1 / 2-3 / 4-8 / 9+) times its token class, from the battery's logged anchor and the
          prompt text, and the per-cell accumulation over all 24 layers that re-standardising
          under the new yardstick needs

    python experiments/b9b10_trial.py --reference pretrain_iter_0001000 [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib
import time

import numpy as np

from experiments.atlas import D, LAST, OUT, REPO, provenance, token

MODEL = "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints"
MODEL_REV = "pretrain_iter_0008000"
#: the within-line bands of alternative (iv), in tokens since the last newline
BANDS = ((1, 1), (2, 3), (4, 8), (9, 10**9))
TRIAL_REF = "pretrain_iter_0001000"


def band_of(k: int) -> str:
    for lo, hi in BANDS:
        if lo <= k <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    raise ValueError(k)


def token_class(piece: str) -> str:
    """The four classes of the stored yardstick, by the piece at the read site."""
    p = piece.lstrip("▁")
    if not p:
        return "space"
    if p[0].isalpha() or p[0] == "_":
        return "keyword" if p.isidentifier() else "word"
    if p[0].isdigit():
        return "number"
    return "symbol"


_FS = None


def _stream(ref: str, name: str):
    """Read one per-prompt file over the network, into memory, bypassing the HF cache.

    The cache is what makes a trial lie. The first attempt used `hf_hub_download` and timed
    1.16 GB at 1,745 MB/s, because those files were already on this machine from earlier work;
    the session will read every one of them over the network. Streaming also keeps the trial
    off a disk with 17 GiB free.
    """
    import io
    global _FS
    if _FS is None:
        from huggingface_hub import HfFileSystem
        _FS = HfFileSystem(token=token())
    with _FS.open(f"datasets/{REPO}/record/{ref}/{name}", "rb") as f:
        raw = f.read()
    return len(raw), io.BytesIO(raw)


def _battery(twin: str, c: str, kind: str):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, f"phase_battery/{twin}/{c}_{kind}.jsonl.gz", repo_type="dataset",
                        token=token())
    return [json.loads(l) for l in gzip.open(p, "rt")]


def _drop(p: pathlib.Path) -> None:
    """Delete the blob and its symlink -- the trial must not fill a 17 GiB disk."""
    try:
        real = p.resolve()
        p.unlink(missing_ok=True)
        real.unlink(missing_ok=True)
    except OSError:
        pass


def probe(X: np.ndarray, y: np.ndarray, cores: np.ndarray, rng) -> dict:
    """A against B, held out by core, with a size-matched random-label control."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    def std(tr, te):
        """Centre and scale on the training fold only -- untransformed fp16 activations do not
        converge in any sane iteration budget, and scaling on the whole set would leak."""
        mu = X[tr].mean(0)
        sd = X[tr].std(0)
        sd[sd == 0] = 1.0
        return (X[tr] - mu) / sd, (X[te] - mu) / sd

    acc, auc, iters = [], [], []
    for c in np.unique(cores):
        tr, te = cores != c, cores == c
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        Xtr, Xte = std(tr, te)
        m = LogisticRegression(max_iter=1000, n_jobs=1).fit(Xtr, y[tr])
        iters.append(int(np.max(m.n_iter_)))
        p = m.predict_proba(Xte)[:, 1]
        acc.append(float(((p >= 0.5).astype(int) == y[te]).mean()))
        auc.append(float(roc_auc_score(y[te], p)) if len(np.unique(y[te])) > 1 else float("nan"))
    yc = rng.permutation(y)
    c0 = np.unique(cores)[0]
    tr, te = cores != c0, cores == c0
    Xtr, Xte = std(tr, te)
    mc = LogisticRegression(max_iter=1000, n_jobs=1).fit(Xtr, yc[tr])
    ctrl = float(((mc.predict_proba(Xte)[:, 1] >= 0.5).astype(int) == yc[te]).mean())
    return {"n": int(X.shape[0]), "d": int(X.shape[1]), "folds": len(acc),
            "max_lbfgs_iterations": max(iters) if iters else None,
            "hit_the_iteration_cap": bool(iters and max(iters) >= 1000),
            "accuracy": float(np.mean(acc)) if acc else None,
            "auc": float(np.nanmean(auc)) if auc else None,
            "control_accuracy": ctrl}


def one_construct(ref: str, c: str, tk, verdict: int, rng) -> dict | None:
    t0 = time.time()
    bufs, arrs, rows, nbytes = {}, {}, {}, 0
    for twin in ("twin1", "twin2"):
        for kind in ("A", "B"):
            try:
                n, buf = _stream(ref, f"{twin}__{c}_{kind}.npz")
            except Exception as exc:                                   # noqa: BLE001
                return {"construct": c, "skipped": f"{type(exc).__name__}: {exc}"}
            bufs[(twin, kind)] = buf
            nbytes += n
    t_read = time.time() - t0
    t0 = time.time()
    for k, buf in bufs.items():
        with np.load(buf) as z:
            arrs[k] = (np.asarray(z["primary"]), np.asarray(z["primary_resid"]))
    bufs.clear()
    for twin in ("twin1", "twin2"):
        for kind in ("A", "B"):
            rows[(twin, kind)] = _battery(twin, c, kind)
    t_decode = time.time() - t0

    # --- B9 -----------------------------------------------------------------
    # Two populations, not one. The 31 constructs of PER_PROMPT_SUBSET.json carry BOTH arrays;
    # the other 133 carry `primary_resid` (720, 24, 2240) and an EMPTY `primary` of shape
    # (0, 24, 1). TESTS-CC section 3 recorded both arrays for all 164 and that was wrong.
    # So the hidden-unit probe -- decision (2)'s object, and B10 (iv)'s exact re-binning, which
    # is in the profile's own space -- can only run on the 31; the residual probe runs on 164.
    has_hidden = all(arrs[k][0].shape[0] > 0 for k in arrs)
    if any(arrs[k][1].shape[0] == 0 for k in arrs):
        return None
    t0 = time.time()
    cores = np.concatenate([[r["core_id"] for r in rows[(t, k)]]
                            for t in ("twin1", "twin2") for k in ("A", "B")])
    y = np.concatenate([np.full(len(rows[(t, k)]), 1 if k == "A" else 0)
                        for t in ("twin1", "twin2") for k in ("A", "B")])
    core_ix = np.unique(cores, return_inverse=True)[1] % 8
    ref_arr = arrs[("twin1", "A")][0 if has_hidden else 1]
    norms = np.linalg.norm(ref_arr.mean(0), axis=1)
    peak = int(np.argmax(norms))
    b9 = {}
    spaces = [("resid_2240", 1)] + ([("hidden_6144", 0)] if has_hidden else [])
    for space, j in spaces:
        for lab, L in (("verdict", verdict), ("peak_norm", peak)):
            X = np.concatenate([arrs[(t, k)][j][:, L, :].astype(np.float32)
                                for t in ("twin1", "twin2") for k in ("A", "B")])
            b9[f"{space}|{lab}|L{L}"] = probe(X, y, core_ix, rng)
    t_b9 = time.time() - t0

    # --- B10 alternative (iv) ----------------------------------------------
    t0 = time.time()
    cells, clean = [], 0
    for twin in ("twin1", "twin2"):
        for kind in ("A", "B"):
            for r in rows[(twin, kind)]:
                site = r.get("site_construct_token")
                if site is None:
                    site = r.get("site_last_token")
                ids = tk(r["text"])["input_ids"]
                pieces = tk.convert_ids_to_tokens(ids)
                k = 1
                for i in range(min(site, len(pieces) - 1), -1, -1):
                    if "\n" in (pieces[i] or ""):
                        k = min(site, len(pieces) - 1) - i
                        break
                else:
                    k = min(site, len(pieces) - 1) + 1
                piece = pieces[min(site, len(pieces) - 1)]
                cells.append(f"{band_of(max(k, 1))}:{token_class(piece)}")
                clean += 1
    cells = np.asarray(cells)
    acc_by_cell = {}
    for cell in np.unique(cells):
        m = cells == cell
        off = 0
        chunks = []
        for t in ("twin1", "twin2"):
            for kind in ("A", "B"):
                n = arrs[(t, kind)][0].shape[0]
                sel = m[off:off + n]
                if sel.any():
                    chunks.append(arrs[(t, kind)][0 if has_hidden else 1][sel]
                                  .astype(np.float32).sum(0))
                off += n
        acc_by_cell[cell] = (int(m.sum()), np.sum(chunks, axis=0) if chunks else None)
    t_b10 = time.time() - t0
    return {"construct": c, "bytes": nbytes, "n_prompts": int(len(y)),
            "has_hidden_units": bool(has_hidden),
            "seconds": {"read": t_read, "decode": t_decode, "b9": t_b9, "b10": t_b10,
                        "total": t_read + t_decode + t_b9 + t_b10},
            "new_cells": len(acc_by_cell), "prompts_binned": clean,
            "verdict_layer": verdict, "peak_norm_layer": peak, "b9": b9}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default=TRIAL_REF)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "B9B10_TRIAL.json"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from profilebattery.profiles.layers import home_layers, verdict_layer
    tk = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV, trust_remote_code=True)
    homes = home_layers()
    rng = np.random.default_rng(20261018)
    p1 = sorted(json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"])
    from profilebattery.profiles.names import tree_name
    names = sorted({tree_name(c) for c in p1})
    if args.limit:
        names = names[:args.limit]
    done, skipped, t_start = [], [], time.time()
    for i, c in enumerate(names, 1):
        try:
            r = one_construct(args.reference, c, tk, verdict_layer(c, homes), rng)
        except Exception as exc:                                       # noqa: BLE001
            print(f"   {i:3d}/{len(names)} {c:24s} FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        if r is None:
            print(f"   {i:3d}/{len(names)} {c:24s} empty per-prompt arrays", flush=True)
            continue
        if "skipped" in r:
            print(f"   {i:3d}/{len(names)} {c:24s} {r['skipped'][:80]}", flush=True)
            skipped.append(r)
            continue
        done.append(r)
        s = r["seconds"]
        el = time.time() - t_start
        per = el / len(done)
        print(f"   {i:3d}/{len(names)} {c:24s} {'both' if r['has_hidden_units'] else 'res '} "
              f"{r['bytes']/1e9:.2f} GB  "
              f"read {s['read']:6.1f}s  decode {s['decode']:5.1f}s  B9 {s['b9']:6.1f}s  "
              f"B10 {s['b10']:6.1f}s  | mean/construct {per:6.1f}s  "
              f"-> one reference {per*len(names)/3600:5.2f} h", flush=True)
    if not done:
        print("nothing measured")
        return
    tot = {k: sum(r["seconds"][k] for r in done) for k in done[0]["seconds"]}
    gb = sum(r["bytes"] for r in done) / 1e9
    n = len(done)
    both = sum(1 for r in done if r["has_hidden_units"])
    out = {"reference": args.reference, "constructs_measured": n,
           "with_hidden_units": both, "residual_only": n - both,
           "two_populations": "the 31 of PER_PROMPT_SUBSET.json carry `primary` (n, 24, 6144) "
                              "and `primary_resid` (n, 24, 2240); the other 133 carry an empty "
                              "`primary` of shape (0, 24, 1) and the full residual array",
           "constructs_in_p1": len(names), "gigabytes_read": gb,
           "read_throughput_MB_s": gb * 1000 / tot["read"] if tot["read"] else None,
           "seconds_measured": tot,
           "per_construct_seconds": {k: v / n for k, v in tot.items()},
           "one_reference_hours": tot["total"] / n * len(names) / 3600,
           "four_references_hours": tot["total"] / n * len(names) * 4 / 3600,
           "not_included": "B10's C re-recording, which is a model run: the design states ~284 s "
                           "per C set and two sets per reference, so 0.63 h over the four",
           "machine": "this Mac, not the Colab runtime; the read term scales with throughput "
                      "and the measured value here is quoted beside the 15.9 MB/s single-file "
                      "and 4.2 MB/s sustained figures from the A46 pass",
           "rows": done, "skipped": skipped, "provenance": provenance([f"record/{args.reference}/*.npz",
                                                   "phase_battery/*"], reference=args.reference)}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\n{n} constructs, {gb:.1f} GB, read {tot['read']/60:.1f} min at "
          f"{out['read_throughput_MB_s']:.1f} MB/s, B9 {tot['b9']/60:.1f} min, "
          f"B10 {tot['b10']/60:.1f} min")
    print(f"one reference {out['one_reference_hours']:.2f} h; four {out['four_references_hours']:.2f} h")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
