r"""B10 — yardstick sensitivity (W4a): does the choice of cell split change the carrier sets?

Four alternatives against the stored 20-cell reference (5 relative-position bands x 4 token
classes):

  (i)   one cell
  (ii)  bands only, 5
  (iii) token classes only, 4
  (iv)  within-line bands (1 / 2-3 / 4-8 / 9+) x class, 16

**(i)-(iii) are exact coarsenings of the stored 20 cells** and need neither a model run nor
per-prompt data: the C pass stored (n, mean, m2, active) per (layer, cell), and those pool
exactly. Pooling them and handing the result back as an ordinary `Yardstick` means the smooth
sigma0 floor, the standardisation and the profile construction are the pipeline's own code, not a
reimplementation of it -- which is what makes the identity check below meaningful.

**(iv) is not a coarsening**: within-line position cuts across the relative-position bands, so C
has to be re-recorded by a model run. `recordc` does one checkpoint at a time.

**The identity check the design requires**: the 20-cell binning put back through this same code
path must reproduce the stored carrier sets at overlap 1.0. It runs as alternative `twenty`.

    python experiments/b10.py coarse                      # (i)-(iii) and the identity check
    python experiments/b10.py recordc --reference untrained   # C under (iv), one checkpoint
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import time

import numpy as np

from experiments.atlas import (CARRIERS, D, K, OUT, REPO, dataset_record, provenance, token)
from profilebattery.profiles.cells import CLASSES, N_BANDS, token_class
from profilebattery.profiles.yardstick import Cell, Yardstick

REFS = ("untrained", "pretrain_iter_0001000", "pretrain_iter_0020000", "pretrain_iter_0484865")
MODEL = "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints"
TOK_REV = "pretrain_iter_0008000"
#: alternative (iv)'s within-line bands, in tokens since the last newline
IV_BANDS = ((1, 1), (2, 3), (4, 8), (9, 10 ** 9))
WORK = D / "validation/paper1/b10"


def iv_band(k: int) -> str:
    for lo, hi in IV_BANDS:
        if lo <= k <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
    raise ValueError(k)


def coarsen(cell: str, how: str) -> str:
    """The stored cell `b{band}:{class}` under one alternative's grouping."""
    b, cls = cell.split(":", 1)
    if how == "one":
        return "all"
    if how == "bands":
        return b
    if how == "classes":
        return cls
    if how == "twenty":
        return cell
    raise ValueError(how)


def pooled_yardstick(ys: Yardstick, how: str) -> Yardstick:
    """A yardstick whose every stored cell carries its coarse group's pooled statistics.

    The returned object is keyed by the **fine** cell, so `from_accumulators` looks cells up
    exactly as it always does and every accumulator still finds its entry -- the coarsening shows
    up only in the numbers. Pooling is the Welford combination, which is exact: n sums, the mean
    is the n-weighted mean, and m2 gains each group member's squared displacement from it.
    """
    groups: dict[tuple[int, str], list] = {}
    for (layer, cell), c in ys.cells.items():
        groups.setdefault((layer, coarsen(cell, how)), []).append(c)
    pooled: dict[tuple[int, str], Cell] = {}
    for key, members in groups.items():
        n = sum(c.n for c in members)
        if n == 0:
            continue
        mean = sum(c.n * c.mean for c in members) / n
        m2 = sum(c.var * max(c.n - 1, 1) + c.n * (c.mean - mean) ** 2 for c in members)
        active = sum(c.n * c.active for c in members) / n
        pooled[key] = Cell(n=n, mean=mean, var=m2 / max(n - 1, 1), active=active)
    out = {}
    for (layer, cell) in ys.cells:
        key = (layer, coarsen(cell, how))
        if key in pooled:
            out[(layer, cell)] = pooled[key]
    return Yardstick(out, ys.eps)


_FS = None


def read_acc(ref: str, name: str):
    """One accumulator file, over the network into memory, never onto this disk."""
    import io
    global _FS
    if _FS is None:
        from huggingface_hub import HfFileSystem
        _FS = HfFileSystem(token=token())
    with _FS.open(f"datasets/{REPO}/record/{ref}/{name}", "rb") as f:
        return io.BytesIO(f.read())


def carriers(v: np.ndarray, k: int = K) -> np.ndarray:
    f = np.asarray(v, np.float64).ravel() ** 2
    return np.sort(np.argsort(-f, kind="stable")[:k])


def concept(ref: str, twin: str, c: str, ys: Yardstick, acc=None) -> np.ndarray:
    """The concept profile under one yardstick, from accumulators held in memory.

    The accumulators are **streamed, not cached**: this pass reads about 96 GB and the machine
    has 17 GiB free, so `hf_hub_download` -- which `dataset_record` uses, and which persists
    every file -- would fill the disk a third of the way through. `acc` lets one construct's two
    files serve all four alternatives, so each is fetched once rather than four times.
    """
    from profilebattery.profiles.profile import from_accumulators
    if acc is None:
        acc = {k: np.load(read_acc(ref, f"{twin}__{c}_{k}_acc.npz")) for k in ("A", "B")}
    a = from_accumulators(acc["A"], ys, c, "A", stream="h").mean
    b = from_accumulators(acc["B"], ys, c, "B", stream="h").mean
    return a - b


def coarse(args) -> None:
    """Alternatives (i)-(iii), and the 20-cell identity check, from the stored accumulators."""
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    hows = ["twenty", "one", "bands", "classes"]
    refs = [args.reference] if args.reference else list(REFS)
    names = sorted(p1_set())
    WORK.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    for ref in refs:
        (WORK / ref).mkdir(parents=True, exist_ok=True)
        idx = load_index(CARRIERS, [ref]).get(ref, {})
        ys = {t: Yardstick.from_npz(dataset_record(ref, f"{t}__C_cells.npz"))
              for t in ("twin1", "twin2")}
        pool = {(t, h): pooled_yardstick(ys[t], h) for t in ys for h in hows}
        for i, c in enumerate(names, 1):
            out = WORK / ref / f"{c}.json"
            if out.exists() and not args.force:
                continue
            try:
                row = {"construct": c, "reference": ref, "alternatives": {}}
                prof = {}
                for t in ("twin1", "twin2"):
                    acc = {k: np.load(read_acc(ref, f"{t}__{c}_{k}_acc.npz"))
                           for k in ("A", "B")}
                    for h in hows:
                        prof[(t, h)] = concept(ref, t, c, pool[(t, h)], acc)
                    acc.clear()
            except Exception as exc:                                   # noqa: BLE001
                print(f"   {ref} {c:22s} SKIP {type(exc).__name__}: {str(exc)[:60]}", flush=True)
                continue
            stored = {}
            for t in ("twin1", "twin2"):
                key = f"{t}/{c}|product"
                if key in idx:
                    stored[t] = np.sort(np.asarray(idx[key], np.int64))
            for h in hows:
                c1, c2 = carriers(prof[("twin1", h)]), carriers(prof[("twin2", h)])
                e = {"twin_ceiling": float(np.intersect1d(c1, c2).size / K)}
                for t, cs in (("twin1", c1), ("twin2", c2)):
                    if t in stored:
                        inter = int(np.intersect1d(cs, stored[t]).size)
                        e[f"overlap_with_stored_{t}"] = inter / K
                        e[f"jaccard_with_stored_{t}"] = inter / (2 * K - inter)
                row["alternatives"][h] = e
            ref_ceiling = row["alternatives"]["twenty"]["twin_ceiling"]
            for h in hows:
                ov = row["alternatives"][h].get("overlap_with_stored_twin1")
                row["alternatives"][h]["below_its_own_ceiling"] = (
                    bool(ov is not None and ov < ref_ceiling))
            out.write_text(json.dumps(row, indent=1, default=float))
            done += 1
            idc = row["alternatives"]["twenty"].get("overlap_with_stored_twin1")
            print(f"   {ref[-8:]:>8s} {i:3d}/{len(names)} {c:22s} identity "
                  f"{idc if idc is None else round(idc, 4)}  "
                  + "  ".join(f"{h} {row['alternatives'][h].get('overlap_with_stored_twin1', float('nan')):.3f}"
                              for h in ("one", "bands", "classes"))
                  + f"  | {(time.time()-t0)/3600:.2f} h", flush=True)
    print(f"\n{done} construct-references in {(time.time()-t0)/3600:.2f} h")
    collect(args)


def collect(args) -> None:
    rows = {}
    for ref in REFS:
        d = WORK / ref
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            r = json.loads(p.read_text())
            rows[f"{ref}|{r['construct']}"] = r
    if not rows:
        print("nothing collected")
        return
    summary = {}
    for how in ("twenty", "one", "bands", "classes"):
        per_ref = {}
        for ref in REFS:
            v = [r["alternatives"][how]["overlap_with_stored_twin1"]
                 for r in rows.values()
                 if r["reference"] == ref and "overlap_with_stored_twin1" in r["alternatives"][how]]
            if not v:
                continue
            below = [r for r in rows.values() if r["reference"] == ref
                     and r["alternatives"][how].get("below_its_own_ceiling")]
            per_ref[ref] = {"n": len(v), "median_overlap": float(np.median(v)),
                            "min": float(min(v)), "max": float(max(v)),
                            "share_below_ceiling": len(below) / len(v)}
        summary[how] = per_ref
    ident = [r["alternatives"]["twenty"]["overlap_with_stored_twin1"] for r in rows.values()
             if "overlap_with_stored_twin1" in r["alternatives"]["twenty"]]
    out = {"question": "does the choice of cell split change the carrier sets (W4a)",
           "reference_binning": f"{N_BANDS} relative-position bands x {len(CLASSES)} token "
                                f"classes = {N_BANDS * len(CLASSES)} cells",
           "alternatives": {"one": "(i) one cell", "bands": "(ii) bands only, 5",
                            "classes": "(iii) token classes only, 4",
                            "twenty": "the 20-cell reference, through this same code path — "
                                      "the design's identity check"},
           "identity_check": {"n": len(ident),
                              "min_overlap": float(min(ident)) if ident else None,
                              "median": float(np.median(ident)) if ident else None,
                              "passes_at_1.0": bool(ident and min(ident) >= 0.9999)},
           "summary": summary, "rows": rows,
           "provenance": provenance(["record/<ref>/twin{1,2}__<c>_{A,B}_acc.npz",
                                     "record/<ref>/twin{1,2}__C_cells.npz"])}
    (OUT / "YARDSTICK_coarse.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {OUT / 'YARDSTICK_coarse.json'}  ({len(rows)} construct-references)")
    print(f"   identity check: min {out['identity_check']['min_overlap']}, "
          f"passes {out['identity_check']['passes_at_1.0']}")
    for how in ("one", "bands", "classes"):
        for ref, e in summary[how].items():
            print(f"   {how:8s} {ref:24s} median overlap {e['median_overlap']:.4f}  "
                  f"min {e['min']:.4f}  n {e['n']}")


# ---------------------------------------------------------------- (iv): C re-recorded
def iv_cells_for(pieces: list[str], n: int) -> list[str]:
    """Alternative (iv)'s cell for every position of one prompt.

    Within-line position is tokens since the last newline piece, 1-indexed, so the first token
    of a line is band `1-1`. The class is the recording's own `token_class`, not a restatement
    of it -- the whole point of (iv) is to vary the *position* axis while the class axis stays
    identical to the 20-cell reference it is measured against.
    """
    out, since = [], 0
    for i in range(n):
        since += 1
        out.append(f"{iv_band(since)}:{token_class(pieces[i])}")
        if "\n" in (pieces[i] or "").replace("\u2581", " "):
            since = 0
    return out


def recordc(args) -> None:
    """Re-record set C at one checkpoint under alternative (iv), with the 20-cell check beside.

    One checkpoint at a time, by Piotr's instruction of 19 Oct: the model is 3.81 GB a
    checkpoint and this machine has about 16 GiB free, so each is fetched, used and deleted
    before the next. Both binnings come off the **same forward pass**, so the 20-cell set
    re-recorded here can be checked against the stored one -- if they disagree, the hook or the
    tokenisation is wrong and (iv)'s numbers mean nothing.
    """
    import torch
    from huggingface_hub import hf_hub_download, scan_cache_dir
    from transformers import AutoTokenizer
    from profilebattery.profiles.cells import band, cell_key
    from profilebattery.record.store import Welford
    ref = args.reference
    dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    # The stored C was recorded on cuda in **bfloat16** (record/<ref>/MANIFEST.json). The first
    # attempt here used float32 and the re-recorded 20-cell means did not match the stored ones
    # at all -- relative 1.07 -- while the position counts matched in all 480 cells, which said
    # the cells were right and the model was not. Through 24 layers of a randomly initialised
    # network the two precisions diverge by order one. Matching the dtype is what makes (iv)'s
    # yardstick comparable to the 20-cell reference it is measured against.
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    t_all = time.time()
    # the untrained reference is not a checkpoint but a null -- the architecture instantiated
    # from the config with a fixed seed, exactly as `sweep.load_reference` builds it
    from profilebattery.record.sweep import UNTRAINED_SEED
    tk = AutoTokenizer.from_pretrained(MODEL, revision=TOK_REV, trust_remote_code=True)
    t0 = time.time()
    if ref == "untrained":
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(MODEL, revision=TOK_REV, trust_remote_code=True)
        torch.manual_seed(args.seed or UNTRAINED_SEED)
        model = AutoModel.from_config(cfg).to(device=dev, dtype=dtype).eval()
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(MODEL, revision=ref, dtype=dtype,
                                          trust_remote_code=True).to(dev).eval()
    t_load = time.time() - t0
    layers = model.layers if hasattr(model, "layers") else model.model.layers
    n_layers = len(layers)
    buf: dict[int, "torch.Tensor"] = {}

    def pre_h(i):
        def fn(_m, a):
            buf[i] = a[0].detach().float().cpu()
            return None
        return fn

    handles = [l.mlp.down_proj.register_forward_pre_hook(pre_h(i))
               for i, l in enumerate(layers)]
    stores: dict[str, dict[str, Welford]] = {}
    t0, n_pos = time.time(), 0
    try:
        for twin in ("twin1", "twin2"):
            store = stores.setdefault(twin, {})
            path = hf_hub_download(REPO, f"phase_battery/{twin}/C.jsonl.gz",
                                   repo_type="dataset", token=token())
            rows = [json.loads(l) for l in gzip.open(path, "rt")]
            if args.limit:
                rows = rows[:args.limit]
            for s_ in range(0, len(rows), args.batch):
                chunk = rows[s_:s_ + args.batch]
                enc = tk([r["text"] for r in chunk], return_tensors="pt", padding=True,
                         add_special_tokens=False)
                buf.clear()
                with torch.no_grad():
                    model(enc["input_ids"].to(dev))
                mask = enc["attention_mask"].numpy()
                for bi, r in enumerate(chunk):
                    pieces = tk.convert_ids_to_tokens(
                        tk(r["text"], add_special_tokens=False)["input_ids"])
                    n = min(len(pieces), int(mask[bi].sum()))
                    if n == 0:
                        continue
                    iv = iv_cells_for(pieces, n)
                    by_iv, by_20 = {}, {}
                    for pos in range(n):
                        by_iv.setdefault(iv[pos], []).append(pos)
                        by_20.setdefault(cell_key(band(pos, n), token_class(pieces[pos])),
                                         []).append(pos)
                    n_pos += n
                    for li in range(n_layers):
                        vh = buf[li][bi].numpy()
                        for ck, ix in by_iv.items():
                            store.setdefault(f"C|L{li}|{ck}", Welford()).update(vh[ix])
                        for ck, ix in by_20.items():
                            store.setdefault(f"T20|L{li}|{ck}", Welford()).update(vh[ix])
            print(f"   {twin}: {len(rows)} prompts done, {time.time()-t0:.0f}s elapsed",
                  flush=True)
    finally:
        for h in handles:
            h.remove()
    t_record = time.time() - t0
    WORK.mkdir(parents=True, exist_ok=True)
    files = {}
    for twin, store in stores.items():
        out = WORK / f"C_iv_{ref}__{twin}.npz"
        flat = {}
        for key, w in store.items():
            for stat, v in w.to_dict().items():
                flat[f"{key}|{stat}"] = np.asarray([v] if np.isscalar(v) else v)
        np.savez_compressed(out, **flat)
        files[twin] = str(out)
    store = stores["twin1"]
    out = pathlib.Path(files["twin1"])
    cells_iv = sorted({k.split("|")[2] for k in store if k.startswith("C|")})
    cells_20 = sorted({k.split("|")[2] for k in store if k.startswith("T20|")})
    meta = {"reference": ref, "device": dev, "dtype": args.dtype, "files": files,
            "dtype_matches_the_record": "record/<ref>/MANIFEST.json says cuda/bfloat16",
            "n_positions": n_pos,
            "cells_iv": cells_iv, "n_cells_iv": len(cells_iv),
            "cells_20_recorded": len(cells_20),
            "seconds": {"model_load": t_load, "record": t_record,
                        "total": time.time() - t_all},
            "megabytes": sum(pathlib.Path(f).stat().st_size for f in files.values()) / 1e6,
            "provenance": provenance([f"{MODEL}@{ref}", "phase_battery/twin{1,2}/C.jsonl.gz"],
                                     reference=ref)}
    (WORK / f"C_iv_{ref}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"\n{ref}: {n_pos:,} positions, {len(cells_iv)} (iv) cells, "
          f"load {t_load:.0f}s, record {t_record/60:.1f} min, total "
          f"{(time.time()-t_all)/60:.1f} min -> {out}")
    if args.drop_model:
        try:
            cache = scan_cache_dir()
            for repo in cache.repos:
                if repo.repo_id == MODEL:
                    revs = [r for r in repo.revisions if ref in r.refs or r.commit_hash == ref]
                    if revs:
                        cache.delete_revisions(*[r.commit_hash for r in revs]).execute()
                        print(f"   model revision {ref} deleted from the cache")
        except Exception as exc:                                       # noqa: BLE001
            print(f"   (could not delete the model: {type(exc).__name__}: {exc})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="item", required=True)
    for name, fn in (("coarse", coarse), ("collect", collect), ("recordc", recordc),
                     ("turnover", turnover)):
        s = sub.add_parser(name)
        s.add_argument("--reference", default=None)
        s.add_argument("--force", action="store_true")
        s.add_argument("--batch", type=int, default=8)
        s.add_argument("--limit", type=int, default=0)
        s.add_argument("--drop-model", action="store_true")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--dtype", default="bfloat16")
        s.add_argument("--device", default=None)
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)




#: item 6: the first two dense-block pairs and the last, measured from the carrier index's own
#: reference list rather than named by hand
TURNOVER_PAIRS = (("pretrain_iter_0001000", "pretrain_iter_0002000"),
                  ("pretrain_iter_0002000", "pretrain_iter_0003000"),
                  ("pretrain_iter_0019000", "pretrain_iter_0020000"))


def turnover(args) -> None:
    """Curve 1's retention ratio under the classes-only split, beside the 20-cell values.

    The ratio is `overlap(t, t+1) / ceiling(t)` with overlap `|A n B| / |A|` on the carrier sets
    and the ceiling the twin-against-twin overlap at `t` -- `experiments/r1_turnover`'s own
    definitions, imported rather than restated, so the classes-only column is comparable to the
    20-cell one by construction.

    Paper 2's robustness reading: if the ratio under a coarser yardstick tracks the 20-cell one,
    the turnover curve is a property of the carriers and not of the cell split.
    """
    from experiments.atlas import p1_set
    refs = sorted({r for p in TURNOVER_PAIRS for r in p})
    names = sorted(p1_set())
    hows = ("twenty", "classes")
    sets: dict = {}
    t0 = time.time()
    for ref in refs:
        ys = {t: Yardstick.from_npz(dataset_record(ref, f"{t}__C_cells.npz"))
              for t in ("twin1", "twin2")}
        pool = {(t, h): pooled_yardstick(ys[t], h) for t in ys for h in hows}
        for i, c in enumerate(names, 1):
            try:
                for t in ("twin1", "twin2"):
                    acc = {k: np.load(read_acc(ref, f"{t}__{c}_{k}_acc.npz"))
                           for k in ("A", "B")}
                    for h in hows:
                        sets[(ref, t, h, c)] = carriers(concept(ref, t, c, pool[(t, h)], acc))
            except Exception as exc:                                   # noqa: BLE001
                print(f"   {ref} {c:20s} SKIP {type(exc).__name__}", flush=True)
                continue
            if i % 40 == 0:
                print(f"   {ref[-8:]:>8s} {i:3d}/{len(names)}  "
                      f"({(time.time()-t0)/60:.0f} min)", flush=True)
    def ov(a, b):
        return float(np.intersect1d(a, b).size / a.size)
    rows, summary = {}, {}
    for a, b in TURNOVER_PAIRS:
        for h in hows:
            vals = []
            for c in names:
                ka, kb = (a, "twin1", h, c), (b, "twin1", h, c)
                kc = (a, "twin2", h, c)
                if ka not in sets or kb not in sets or kc not in sets:
                    continue
                ceil = ov(sets[ka], sets[kc])
                if ceil <= 0:
                    continue
                r = ov(sets[ka], sets[kb]) / ceil
                rows[f"{a}->{b}|{h}|{c}"] = {"overlap": ov(sets[ka], sets[kb]),
                                             "ceiling": ceil, "ratio": r}
                vals.append(r)
            summary[f"{a}->{b}|{h}"] = {
                "n": len(vals), "median_ratio": float(np.median(vals)) if vals else None,
                "q1": float(np.quantile(vals, 0.25)) if vals else None,
                "q3": float(np.quantile(vals, 0.75)) if vals else None}
    for a, b in TURNOVER_PAIRS:
        e20, ecl = summary.get(f"{a}->{b}|twenty"), summary.get(f"{a}->{b}|classes")
        if e20 and ecl and e20["median_ratio"] is not None:
            print(f"  {a[-7:]}->{b[-7:]}  20-cell {e20['median_ratio']:.4f}   "
                  f"classes-only {ecl['median_ratio']:.4f}   "
                  f"difference {ecl['median_ratio']-e20['median_ratio']:+.4f}  n {e20['n']}")
    out = {"what": "curve 1's retention ratio under the classes-only split beside the 20-cell "
                   "values, at the first two dense-block pairs and the last",
           "pairs": [list(p) for p in TURNOVER_PAIRS],
           "ratio": "overlap(t, t+1) / ceiling(t); overlap is |A n B| / |A| and the ceiling is "
                    "twin against twin at t",
           "summary": summary, "rows": rows, "hours": (time.time() - t0) / 3600,
           "provenance": provenance(["record/<ref>/twin{1,2}__<c>_{A,B}_acc.npz"])}
    (OUT / "TURNOVER_CLASSES_ONLY.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {OUT / 'TURNOVER_CLASSES_ONLY.json'}")


if __name__ == "__main__":
    main()
