r"""Item 4 — the carrier sets named by three yardstick splits, for the ablation comparison.

For the 16 position-marked constructs at the three trained references, the carrier set under:

  `v`       the argued code-structural split (48 cells)
  `twenty`  the record's own split, as stored
  `one`     a single pooled cell

`masks` computes and saves the three sets per construct and reference; `ablation_splits` then
feeds them to the ablation, which is the model run. Splitting it this way means the expensive
pass reads a file rather than recomputing profiles inside a GPU loop, and the sets can be
checked before any of that is spent.

Supporting evidence for the argued choice, not a selection rule (the Design, 19 Oct).

    python experiments/ablation_splits.py masks
"""
from __future__ import annotations

import argparse
import gzip
import json
import time

import numpy as np

from experiments.atlas import CARRIERS, D, K, OUT, REPO, provenance, token
from experiments.b10v import (TRAINED, WORK, MODEL, TOK_REV, _car, clean_mapping,
                              new_cells_of_set, remapped_yardstick, yardstick_from)

SPLITS = ("v", "twenty", "one")
OUTDIR = D / "validation/paper1/ablation_splits"


def sixteen() -> list[str]:
    """The 16 position-marked constructs: step 4's forced and dependent classes reach exactly
    these, which is how `p8_cheap` defines the same population."""
    s4 = json.loads((D / "validation/r2/STEP4_FLAT_pretrain.json").read_text())
    out = {k.split("|")[0] for k in s4["forced"]["series"]}
    for cls in ("governed", "scope"):
        out |= {k.split("|")[0] for k in (s4["dependent"].get(cls) or {}).get("series", {})}
    return sorted(out)


def masks(args) -> None:
    from experiments.b9_b10 import stream as _rs
    from experiments.b10 import pooled_yardstick
    from experiments.r1_turnover import load_index
    from profilebattery.profiles.profile import cell_of, from_accumulators, recorder_order
    from profilebattery.profiles.yardstick import Yardstick
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from experiments.atlas import dataset_record
    tk = AutoTokenizer.from_pretrained(MODEL, revision=TOK_REV, trust_remote_code=True)
    names = sixteen()
    print(f"the position-marked population: {len(names)} constructs")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0, rows = time.time(), {}
    for ref in TRAINED:
        idx = load_index(CARRIERS, [ref]).get(ref, {})
        ys_v = {t: yardstick_from(str(WORK / f"C_{ref}__{t}.npz"), "V")
                for t in ("twin1", "twin2")}
        ys20 = {t: Yardstick.from_npz(dataset_record(ref, f"{t}__C_cells.npz"))
                for t in ("twin1", "twin2")}
        layers = sorted({l for l, _ in ys_v["twin1"].cells})
        for c in names:
            got = {}
            try:
                for t in ("twin1", "twin2"):
                    acc = {k: np.load(_rs(ref, f"{t}__{c}_{k}_acc.npz")[1]) for k in ("A", "B")}
                    rws = {k: recorder_order([json.loads(l) for l in gzip.open(
                        hf_hub_download(REPO, f"phase_battery/{t}/{c}_{k}.jsonl.gz",
                                        repo_type="dataset", token=token()), "rt")])
                        for k in ("A", "B")}
                    for split in SPLITS:
                        parts = {}
                        for k in ("A", "B"):
                            if split == "v":
                                mapping, _ = clean_mapping(rws[k], tk, "V")
                                old = {cell_of(r) for r in rws[k]}
                                y = remapped_yardstick(ys_v[t], mapping, old, layers)
                            elif split == "twenty":
                                y = ys20[t]
                            else:
                                y = pooled_yardstick(ys20[t], "one")
                            parts[k] = from_accumulators(acc[k], y, c, k, stream="h").mean
                        got[(t, split)] = _car(parts["A"] - parts["B"])
            except Exception as exc:                                   # noqa: BLE001
                print(f"   {ref} {c:20s} SKIP {type(exc).__name__}: {str(exc)[:50]}", flush=True)
                continue
            e = {}
            for split in SPLITS:
                np.save(OUTDIR / f"{ref}__{c}__{split}.npy", got[("twin1", split)])
                e[split] = {"n": int(got[("twin1", split)].size),
                            "twin_ceiling": float(np.intersect1d(
                                got[("twin1", split)], got[("twin2", split)]).size / K)}
                key = f"twin1/{c}|product"
                if key in idx:
                    st = np.sort(np.asarray(idx[key], np.int64))
                    e[split]["overlap_with_stored"] = float(
                        np.intersect1d(got[("twin1", split)], st).size / K)
            rows[f"{ref}|{c}"] = e
            print(f"   {ref[-8:]:>8s} {c:20s} v {e['v']['overlap_with_stored']:.4f}  "
                  f"twenty {e['twenty']['overlap_with_stored']:.4f}  "
                  f"one {e['one']['overlap_with_stored']:.4f}  "
                  f"| {(time.time()-t0)/60:.1f} min", flush=True)
    doc = {"what": "carrier sets under the argued split (v), the record's 20 cells and one "
                   "pooled cell, for the 16 position-marked constructs",
           "population": names, "splits": list(SPLITS), "k": K,
           "identity_expectation": "the `twenty` row must reproduce the stored carrier set at "
                                   "overlap 1.0; anything else means the route is wrong, not "
                                   "the split",
           "rows": rows, "minutes": (time.time() - t0) / 60, "directory": str(OUTDIR),
           "provenance": provenance(["record/<ref>/twin{1,2}__<c>_{A,B}_acc.npz",
                                     "validation/paper1/b10v/C_<ref>__twin{1,2}.npz"])}
    (OUT / "ABLATION_SPLIT_MASKS.json").write_text(json.dumps(doc, indent=1, default=float))
    ident = [e["twenty"].get("overlap_with_stored") for e in rows.values()
             if "overlap_with_stored" in e["twenty"]]
    if ident:
        print(f"\nidentity check on `twenty`: min {min(ident):.6f}, "
              f"median {np.median(ident):.6f} over {len(ident)}")
    print(f"-> {OUT / 'ABLATION_SPLIT_MASKS.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="item", required=True)
    s = sub.add_parser("masks")
    s.set_defaults(fn=masks)
    r = sub.add_parser("run")
    r.add_argument("--reference", default=None)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--per-set", type=int, default=96)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--n-random", type=int, default=20)
    r.add_argument("--device", default=None)
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=run)
    a = ap.parse_args()
    a.fn(a)




def run(args) -> None:
    """Zero each split's carrier set and read the loss rise at the marked positions.

    Built on the **pilot's** machinery, not the full pass's: the full pass asserts CUDA because
    it ablates 169 constructs, while the pilot ran this same measurement on MPS. Item 4 is 16
    constructs, so the question is one of measurement rather than of principle -- and the rate is
    measured on one construct before the rest is committed, which is the rule A46 left.

    Per construct and reference: the baseline loss at the marked positions, the loss with each
    split's carrier set zeroed, and `n_random` size-matched random masks. The reported number is
    the rise over the random control's mean in units of the control's own standard deviation,
    which is what makes the three splits comparable when their sets differ in composition but
    not in size.
    """
    import gzip
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.ablation_pilot import Ablator, loss_at_positions, mark_positions
    from profilebattery.record.sweep import REPO as MODEL_REPO_ID
    dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_REPO_ID, revision=TOK_REV, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    names = sixteen()[:args.limit] if args.limit else sixteen()
    refs = [args.reference] if args.reference else list(TRAINED)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261019)

    def drop_model(r):
        """Delete a checkpoint from the HF cache once its reference is finished.

        The three checkpoints are 3.81 GB each and this volume runs at 100%; left in place they
        stack to 11.4 GB and the second fetch fails mid-pass. They are public weights and
        re-downloadable, so the cache is the right place to reclaim.
        """
        try:
            from huggingface_hub import scan_cache_dir
            c = scan_cache_dir()
            hashes = [rev.commit_hash for repo in c.repos if repo.repo_id == MODEL_REPO_ID
                      for rev in repo.revisions if r in rev.refs]
            if hashes:
                c.delete_revisions(*hashes).execute()
                print(f"  freed the {r} checkpoint from the cache", flush=True)
        except Exception as exc:                                       # noqa: BLE001
            print(f"  (could not free {r}: {type(exc).__name__})", flush=True)

    for ref in refs:
        fp = OUTDIR / f"ABLATION_{ref}.json"
        out = json.loads(fp.read_text())["constructs"] if fp.exists() else {}
        # **Nothing to do here? Then do not fetch the checkpoint.** Freeing each model after its
        # reference (for disk) means a resume would otherwise re-download 3.81 GB only to skip
        # every construct in it.
        if not args.force and all(c in out for c in names):
            print(f"=== {ref}: all {len(names)} already measured, skipped ===", flush=True)
            continue
        t0 = time.time()
        # **Retry the fetch.** A transient network failure here killed the pass once, at
        # 18:45, between the second reference and the third; nothing ran for four and a half
        # hours. The constructs already measured survive -- the file is written after each one
        # -- but the pass should not die of a dropped connection it could wait out.
        model = None
        for attempt in range(1, 7):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_REPO_ID, revision=ref, dtype=torch.bfloat16,
                    trust_remote_code=True).to(dev).eval()
                break
            except Exception as exc:                                   # noqa: BLE001
                wait = min(60 * attempt, 300)
                print(f"  fetch of {ref} failed ({type(exc).__name__}); "
                      f"attempt {attempt} of 6, waiting {wait}s", flush=True)
                time.sleep(wait)
        if model is None:
            print(f"  giving up on {ref} after six attempts", flush=True)
            continue
        core = model.model
        n_layers, width = core.config.num_hidden_layers, core.config.intermediate_size
        abl = Ablator(core, n_layers)
        k = K
        rnd = [np.sort(rng.choice(n_layers * width, size=k, replace=False))
               for _ in range(args.n_random)]
        print(f"=== {ref}: {n_layers} x {width}, k={k:,}, {args.n_random} controls, "
              f"model in {time.time()-t0:.0f}s, device {dev} ===", flush=True)
        for c in names:
            if c in out and not args.force:
                continue
            f = D / "phase_battery" / "twin1" / f"{c}_A.jsonl.gz"
            if not f.exists():
                out[c] = {"error": "no A set"}
                continue
            rows = [json.loads(l) for l in gzip.open(f, "rt")][:args.per_set]
            mark_positions(rows, c, tok)
            n_pos = sum(len(r.get("positions") or []) for r in rows)
            if n_pos == 0:
                out[c] = {"error": "no diagnostic positions"}
                print(f"  {c}: no diagnostic positions", flush=True)
                continue
            t1 = time.time()
            abl.mask = None
            base = loss_at_positions(model, tok, rows, dev, args.batch)
            e = {"baseline_nats": base, "n_prompts": len(rows), "n_positions": n_pos}
            for split in SPLITS:
                p = OUTDIR / f"{ref}__{c}__{split}.npy"
                if not p.exists():
                    e[split] = {"error": "no mask"}
                    continue
                abl.mask = _mask(np.load(p), n_layers, width)
                e[split] = {"nats": loss_at_positions(model, tok, rows, dev, args.batch)}
            vals = []
            for m in rnd:
                abl.mask = _mask(m, n_layers, width)
                vals.append(loss_at_positions(model, tok, rows, dev, args.batch))
            abl.mask = None
            mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            e["random_mean"], e["random_sd"], e["n_random"] = mu, sd, len(vals)
            for split in SPLITS:
                if "nats" in e.get(split, {}):
                    # **Both, each for its own job** (Piotr, 21 Oct): the sd figure against the
                    # control is the test, the nat rise over the control's mean is the size.
                    # Keeping only the sd figure is how Section 4.8 came to quote +6.00 for
                    # `Try` -- a nat rise -- beside medians in standard deviations, with no unit
                    # on either. With 20 draws the control sd carries about +/-16% of sampling
                    # error, which moved `Try` from +81 to +49 sd between two runs whose nat
                    # rises differ by 0.003.
                    e[split]["rise_over_baseline"] = e[split]["nats"] - base
                    e[split]["rise_over_random_mean"] = e[split]["nats"] - mu
                    e[split]["over_random_in_sd"] = ((e[split]["nats"] - mu) / sd
                                                     if sd > 0 else float("nan"))
            e["seconds"] = time.time() - t1
            out[c] = e
            fp.write_text(json.dumps({"reference": ref, "device": dev, "k": k,
                                      "n_random": args.n_random, "per_set": args.per_set,
                                      "constructs": out,
                                      "provenance": provenance(
                                          ["validation/paper1/ablation_splits/*.npy"],
                                          reference=ref)}, indent=1, default=float))
            print(f"  {c:16s} base {base:.4f}  " + "  ".join(
                f"{s} {e[s].get('over_random_in_sd', float('nan')):+.2f}sd" for s in SPLITS)
                + f"   ({e['seconds']:.0f}s)", flush=True)
        del model, core, abl
        import gc
        gc.collect()
        drop_model(ref)
    print("-> " + str(OUTDIR))


def _mask(indices, n_layers: int, width: int) -> np.ndarray:
    m = np.zeros(n_layers * width, bool)
    m[np.asarray(indices, np.int64)] = True
    return m.reshape(n_layers, width)


if __name__ == "__main__":
    main()
