"""The full ablation pass, on the pilot's base, with the drift condition.

**What the pilot settled and this inherits.** D3's gate ran on three constructs at two
checkpoints: the **product-space** carrier set moves the loss (`For` +0.135 and +0.111 nats,
2.03 and 2.81 sd above a size-matched random control; `ListComp` +0.636 at 83.9 B, 22.7 sd) and
the **verdict-layer** set is a measurement null at every construct and reference (every effect
within +/-0.015 nats). So this pass carries the **product space only**, which is D3's own clause:
*the pass is built with whichever unit set the pilot earns it with*.

**What is new here, and it is the Settle chat's addition of 15 Oct: the drift condition.** A
carrier set is measured at a checkpoint, and the question is whether the set that mattered *then*
still matters *now*. So at each reference after the first, **two** sets are ablated per construct
and reported side by side:

* the set measured at **that** reference -- the current carriers;
* the set measured at the **previous** reference -- the drifted carriers.

At 83.9 B that is the 4.19 B set beside the 83.9 B set; at the last pretraining reference it is the
83.9 B set beside the current one. If the drifted set still moves the loss as much as the current
one, the turnover curve 1 measures is not turnover of the units that carry the construct's
behaviour -- which is a reading curve 1 cannot give on its own.

**Twenty random draws a set**, against the pilot's five: the pilot's 2.03 sd separation rested on
a spread estimated from five numbers, and 20 is what makes the weaker separations readable.

**The carrier sets are read from the carrier index, not the profiles.** `validation/carriers/
<ref>.npz` already holds the top-5% product-space indices per construct per twin -- 4.6 MB a
reference, built and verified in S3 -- against 95 MB per construct per reference for the profile
npz. That is the difference between 14 MB of downloads and 48 GB.

**Cost, stated as arithmetic with its assumption named rather than as a promise.** Per reference:
one baseline pass, 20 random passes, and one pass per construct per set (one set at the first
reference, two after it). With `n` constructs that is `n` or `2n` construct-sized passes plus 21
shared ones. The pilot's 48 passes over 96 prompts ran inside half an hour **including two ~3 GB
checkpoint downloads**, which does not separate compute from download, so this module **measures
the rate on the first construct and projects the total** before committing to it -- the third
Colab estimate this project has made from a rate it had not measured, and the first two were
wrong by factors of two and three.
"""
from __future__ import annotations
from profilebattery.paths import data_root

import argparse
import json
import os
import pathlib
import time

import numpy as np

D = data_root()
OUT = D / "validation/ablation"
#: the carrier index, wherever it is: the downloaded copy on this laptop, the dataset path on
#: Colab. S3 built it and VERIFY reported 0 mismatches, so it is the product-space set of record.
CARRIERS = next((p for p in (D / "validation/carriers",
                             D / "validation/carriers") if p.exists()),
                D / "validation/carriers")
N_RANDOM = 20
#: the nested prefix sizes the control's SD is read at, each double the last, so "the SD changes
#: by under 5% when the mask count doubles" is a comparison between adjacent entries
NESTED_AT = (50, 100, 200)
K_FRAC = 0.05
#: 4.19 B, 83.9 B and the last pretraining reference
REFERENCES = ("pretrain_iter_0001000", "pretrain_iter_0020000", "pretrain_iter_0484865")


def carrier_index(ref: str, root: pathlib.Path = None) -> dict:
    """`{construct: flat indices}` for the product space, from the carrier index."""
    root = root or CARRIERS
    p = root / f"{ref}.npz"
    if not p.exists():
        raise FileNotFoundError(f"no carrier index at {p}")
    out = {}
    with np.load(p) as z:
        for k in z.files:
            if k.startswith("twin1/") and k.endswith("|product"):
                out[k[len("twin1/"):-len("|product")]] = np.asarray(z[k], np.int64)
    return out


#: the mask draw's seed. Named so a rerun can state that it drew the SAME masks as the run
#: before it; the draw depends only on (k, n_layers, width, n, seed), and k is the carrier set
#: size, which is fixed at 5% of the neurons at every reference.
MASK_SEED = 20261015


def mask_from(indices, n_layers: int, width: int) -> np.ndarray:
    m = np.zeros(n_layers * width, dtype=bool)
    m[np.asarray(indices, np.int64)] = True
    return m.reshape(n_layers, width)


def random_masks(k: int, n_layers: int, width: int, n: int = N_RANDOM,
                 seed: int = MASK_SEED) -> list:
    """`n` size-matched random masks, drawn once a reference and reused for every construct.

    A random mask does not depend on the construct, so drawing it per construct would only add
    variance between constructs that is not in the quantity. The loss is still measured on each
    construct's own prompts, which is where the construct enters.
    """
    rng = np.random.default_rng(seed)
    return [mask_from(rng.choice(n_layers * width, size=k, replace=False), n_layers, width)
            for _ in range(n)]


def plan(n_constructs: int, refs=REFERENCES, n_random: int = N_RANDOM) -> dict:
    """The pass's size in forward passes, before anything is run."""
    per_ref = []
    for i, r in enumerate(refs):
        sets_per_construct = 1 if i == 0 else 2          # current, plus the drifted set
        per_ref.append({"reference": r, "sets_per_construct": sets_per_construct,
                        "construct_passes": n_constructs * sets_per_construct,
                        "shared_passes": 1 + n_random})
    total_construct = sum(e["construct_passes"] for e in per_ref)
    total_shared = sum(e["shared_passes"] for e in per_ref)
    return {"n_constructs": n_constructs, "references": list(refs),
            "n_random_draws": n_random, "per_reference": per_ref,
            "total_construct_passes": total_construct,
            "total_shared_passes": total_shared,
            "n_checkpoint_downloads": len(refs),
            "note": "a shared pass covers every construct's prompts at once, so it costs "
                    "n_constructs construct-sized passes; the projection below uses the measured "
                    "rate of the first construct-sized pass and nothing else"}


def project(plan_: dict, seconds_per_construct_pass: float) -> dict:
    """The wall time the plan implies at a measured rate, with the download time named apart."""
    cp = plan_["total_construct_passes"]
    sp = plan_["total_shared_passes"] * plan_["n_constructs"]
    compute = (cp + sp) * seconds_per_construct_pass
    return {"seconds_per_construct_pass": seconds_per_construct_pass,
            "construct_passes": cp, "shared_pass_equivalents": sp,
            "compute_hours": compute / 3600.0,
            "checkpoint_downloads": plan_["n_checkpoint_downloads"],
            "download_note": "about 3 GB a checkpoint; the pilot's two took a noticeable part of "
                             "its half hour, so the download time is reported apart from the "
                             "compute rather than folded into one number"}


def readable_population(battery: pathlib.Path, constructs, tok, per_set: int = 96) -> dict:
    """The constructs whose prompts carry **diagnostic positions** -- the only ones readable here.

    **This, and not the runtime, is what bounds the pass.** The ablation's loss is the mean
    next-token loss over each prompt's marked **obliged and dependent** positions, and the phase
    battery marks none for a construct that has none: measured over all 169 of P1 at 96 prompts
    each, **16** carry positions and 153 carry zero -- `For` 2,640, `ListComp` 2,970, `With` 1,680,
    against `ArithmeticError`, `Assert` and `AssertionError` at 0. The 16 are exactly the
    constructs the positional classes reach in `POSITIONS.json`, which step 4 measured
    independently. A pass over 169 would silently skip 153 of them, which is what the first timed
    trial did: its three constructs were `ArithmeticError`, `Assert` and `AssertionError`, it
    produced no output at all, and it exited 0.
    """
    import gzip

    from experiments.ablation_pilot import mark_positions
    out = {}
    for c in constructs:
        f = battery / "twin1" / f"{c}_A.jsonl.gz"
        if not f.exists():
            out[c] = {"n_positions": 0, "why": "no A set"}
            continue
        rows = [json.loads(l) for l in gzip.open(f, "rt")][:per_set]
        stored = sum(len(r.get("positions") or []) for r in rows)
        mark_positions(rows, c, tok)
        out[c] = {"n_positions": sum(len(r.get("positions") or []) for r in rows),
                  "n_stored": stored}
    return out


def require(refs, battery: pathlib.Path) -> list:
    """Every input the pass needs, as a list of what is absent and where to get it.

    Checked together rather than on first use: the pilot lost a GPU session to an absent
    `VERDICT_LAYERS.json` and this pass lost one to an absent `inventory_v2.json`, each surfacing
    as a bare `FileNotFoundError` from inside a helper after the runtime was already warm.
    """
    want = [(D / "inventory/inventory_v2.json", "inventory/inventory_v2.json"),
            (D / "validation/battery/SPLIT.json", "validation/battery/SPLIT.json")]
    for r in refs:
        want.append((CARRIERS / f"{r}.npz", f"validation/carriers/{r}.npz"))
    out = [f"{p}  <- dataset: {rel}" for p, rel in want if not p.exists()]
    sets = list((battery / "twin1").glob("*_A.jsonl.gz")) if (battery / "twin1").exists() else []
    if not sets:
        out.append(f"{battery / 'twin1'}/*_A.jsonl.gz  <- dataset: phase_battery/twin1/*")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--constructs", nargs="*", default=None,
                    help="default: every construct of P1 that the carrier index carries")
    ap.add_argument("--references", nargs="*", default=list(REFERENCES))
    ap.add_argument("--limit", type=int, default=0, help="first N constructs")
    ap.add_argument("--per-set", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--no-store-per-mask", dest="store_per_mask", action="store_false",
                    help="omit each mask's own loss rise. On by default: without it no RANK "
                         "can be taken, and the rank is what the long-tailed control needs")
    ap.add_argument("--all-constructs", action="store_true",
                    help="keep constructs with no diagnostic position, which the pass can only "
                         "skip; off by default so the plan's size is the readable size")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the pass's size and the projection, run nothing")
    ap.add_argument("--rate", type=float, default=None,
                    help="seconds per construct-sized pass, for --plan-only; measured otherwise")
    ap.add_argument("--battery", default=str(D / "phase_battery"),
                    help="the PHASE battery: P1's 169 constructs are its sets, and the carrier "
                         "index is the phase recording's")
    ap.add_argument("--twin", default="twin1")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # **Every input is checked at once, before anything is loaded.** Two GPU sessions were spent
    # discovering missing inputs one at a time -- VERDICT_LAYERS.json in the pilot, then
    # inventory_v2.json here -- each failing deep inside a helper with a bare FileNotFoundError.
    # A pass that needs five files should say which five are absent in one line, and before it
    # downloads a 3 GB checkpoint.
    missing = require(args.references, pathlib.Path(os.path.expanduser(args.battery)))
    if missing:
        raise SystemExit("missing inputs, none of which this pass can invent:\n  "
                         + "\n  ".join(missing))

    from profilebattery.profiles.names import tree_name
    p1 = [tree_name(c) for c in json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]]
    first = carrier_index(args.references[0])
    cs = args.constructs or [c for c in p1 if c in first]

    # the readable population, measured before the plan is printed rather than discovered
    # construct by construct inside the loop
    from transformers import AutoTokenizer
    from profilebattery.record.sweep import REPO as MODEL_REPO
    tk = AutoTokenizer.from_pretrained(MODEL_REPO, revision="pretrain_iter_0008000",
                                       trust_remote_code=True)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token or tk.unk_token
    B = pathlib.Path(os.path.expanduser(args.battery))
    pos = readable_population(B, cs, tk, args.per_set)
    readable = [c for c in cs if pos[c]["n_positions"] > 0]
    print(f"{len(cs)} constructs asked for; {len(readable)} carry diagnostic positions and are "
          f"readable, {len(cs) - len(readable)} carry none")
    if len(readable) < len(cs):
        print(f"  the readable ones: {readable}")
    if not args.all_constructs:
        cs = readable
    if args.limit:
        cs = cs[:args.limit]
    if not cs:
        raise SystemExit("no construct in the request carries a diagnostic position: the "
                         "ablation's loss is read at marked obliged and dependent positions, and "
                         "a construct with none cannot be read here")
    pl = plan(len(cs), args.references, args.n_random)
    print(f"{len(cs)} constructs of P1 in the carrier index; {len(args.references)} references")
    for e in pl["per_reference"]:
        print(f"  {e['reference']:24s} {e['sets_per_construct']} set(s) a construct -> "
              f"{e['construct_passes']} construct passes + {e['shared_passes']} shared")
    print(f"  total: {pl['total_construct_passes']} construct passes, "
          f"{pl['total_shared_passes']} shared (each costing {len(cs)} construct passes), "
          f"{pl['n_checkpoint_downloads']} checkpoint downloads")
    if args.plan_only:
        for rate in ([args.rate] if args.rate else (1.0, 2.0, 5.0, 10.0, 25.0)):
            pr = project(pl, rate)
            print(f"  at {rate:5.1f} s a construct pass -> {pr['compute_hours']*60:6.0f} min "
                  f"({pr['compute_hours']:5.2f} h) of compute, at {args.n_random} masks")
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(args.out) / "PLAN.json").write_text(json.dumps(
            {"plan": pl, "projections": {str(r): project(pl, r)
                                         for r in (1.0, 2.0, 5.0, 10.0, 25.0)}},
            indent=1, default=float))
        print(f"  -> {pathlib.Path(args.out) / 'PLAN.json'}")
        return

    import gzip

    import torch
    from transformers import AutoModelForCausalLM
    from experiments.ablation_pilot import Ablator, loss_at_positions, mark_positions
    from profilebattery.record.sweep import REPO
    assert torch.cuda.is_available(), (
        "this pass needs CUDA: the pilot measured 33 minutes on MPS without finishing one "
        "construct")
    device, dtype = "cuda", torch.bfloat16
    tok = tk

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    # the device is recorded: the 22 Oct artefacts carry no device field, so the hardware
    # behind their timings could not be recovered and a later cost estimate had to be
    # reconstructed from memory
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"device: {device} / {gpu} / {dtype}", flush=True)
    report = {"plan": pl, "n_random": args.n_random, "device": device, "gpu": gpu,
              "dtype": str(dtype), "nested_at": list(NESTED_AT),
              "store_per_mask": bool(args.store_per_mask), "mask_seed": MASK_SEED,
              "references": {}}
    prev_sets = None
    for i, ref in enumerate(args.references):
        t0 = time.time()
        sets = carrier_index(ref)
        model = AutoModelForCausalLM.from_pretrained(REPO, revision=ref, dtype=dtype,
                                                     trust_remote_code=True).to(device).eval()
        core = model.model
        n_layers = core.config.num_hidden_layers
        width = core.config.intermediate_size
        abl = Ablator(core, n_layers)
        k = max(1, int(round(K_FRAC * n_layers * width)))
        rnd_masks = random_masks(k, n_layers, width, args.n_random)
        print(f"\n=== {ref}: {n_layers} layers x {width}, k = {k:,}; "
              f"model loaded in {time.time() - t0:.0f}s ===", flush=True)
        # **resumable, because a Colab session is reclaimed when its browser disconnects.** The
        # per-reference file is rewritten every ten constructs and a construct already in it is
        # skipped, which is what `run_s15` does and what S15 needed three times.
        fp = pathlib.Path(args.out) / f"FULL_{ref}.json"
        out = (json.loads(fp.read_text()).get("constructs") or {}) if fp.exists() else {}
        # **A construct is done only if it carries every set this reference expects.** The timed
        # trial ran one reference, where no drifted set exists, and wrote For, If and ListComp
        # with `current` alone; treating those as complete would make the real pass skip them and
        # never measure their drifted sets -- the condition the pass exists for.
        expect_drift = i > 0
        done = {c for c, v in out.items()
                if "error" in v or ("current" in v
                                    and (not expect_drift or "drifted" in v
                                         or v.get("drift_unavailable")))}
        measured = 0
        if done:
            print(f"  resuming: {len(done)} constructs already in {fp.name}", flush=True)
        for j, c in enumerate(cs, 1):
            if c in done:
                continue
            if c not in sets:
                out[c] = {"error": "absent from this reference's carrier index"}
                print(f"  {c}: absent from the carrier index -- skipped", flush=True)
                continue
            f = B / "twin1" / f"{c}_A.jsonl.gz"
            if not f.exists():
                out[c] = {"error": f"no A set at {f.name}"}
                print(f"  {c}: {out[c]['error']} -- skipped", flush=True)
                continue
            rows = [json.loads(l) for l in gzip.open(f, "rt")][: args.per_set]
            derived = mark_positions(rows, c, tok)
            n_pos = sum(len(r.get("positions") or []) for r in rows)
            if n_pos == 0:
                out[c] = {"error": "no diagnostic positions"}
                print(f"  {c}: no diagnostic positions -- skipped", flush=True)
                continue
            abl.mask = None
            t1 = time.time()
            base = loss_at_positions(model, tok, rows, device, args.batch)
            e = {"baseline_nats": base, "n_prompts": len(rows), "n_positions": n_pos,
                 "positions_derived": derived}
            # the current set, and -- after the first reference -- the one measured at the
            # PREVIOUS reference, which is the drift condition
            todo = {"current": sets[c]}
            if prev_sets is None or c not in prev_sets:
                e["drift_unavailable"] = ("this is the first reference of the run, so no earlier "
                                          "set exists to ablate beside the current one"
                                          if prev_sets is None else
                                          "absent from the previous reference's carrier index")
            if prev_sets is not None and c in prev_sets:
                todo["drifted"] = prev_sets[c]
                e["drifted_from"] = args.references[i - 1]
                shared = len(np.intersect1d(sets[c], prev_sets[c]))
                e["overlap_with_previous_set"] = int(shared)
                e["overlap_share"] = float(shared / len(sets[c]))
            for name, idxs in todo.items():
                abl.mask = mask_from(idxs, n_layers, width)
                e[name] = loss_at_positions(model, tok, rows, device, args.batch) - base
            rnd = []
            for m in rnd_masks:
                abl.mask = m
                rnd.append(loss_at_positions(model, tok, rows, device, args.batch) - base)
            abl.mask = None
            e["random_mean"] = float(np.mean(rnd))
            e["random_sd"] = float(np.std(rnd, ddof=1))
            # EVERY mask's rise, not just the mean and SD. The control is long-tailed at the
            # two later references -- its SD is ~2.2x its own mean and still rising at 200
            # masks -- so a control UNIT has no settled denominator there. A RANK among the
            # masks needs none: "the carrier set's rise exceeds all 200 random sets" is exact
            # however the tail behaves. 200 floats a construct; the passes were run anyway.
            if args.store_per_mask:
                e["random_per_mask"] = [float(x) for x in rnd]
            # NESTED prefixes of the SAME draws (A1.1, 23 Oct). Two draws of the control gave
            # +5.16 and +3.7 control units at the end, so the control's own spread has to be
            # shown to have settled. Reading the SD at 50, 100 and 200 from prefixes of one
            # 200-draw set costs 200 passes, not 50+100+200; and the three figures then differ
            # only in sample size, not in which masks were drawn.
            e["control_stability"] = {}
            prev = None
            for m in NESTED_AT:
                if len(rnd) < m:
                    continue
                sd_m = float(np.std(rnd[:m], ddof=1))
                row = {"n_masks": m, "control_mean": float(np.mean(rnd[:m])),
                       "control_sd": sd_m,
                       "change_from_half": (None if prev in (None, 0)
                                            else float(abs(sd_m - prev) / prev))}
                e["control_stability"][str(m)] = row
                prev = sd_m
            last = e["control_stability"].get(str(NESTED_AT[-1]), {})
            e["control_sd_settled"] = (last.get("change_from_half") is not None
                                       and last["change_from_half"] < 0.05)
            e["n_masks_reached"] = len(rnd)
            for name in todo:
                e[f"{name}_over_random_in_sd"] = (
                    None if e["random_sd"] == 0 else
                    float((e[name] - e["random_mean"]) / e["random_sd"]))
            e["seconds"] = time.time() - t1
            out[c] = e
            measured += 1
            if measured == 1:
                npass = 1 + len(todo) + len(rnd_masks)
                rate = e["seconds"] / npass
                pr = project(pl, rate)
                # minutes, not hours to one decimal: a 3-construct trial printed "0.0 h",
                # which is the line a reader acts on. And the mask count is named, because the
                # projection is only for the n_random this trial ran at.
                print(f"  measured {rate:.2f}s a construct-sized pass on the first construct "
                      f"-> AT {args.n_random} MASKS the whole pass projects to "
                      f"{pr['compute_hours'] * 60:.0f} min "
                      f"({pr['compute_hours']:.2f} h) of compute plus "
                      f"{pr['checkpoint_downloads']} checkpoint downloads", flush=True)
            if j % 10 == 0:
                print(f"  {j}/{len(cs)} constructs, {time.time() - t0:.0f}s", flush=True)
                (pathlib.Path(args.out) / f"FULL_{ref}.json").write_text(
                    json.dumps({"reference": ref, "n_random": args.n_random,
                                "constructs": out}, indent=1, default=float))
        (pathlib.Path(args.out) / f"FULL_{ref}.json").write_text(
            json.dumps({"reference": ref, "n_random": args.n_random, "plan": pl,
                        "constructs": out}, indent=1, default=float))
        ok = [v for v in out.values() if "current" in v]
        if not ok:
            import collections as _c
            print(f"  {ref}: NOTHING measured -- "
                  f"{dict(_c.Counter(v.get('error', '?') for v in out.values()))}", flush=True)
        if ok:
            cur = [v["current_over_random_in_sd"] for v in ok
                   if v.get("current_over_random_in_sd") is not None]
            dr = [v["drifted_over_random_in_sd"] for v in ok
                  if v.get("drifted_over_random_in_sd") is not None]
            print(f"  {ref}: {len(ok)} constructs; current set median "
                  f"{np.median(cur):+.2f} sd above the control"
                  + (f", drifted set median {np.median(dr):+.2f} sd" if dr else ""))
            if dr:
                keeps = sum(1 for v in ok
                            if v.get("drifted_over_random_in_sd") is not None
                            and v["drifted_over_random_in_sd"] >= 2.0)
                print(f"      the drifted set still moves the loss (>= 2 sd) for {keeps} of "
                      f"{len(dr)}; median overlap with the current set "
                      f"{np.median([v['overlap_share'] for v in ok if 'overlap_share' in v]):.3f}")
        del model
        torch.cuda.empty_cache()
        report["references"][ref] = out
        prev_sets = sets


if __name__ == "__main__":
    main()
