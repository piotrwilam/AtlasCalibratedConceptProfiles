"""The loss at a position identified by class and token — S16 as rebuilt, 11 Oct.

    python experiments/position_loss.py --reference anneal_iter_0001000 --out POSITIONS.json

**Why this exists rather than another offset pass.** `loss_probe` anchors every reading at an
offset from the construct token. That is correct for what it was built for and wrong for two
things the plan needs:

* **the matched controls have no construct token.** `_CTL` rows carry
  `site_construct_token = None` by construction — the header is replaced by a plain assignment —
  so an offset read produces nothing. S16's first pass wrote 11 empty sets at 54 references
  before anything noticed.
* **the dependent positions were never read on either side.** `slot` is `None` for the governed
  and scope positions of the **A** rows too, so the A/B pass never read them. Neither half of
  the role gap existed.

Both are the same fault, and the plan already names the cure: *a loss is read at a position
identified by class and token, never by offset* (S3's join). Every row carries `positions`, a
record per position with `class` in {obliged, governed, scope}, a `role`, and `token_index` —
the index **in that row's own prompt**. This reads there.

**The two pairings, and neither is `pair_id`.** `pair_id` is `None` in all 720 rows on both
sides, so a join on it matches every control to one arbitrary A prompt and reports success;
that cost me a wrong claim on 10 Oct. The prompt pairing is `control_of` → the A row's
`prompt_id` (7,230 of 7,230). The *position* pairing is a control position's `control_index`,
which is its counterpart's `token_index` in the A prompt — the positions sit at different
indices because the replaced header is a different length, so sharing an index would read the
wrong token and return a plausible number.

**What it writes.** `record/<ref>/POSITIONS.json`, a new artefact and deliberately not a merge
into `LOSS.json`: that file is offset-keyed, this reading is position-keyed, and keeping them
apart removes the fetch-modify-reupload hazard that nearly cost the 376-set A and B pass.
"""
from __future__ import annotations
from profilebattery.paths import data_root

import argparse
import collections
import gzip
import json
import os
import pathlib
import time

import numpy as np

TIME_CHUNK = 64
#: the dependent classes: the role gap is read here and nowhere else (Cowork, call 1, 10 Oct).
#: A control's header is a plain assignment, so A's obliged positions have no counterpart in
#: it and the gap there does not exist by construction.
DEPENDENT = ("governed", "scope")


def battery_dir() -> pathlib.Path:
    for var in ("PB_BATTERY", "PB_BATTERY_DIR"):
        if os.environ.get(var):
            return pathlib.Path(os.path.expanduser(os.environ[var]))
    return (data_root() / "phase_battery")


def sets_with_positions(batt: pathlib.Path, probe_rows: int = 20):
    """`(constructs whose A set names positions, the rest)`.

    **Only 27 of the 188 A sets carry any positions**, measured on battery `570b42db`. The other
    161 are builtins and exception names -- `ArithmeticError`, `BinOp`, `Break`, `abs`, `zip` --
    whose prompts have no obliged, governed or scope position by design. Reading them costs 74%
    of the pass's tokens and yields nothing: 4.4 of every 6.1 minutes a reference, measured at
    24.0k tok/s on an A100.

    The first timed reference showed it as a column of `0 class/role`, and `verify` would have
    refused the run on the first reference -- correctly, because a set that reads nothing is a
    failure. The set list was what was wrong, not the guard.

    A construct is included if any of its first `probe_rows` prompts names a position. Probing
    rather than scanning because the distinction is structural -- a construct either has a
    grammar with obliged positions or it does not -- and a full scan of 188 sets is 135,000 JSON
    parses to learn something constant. The excluded list is returned, not discarded, so the
    output can record them and an absent set never looks like a forgotten one.
    """
    inc, exc = [], []
    for f in sorted((batt / "twin1").glob("*_A.jsonl.gz")):
        c = f.name[:-len("_A.jsonl.gz")]
        found = False
        with gzip.open(f, "rt") as fh:
            for i, line in enumerate(fh):
                if i >= probe_rows:
                    break
                if json.loads(line).get("positions"):
                    found = True
                    break
        (inc if found else exc).append(c)
    return inc, exc


def rows_of(path: pathlib.Path) -> list:
    return [json.loads(l) for l in gzip.open(path, "rt")]


def losses_at_positions(model, tok, rows, device, batch):
    """`prompt_id -> {core, control_of, positions: [...]}` with a loss at each named position.

    `ce[:, i]` is the loss of predicting `ids[:, i+1]`, so the loss *of* the token at index `j`
    is `ce[:, j-1]` — the same indexing `loss_probe` uses, with `j` taken from the position
    record instead of from an offset. A position outside the unpadded length is recorded with
    `loss: None` and a reason rather than dropped, so a set that reads nothing cannot look like
    a set that read zero.
    """
    import torch

    out = {}
    rows = sorted(rows, key=lambda r: r.get("n_tokens") or len(r["text"]))
    # **`no_grad`, which `loss_probe.losses_for` carries as a decorator and I copied the body
    # without.** Without it the logits keep a graph, `ce.cpu().numpy()` raises "Can't call
    # numpy() on Tensor that requires grad", and before raising it holds a 96,640-wide
    # vocabulary's worth of activations per batch. It is a `with` block rather than a decorator
    # so that `torch` stays a local import and the pure functions here remain testable without
    # it. This is the third thing today I carried over from a working version minus something
    # that sat just outside the function: `-m` instead of a script path, the xet retry, this.
    with torch.no_grad():
        out.update(_forward(model, tok, rows, device, batch, torch))
    return out


def _forward(model, tok, rows, device, batch, torch):
    out = {}
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        enc = tok([r["text"] for r in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        T = ids.shape[1]
        ce = torch.empty(ids.shape[0], T - 1, dtype=torch.float32, device=device)
        for a in range(0, T - 1, TIME_CHUNK):
            b = min(a + TIME_CHUNK, T - 1)
            lp = torch.log_softmax(logits[:, a:b, :].float(), dim=-1)
            ce[:, a:b] = -lp.gather(-1, ids[:, a + 1:b + 1].unsqueeze(-1)).squeeze(-1)
        ce = ce.cpu().numpy()
        lens = mask.sum(1).cpu().numpy()
        idn = ids.cpu().numpy()
        for bi, r in enumerate(chunk):
            n = int(lens[bi])
            recs = []
            for q in r.get("positions") or []:
                j = q.get("token_index")
                ok = j is not None and 1 <= int(j) < n
                recs.append({
                    "class": q.get("class"), "role": q.get("role"),
                    "token_index": j, "control_index": q.get("control_index"),
                    "matched": bool(q.get("matched")), "forced": q.get("forced"),
                    "floor_nats": q.get("floor_nats"), "bound_name": q.get("bound_name"),
                    "piece": (tok.convert_ids_to_tokens(int(idn[bi, int(j)])) if ok else None),
                    "loss": (float(ce[bi, int(j) - 1]) if ok else None),
                    "why": None if ok else f"token_index {j} outside 1..{n - 1}",
                })
            out[r["prompt_id"]] = {"core": r.get("core_id"),
                                   "control_of": r.get("control_of"), "positions": recs}
    return out


def core_stats(by_core: dict) -> dict:
    """Mean, n, and the between-core SE — the scale every reading in the plan is in."""
    cores = {c: float(np.mean(v)) for c, v in by_core.items() if v}
    n = sum(len(v) for v in by_core.values())
    if not cores:
        return {"mean": None, "n": 0, "cores": 0, "se": None}
    vals = np.array(sorted(cores.values()), dtype=np.float64)
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return {"mean": float(np.mean(vals)), "n": n, "cores": len(vals), "se": se,
            "core_means": {c: cores[c] for c in sorted(cores)}}


def aggregate(detail: dict) -> dict:
    """One set's losses, by (class, role) and by (class, role, piece)."""
    cr = collections.defaultdict(lambda: collections.defaultdict(list))
    crp = collections.defaultdict(lambda: collections.defaultdict(list))
    unread = collections.Counter()
    for pid, d in detail.items():
        for q in d["positions"]:
            if q["loss"] is None:
                unread[q["why"]] += 1
                continue
            cr[(q["class"], q["role"])][d["core"]].append(q["loss"])
            crp[(q["class"], q["role"], q["piece"])][d["core"]].append(q["loss"])
    return {
        "by_class_role": {f"{a}|{b}": core_stats(v) for (a, b), v in sorted(cr.items())},
        "by_class_role_piece": {f"{a}|{b}|{p}": core_stats(v)
                                for (a, b, p), v in sorted(crp.items())},
        "unread": dict(unread),
    }


def role_gap(a_detail: dict, c_detail: dict) -> dict:
    """The paired role gap: control loss minus A loss, at the same body position.

    **Paired per prompt and per position, then aggregated by core.** The two sides share
    `core_id`, so the difference of two means would throw away the pairing the battery was
    built to provide; this keeps it, which is what C0.14's "exceeds one noise unit" gate needs.
    A control position is matched to the A position whose `token_index` equals the control
    record's `control_index`; unmatched positions and unresolvable pairs are counted and
    excluded, never silently dropped.
    """
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    skipped = collections.Counter()
    for pid, c in c_detail.items():
        a = a_detail.get(c["control_of"])
        if a is None:
            skipped["no paired A prompt"] += 1
            continue
        a_at = {q["token_index"]: q for q in a["positions"] if q["loss"] is not None}
        for q in c["positions"]:
            if q["class"] not in DEPENDENT:
                continue
            if not q["matched"]:
                skipped["position not matched"] += 1
                continue
            if q["loss"] is None:
                skipped["control position unread"] += 1
                continue
            m = a_at.get(q["control_index"])
            if m is None:
                skipped["no A position at control_index"] += 1
                continue
            by[(q["class"], q["role"])][c["core"]].append(q["loss"] - m["loss"])
    return {"by_class_role": {f"{a}|{b}": core_stats(v) for (a, b), v in sorted(by.items())},
            "skipped": dict(skipped)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-sets", type=int, default=None,
                    help="first N constructs only, for a timed reference")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from experiments.loss_probe import load_lm, pick_device
    device = pick_device()
    if device == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU: this pass is 8.4 million tokens a reference and would take "
                         "days on cpu. Use an L4 or A100, or pass --allow-cpu.")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    batt = battery_dir()
    tok = AutoTokenizer.from_pretrained(
        __import__("profilebattery.record.sweep", fromlist=["REPO"]).REPO,
        revision="pretrain_iter_0008000", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token

    constructs, skipped = sets_with_positions(batt)
    if args.limit_sets:
        constructs = constructs[:args.limit_sets]
    have_ctl = {q.name[:-len("_CTL.jsonl.gz")]
                for q in (batt / "twin1").glob("*_CTL.jsonl.gz")}
    print(f"{args.reference}: {len(constructs)} A sets name positions, "
          f"{len(have_ctl & set(constructs))} of them with a control; "
          f"{len(skipped)} sets skipped as positionless", flush=True)

    t0 = time.time()
    model, _ = load_lm(args.reference, device, dtype)
    t_load = time.time() - t0
    print(f"{device} | {dtype} | loaded in {t_load:.0f}s\n", flush=True)

    res = {"reference": args.reference, "device": device, "dtype": str(dtype),
           "battery": str(batt), "batch": args.batch,
           "classes_for_gap": list(DEPENDENT), "sets": {}, "role_gap": {},
           # recorded, not discarded: an absent set must never look like a forgotten one
           "skipped_no_positions": skipped}
    t_fwd, ntok = time.time(), 0
    for i, c in enumerate(constructs, 1):
        a_rows = rows_of(batt / "twin1" / f"{c}_A.jsonl.gz")
        a_detail = losses_at_positions(model, tok, a_rows, device, args.batch)
        res["sets"][f"{c}|A"] = aggregate(a_detail)
        ntok += sum(r.get("n_tokens") or 0 for r in a_rows)
        if c in have_ctl:
            c_rows = rows_of(batt / "twin1" / f"{c}_CTL.jsonl.gz")
            c_detail = losses_at_positions(model, tok, c_rows, device, args.batch)
            res["sets"][f"{c}|CTL"] = aggregate(c_detail)
            res["role_gap"][c] = role_gap(a_detail, c_detail)
            ntok += sum(r.get("n_tokens") or 0 for r in c_rows)
        dt = time.time() - t_fwd
        g = res["role_gap"].get(c, {}).get("by_class_role", {})
        print(f"  [{i}/{len(constructs)}] {c:20} "
              f"{len(res['sets'][f'{c}|A']['by_class_role'])} class/role"
              + (f" | gap " + ", ".join(
                  f"{k.split('|')[0]} {v['mean']:+.3f}" for k, v in g.items()
                  if v["mean"] is not None) if g else "")
              + f" | {ntok/max(dt,1e-9)/1000:.1f}k tok/s", flush=True)
    res["seconds"] = {"load": round(t_load, 1), "forward": round(time.time() - t_fwd, 1)}
    res["tokens"] = ntok
    res["n_sets"] = len(res["sets"])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"\n{res['n_sets']} sets, {ntok:,} tokens in {res['seconds']['forward']/60:.1f} min "
          f"-> {out}", flush=True)
    # the rate only -- the total belongs to the driver, which knows what a full reference is.
    # Printing a hardcoded 8,366,284 here while `run_s16` printed the corrected 2,135,922 put
    # two different "a full reference is" figures in adjacent lines of the same output.
    print(f"RATE {ntok/max(res['seconds']['forward'],1e-9)/1000:.1f}k tok/s on {device} "
          f"over {len(constructs)} A sets and {len(res['role_gap'])} controls", flush=True)


if __name__ == "__main__":
    main()
