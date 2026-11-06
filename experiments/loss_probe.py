r"""The loss probe (TODO 43): is FLAT meaningful on this record at all?

    python experiments/loss_probe.py --reference pretrain_iter_0001000
    python experiments/loss_probe.py --reference untrained --limit 40

C2 reads whether a construct's behaviour loss is flat while its carrier set turns over. The
pilot stored no loss, so P1 -- the [min] prediction read first, whose answer fixes what every
date in the study means -- has never been rehearsed. This probe settles the stage-1 question of
C0.14: is the loss at the construct's positions already at its floor at 4.19B, in which case
FLAT is not a test on this record but an **M** row with cause *pre-record*, or is it still
falling, in which case FLAT is testable and the spans of C2 mean something.

It needs no position table, which does not exist yet (TODO 10). Instead of naming the obliged
and dependent positions it takes the loss at every **offset from the construct token**, which is
recorded for every prompt in the battery, and it records the commonest target piece at each
offset so that the write-up can say what an offset actually is. That last part is not decoration:
offset +1 after `for` is the loop *variable*, a generated identifier that is unpredictable by
construction and costs about nine nats, while the obliged `in` sits at +2 or later depending on
how many pieces the variable takes. So an absolute level at a fixed offset means little, and the
reading that answers the stage-1 question is the **change between references at the same
offset** -- which is valid whatever grammatical role the offset happens to hold.

The probe is also the first look at the distance profile the design owes (TODO 3), and the
A-minus-B difference at each offset is the first look at the role gap C2 now requires.

Per-core means are kept so that a noise scale can be had by halving a construct's cores, which
is the core bootstrap's own unit -- no second twin is needed for a stage-1 reading.

Writes validation/loss/PROBE_<reference>.json.
"""

from profilebattery.paths import data_root
import argparse
import collections
import json
import math
import os
import pathlib
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from profilebattery.record.store import read_battery
from profilebattery.record.sweep import REPO, UNTRAINED_SEED

#: **The phase battery by default.** Item 90's artefact
#: (`PROBE_pretrain_iter_0001000.json`, 60 of 60 sets) was made against the *pilot* battery's
#: 30 constructs, which is what that item asked for. Cowork's item 105 decision of 4 Oct is
#: that C2's FLAT needs all 188: the pilot's 30 hold only 13 constructs in P1, so a loss curve
#: on them cannot give "the share of P2 that is flat". Pass `--battery` to reproduce item 90.
BATT = pathlib.Path(os.path.expanduser(
    os.environ.get("PB_BATTERY", "$ATLAS_DATA/phase_battery")))
OUTDIR = (data_root() / "validation/loss")
TOK_REV = "pretrain_iter_0008000"
OFFSETS = list(range(0, 13))         # 0 = the construct token itself as a target
TIME_CHUNK = 16                      # positions per slice: the vocabulary is 96,640 wide, so a
                                     # whole-batch log-softmax in float32 is a gigabyte


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_lm(reference, device, dtype):
    # `untrained_seedN` is an untrained model with a different initialisation seed, not a
    # branch name. The recorder learned this the hard way (item 12 was unreachable without
    # it); the probe would otherwise hand `untrained_seed2` to HuggingFace as a revision.
    if reference == "untrained" or reference.startswith("untrained_seed"):
        suffix = reference[len("untrained_seed"):]
        if suffix and not suffix.isdigit():
            raise SystemExit(f"{reference!r}: expected `untrained_seedN` with N a number")
        seed = UNTRAINED_SEED + (int(suffix) - 1 if suffix else 0)
        cfg = AutoConfig.from_pretrained(REPO, revision=TOK_REV, trust_remote_code=True)
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        return model.to(device=device, dtype=dtype).eval(), cfg
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=reference, dtype=dtype,
                                                trust_remote_code=True)
    return model.to(device).eval(), model.config


@torch.no_grad()
def losses_for(model, tok, rows, device, batch, offsets):
    """Mean cross-entropy at each offset from the construct token, per core.

    Returns (per-core losses, per-core whole-prompt means, target-piece counts). Right padding,
    and every index is taken from the unpadded length, so a padded tail contributes nothing.
    The cross-entropy is accumulated in slices along time because the vocabulary is 96,640 wide:
    one float32 log-softmax over a whole batch is about a gigabyte, which on MPS turns a fast
    forward pass into a slow one -- measured, not assumed (482 tok/s falling to 126).
    """
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    whole = collections.defaultdict(list)
    pieces = collections.defaultdict(collections.Counter)
    by_piece = collections.defaultdict(lambda: [0.0, 0])   # (offset, piece) -> sum, n
    by_core_piece = collections.defaultdict(lambda: [0.0, 0])  # (core, piece) -> sum, n
    rows = [r for r in rows if r.get("site_construct_token") is not None]
    # Length-bucketed batches. The battery varies prompt length deliberately (the padding axis
    # of the cell design), so a batch drawn in file order pads to its longest member and most
    # of the forward pass is spent on pad tokens: measured at 270 tok/s in file order against
    # ~1,000 on a length-homogeneous slice. Order is irrelevant to every statistic here --
    # everything is aggregated by core and by target piece, never by position in the file.
    rows = sorted(rows, key=lambda r: r.get("n_tokens") or len(r["text"]))
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        enc = tok([r["text"] for r in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        T = ids.shape[1]
        # ce[:, i] is the loss of predicting ids[:, i + 1]
        ce = torch.empty(ids.shape[0], T - 1, dtype=torch.float32, device=device)
        for a in range(0, T - 1, TIME_CHUNK):
            b = min(a + TIME_CHUNK, T - 1)
            lp = torch.log_softmax(logits[:, a:b, :].float(), dim=-1)
            ce[:, a:b] = -lp.gather(-1, ids[:, a + 1:b + 1].unsqueeze(-1)).squeeze(-1)
        ce = ce.cpu().numpy()
        lens = mask.sum(1).cpu().numpy()
        idn = ids.cpu().numpy()
        for bi, r in enumerate(chunk):
            core = r.get("core_id", "?")
            n = int(lens[bi])
            site = int(r["site_construct_token"])
            whole[core].append(float(np.mean(ce[bi, :max(n - 1, 1)])))
            for off in offsets:
                j = site + off
                if 1 <= j < n:
                    v = float(ce[bi, j - 1])
                    piece = tok.convert_ids_to_tokens(int(idn[bi, j]))
                    per[core][off].append(v)
                    pieces[off][piece] += 1
                    # keyed by the target piece as well as the offset: a fixed offset mixes
                    # grammatical roles (offset 2 after `for` is `in` only when the loop
                    # variable is one piece), and conditioning on the piece recovers the
                    # obliged position without the position table that does not exist yet
                    e = by_piece[(off, piece)]
                    e[0] += v
                    e[1] += 1
                    e2 = by_core_piece[(core, piece)]
                    e2[0] += v
                    e2[1] += 1
    return per, whole, pieces, by_piece, by_core_piece


def summarise(per, whole, pieces, by_piece, by_core_piece, min_n=20):
    cores = sorted(per)
    out = {"n_cores": len(cores), "offsets": {}, "whole_prompt_mean": None}
    wm = [v for c in cores for v in whole[c]]
    out["whole_prompt_mean"] = float(np.mean(wm)) if wm else None
    for off in OFFSETS:
        vals = [v for c in cores for v in per[c].get(off, [])]
        if not vals:
            continue
        core_means = [float(np.mean(per[c][off])) for c in cores if per[c].get(off)]
        # a noise scale from halving the cores, which is the core bootstrap's own unit
        half = len(core_means) // 2
        gap = (abs(np.mean(core_means[:half]) - np.mean(core_means[half:]))
               if half else float("nan"))
        top = pieces[off].most_common(4)
        out["offsets"][str(off)] = {
            "top_targets": [[t, c] for t, c in top],
            "n": len(vals), "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "core_means": core_means,
            "core_sd": float(np.std(core_means, ddof=1)) if len(core_means) > 1 else None,
            "core_half_gap": float(gap),
        }
    agg = collections.defaultdict(lambda: [0.0, 0])
    for (off, piece), (tot, n) in by_piece.items():
        e = agg[piece]
        e[0] += tot
        e[1] += n
    out["by_target_piece"] = {p: {"mean": t / n, "n": n}
                              for p, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][1])
                              if n >= min_n}
    # per-core means for each target piece: the core bootstrap's own unit, applied to the
    # quantity C2 actually reads. A core-half gap taken over offsets is the wrong scale here.
    cores = collections.defaultdict(dict)
    for (core, piece), (tot, n) in by_core_piece.items():
        cores[piece][core] = tot / n
    for piece, m in out["by_target_piece"].items():
        cm = sorted(cores.get(piece, {}).items())
        vals = [v for _, v in cm]
        m["core_means"] = vals
        m["n_cores"] = len(vals)
        m["core_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        m["core_se"] = (float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                        if len(vals) > 1 else None)
    out["by_offset_and_piece"] = {f"{off}|{piece}": {"mean": t / n, "n": n}
                                  for (off, piece), (t, n) in by_piece.items() if n >= min_n}
    return out


def set_files(batt: pathlib.Path, kinds: str, constructs: str | None = None) -> list:
    """The battery files to read, by set kind.

    **The kinds are a parameter because the first pass hardcoded `[AB]`, and that is S16.** The
    loss pass of item 105 ran on the 376 A and B sets and silently never touched the 11
    matched-control (`_CTL`) sets, whose accumulators are in the record. Without their loss the
    role gap -- the control prompt's loss at the same position minus the A prompt's -- does not
    exist, and with it go the dependent sub-classes of curve 4, the dependent arm of P1 FLAT and
    C0.14's gate in P8. Nothing announced the omission: `LOSS.json` held 376 sets and read as
    complete, and the gap surfaced only when the analysis plan was acknowledged line by line.
    """
    want = [k.strip() for k in kinds.split(",") if k.strip()]
    files = sorted(f for k in want for f in (batt / "twin1").glob(f"*_{k}.jsonl.gz"))
    if constructs:
        keep = set(constructs.split(","))
        files = [f for f in files if f.name.rsplit("_", 1)[0] in keep]
    if not files:
        raise SystemExit(f"no sets of kind {want} in {batt / 'twin1'}; nothing to do")
    return files


def carry_over(out: pathlib.Path, limit, resume: bool) -> dict:
    """The sets already in `out` that this run must keep, or a refusal.

    **The file is rewritten whole after every set**, so starting with an empty `sets` destroys
    what is there. S16 adds 11 `_CTL` sets to a `LOSS.json` holding 376 A and B sets that cost
    about 22 minutes of A100 each and are not recomputed here; a mismatched `--limit`, or
    `--no-resume` passed out of habit, would have taken them out and said nothing, because the
    write is unconditional and nothing compared the two set lists.
    """
    if not out.exists():
        return {}
    prev = json.loads(out.read_text())
    had = prev.get("sets", {})
    if resume and prev.get("limit") == limit:
        return dict(had)
    if had:
        raise SystemExit(
            f"{out} already holds {len(had)} set(s) and this run would not carry them over: "
            f"its limit is {limit!r} against the file's {prev.get('limit')!r}"
            + ("" if resume else " and --no-resume was given")
            + ". The file is rewritten whole, so those sets would be lost. Match the limit "
              "and keep resume on, or write to a different --out.")
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="prompts per set, for timing")
    ap.add_argument("--constructs", default=None, help="comma list; default all")
    ap.add_argument("--kinds", default="A,B",
                    help="set kinds to read, comma-separated. `CTL` is S16: the 11 matched "
                         "controls the first loss pass never ran, which the role gap needs.")
    ap.add_argument("--battery", default=None,
                    help="battery directory; defaults to phase_battery (Cowork, item 105). "
                         "Pass $ATLAS_DATA/battery to reproduce item 90's artefact.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    global BATT
    if args.battery:
        BATT = pathlib.Path(os.path.expanduser(args.battery))
    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(REPO, revision=TOK_REV, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    t0 = time.time()
    model, cfg = load_lm(args.reference, device, dtype)
    print(f"{args.reference} | {device} | {dtype} | loaded in {time.time()-t0:.0f}s", flush=True)

    files = set_files(BATT, args.kinds, args.constructs)
    out = pathlib.Path(args.out) if args.out else OUTDIR / f"PROBE_{args.reference}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"reference": args.reference, "device": device, "dtype": str(dtype),
           "offsets": OFFSETS, "batch": args.batch, "limit": args.limit,
           "battery": str(BATT),
           "kinds": [k.strip() for k in args.kinds.split(",") if k.strip()],
           "n_sets": len(files), "sets": {}}
    res["sets"] = carry_over(out, args.limit, args.resume)
    if res["sets"]:
        print(f"resuming: {len(res['sets'])} sets already in {out.name}", flush=True)
    t0, ntok = time.time(), 0
    for i, f in enumerate(files, 1):
        construct, kind = f.stem.replace(".jsonl", "").rsplit("_", 1)
        rows = read_battery(f)
        if args.limit:
            rows = rows[: args.limit]
        key = f"{construct}|{kind}"
        if key in res["sets"]:
            print(f"  [{i}/{len(files)}] {key} already done", flush=True)
            continue
        per, whole, pieces, by_piece, bcp = losses_for(model, tok, rows, device, args.batch,
                                                       OFFSETS)
        entry = summarise(per, whole, pieces, by_piece, bcp)
        # **refuse to record a set with no usable position.** `order` keeps only rows with a
        # `site_construct_token`, and `_CTL` rows have none -- the header is replaced, so there
        # is no construct token -- so every offset came back empty and the set was written as
        # `offsets: {}` and counted as done. 26 references took that before anything noticed.
        if not (entry.get("offsets") or {}):
            raise SystemExit(
                f"{key}: no offset produced a loss. Every row in this set lacks "
                f"`site_construct_token`, which an offset-anchored read needs; "
                f"`_CTL` sets are like that by construction. Read them at the token indices in "
                f"each row's `positions` instead (S3's join). Nothing written for {key}.")
        res["sets"][key] = entry
        # **`n_sets` describes the file, not this run's slice.** Resume loads the sets already
        # there, so a `--kinds CTL` pass over a file holding 376 A and B sets would otherwise
        # record `n_sets: 11` and make a complete file look like a truncated one.
        res["n_sets"] = len(res["sets"])
        out.write_text(json.dumps(res, indent=1, default=float))   # after every set
        nt = sum(r.get("n_tokens", 79) for r in rows)
        ntok += nt
        el = time.time() - t0
        # **Name the piece the offset lands on.** The line used to read `off+1 mean 9.13`, and
        # on 5 Oct that was read as the loss at `Assign`'s obliged `=` -- implausible, and
        # rightly challenged. It is not: off+0 is `=` in 100% of `Assign` prompts and off+1 is
        # the *right-hand side*, an arbitrary expression, for which nine nats is expected. The
        # same holds for `For`, whose off+1 is the generated loop variable and whose obliged
        # `in` sits at off+2. A figure without its grammatical role is not a figure, so the
        # role goes on the line.
        # `top_targets` was always in the JSON -- it is only the progress line that lacked it
        e = res["sets"][f"{construct}|{kind}"]["offsets"].get("1", {})
        tt = e.get("top_targets") or []
        # escaped: target pieces contain newlines and tabs, which split the progress line
        piece = repr(tt[0][0])[1:-1] if tt else "?"
        print(f"  [{i}/{len(files)}] {construct}|{kind}  "
              f"off+1 ({piece}) mean {e.get('mean', float('nan')):.4f}"
              f"  | {ntok/el:.0f} tok/s, {el/60:.1f} min elapsed", flush=True)

    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {out}  ({ntok} tokens in {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
