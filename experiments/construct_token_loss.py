"""The construct-token loss: amendment A29's third loss class, read on all 188 constructs.

**What it reads.** `record/<ref>/LOSS.json`, field `sets["<construct>|A"]["offsets"]["0"]`. That
entry is the loss *at* the construct's own token: `losses_for` takes `j = site + off` and reads
`ce[bi, j-1]`, and `ce[:, i]` is the loss of predicting `ids[:, i+1]`, so at offset 0 the quantity
is the negative log probability the model assigns to the construct's token given the code before
it. It is built from `site_construct_token` alone, so it exists for all 188 A sets whether or not
the construct has a diagnostic position -- which is the point of A29: curve 4's hashed classes
reach 27 constructs, this reaches 188.

**Why no pooling rule was needed after all.** The qualification recorded in A29 was that 28 of the
188 have more than one piece at offset 0 and `by_offset_and_piece` carries no `0|_all`. That is
true of `by_offset_and_piece`, but `offsets["0"]` **is already the pooled reading** -- `mean`, `n`,
`median`, `core_means`, `core_sd` over all pieces at that offset -- so the pooled curve needs no
rule from anyone. The per-piece curves remain available from `by_offset_and_piece` and are
reported beside the pooled one for the 28, which is what A29's default said.

**The two branches are never one curve.** OpenCoder forks at step 373,000: the pretraining branch
runs to 484,865 and the annealing branch is 24 checkpoints off the fork. A single ordered curve
across both would read the fork as a jump in training. Each branch is dated on its own, and the
untrained models are the left end of neither -- they are reported as the untrained value.

**Dating (curve 4's rule, `PLAN-Analysis` part 3).** The competence date is the first checkpoint
at which the loss is within one noise unit of its minimum over the branch *and stays so*. A loss
already there at the first checkpoint is `competent_before_record`, a left-censored category.
A curve whose noise unit exceeds a fifth of its whole range is `unreadable`. The noise unit is the
core-bootstrap SE: 1,000 draws of eight cores with replacement from `core_means`, the SD of the
resampled means.
"""
from __future__ import annotations
from profilebattery.paths import data_root, dataset_repo

import argparse
import collections
import json
import os
import pathlib

import numpy as np

REPO = dataset_repo()
D = data_root()
OUT = D / "validation/loss_offset0"
DRAWS = 1_000
STEP_FACTOR = 1.5               # A30: the step is to the checkpoint nearest 1.5x later
SEED = 20261012


def branches(schedule: dict) -> dict[str, list[tuple[str, int]]]:
    """`{branch: [(reference, tokens)]}`, each branch ordered by tokens.

    The fork is why this exists rather than one sorted list: an annealing checkpoint at step
    1,000 is 1.5645T tokens into training, not 4.19B, so ordering by the number in the name
    interleaves the two branches and dates every construct at the fork.
    """
    tps = schedule["tokens_per_step"]
    fork = schedule["fork_step"]
    out: dict[str, list[tuple[str, int]]] = {"pretrain": [], "anneal": []}
    for s in schedule["pretrain_steps"]:
        out["pretrain"].append((f"pretrain_iter_{s:07d}", s * tps))
    for s in schedule["anneal_steps"]:
        out["anneal"].append((f"anneal_iter_{s:07d}", (fork + s) * tps))
    out["pretrain"].append((f"pretrain_iter_{484865:07d}", 484865 * tps))
    for k in out:
        seen, keep = set(), []
        for ref, t in sorted(out[k], key=lambda x: x[1]):
            if ref not in seen:
                seen.add(ref)
                keep.append((ref, t))
        out[k] = keep
    return out


def read_one(loss: dict) -> dict[str, dict]:
    """`{construct: {...}}` from one reference's `LOSS.json`, A sets only, offset 0 only.

    `pieces` keeps the per-piece means for the constructs whose construct token is more than one
    piece; `_all` is never among them because `by_offset_and_piece` has no such key at offset 0.
    """
    out = {}
    for key, entry in loss.get("sets", {}).items():
        if not key.endswith("|A"):
            continue
        o = (entry.get("offsets") or {}).get("0")
        if o is None:
            continue
        construct = key[:-2]
        bp = entry.get("by_offset_and_piece") or {}
        pieces = {k.split("|", 1)[1]: v for k, v in bp.items() if k.startswith("0|")}
        top = (o.get("top_targets") or [[None]])[0][0]
        out[construct] = {
            "mean": o["mean"], "n": o["n"],
            "core_means": list(o.get("core_means") or []),
            "core_sd": o.get("core_sd"),
            "pieces": pieces,
            "top_piece": top,
            "piece_class": piece_class(construct, top),
        }
    return out


def piece_class(construct: str, top_piece: str | None) -> str:
    """What offset 0 is actually scoring for this construct -- three different quantities.

    A29 says offset 0 is the construct token "by construction", and it is. But *which* token that
    is differs in kind across the 188, and the three kinds are not comparable as competence:

    * `keyword` -- the construct's token is a keyword or operator, not its class name: `AsyncFor`
      is scored on `for`, `AsyncFunctionDef` on `def`, `Assign` on `=`. This is the quantity the
      amendment describes, and it is the only one of the three that is.
    * `whole_name` -- the construct *is* its name and the name is one piece: `len`, `zip`. Offset
      0 scores the choice of this builtin among the alternatives the prompt admits.
    * `first_piece` -- the name takes several pieces and offset 0 scores only the **first**:
      `ArithmeticError` is scored on `Ar`, `BrokenPipeError` on `Broken`. The remaining pieces are
      nearly free once the first is out (`InterruptedError`: 15.21 nats at offset 0, 1.40 at
      offset 1), so this is a reading about sub-word prediction of a rare name.

    Recorded per construct so that no curve from one kind is pooled with or dated against another.
    """
    if not top_piece:
        return "unknown"
    piece = top_piece.lstrip("▁")
    name = construct.split("__")[0]
    if not piece:
        return "unknown"
    if piece.lower() == name.lower():
        return "whole_name"
    if name.lower().startswith(piece.lower()):
        return "first_piece"
    return "keyword"


def bootstrap_se(core_means, draws: int = DRAWS, seed: int = SEED) -> float | None:
    """The core-bootstrap SE: SD of `draws` means of eight cores drawn with replacement.

    `None` when there is fewer than one core to draw from, which is how a set with no usable
    core reaches the outcome `unreadable` instead of a spurious zero.
    """
    c = np.asarray([x for x in core_means if x is not None and np.isfinite(x)], dtype=float)
    if c.size < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c.size, size=(draws, c.size))
    return float(np.std(c[idx].mean(axis=1), ddof=1))


#: How far from `factor` a partner may sit and still be called a 1.5x step. Measured, not
#: chosen: on the 35 recorded pretraining references the realised factor has median exactly
#: 1.500 and 33 of 34 points fall in this band, the exception being `pretrain_iter_0455000`
#: at 1.066. On the **annealing** branch **not one of 23 points falls in it** -- that branch
#: spans x1.06 in tokens end to end, so no checkpoint is 1.5x later than any other and the
#: nearest-later search returns the FINAL checkpoint for every point. A curve built that way
#: rises to 1 by construction as `t` approaches the end, which is exactly what it did: curve 1
#: read +0.998 on the annealing branch with all 169 constructs dated, and none of it was a
#: measurement. Hence a band, and `None` rather than a substitute partner outside it.
FACTOR_BAND = (1.25, 2.0)


def partner_index(tokens, i: int, factor: float = STEP_FACTOR, band=None,
                  step: str = "factor") -> int | None:
    """Dispatch on A33's step rule: `"factor"` for the 1.5x step, `"consecutive"` for the next
    recorded reference.

    **A33 (13 Oct).** The annealing branch spans x1.06 in tokens end to end, so no checkpoint is
    1.5x later than any other and the factor rule has nothing to pair. Every curve on that branch
    -- curve 4 included -- is dated on **consecutive references**, and the 1.5x rule applies to
    the pretraining branch only.
    """
    if step == "consecutive":
        return i + 1 if i + 1 < len(tokens) else None
    if step != "factor":
        raise ValueError(f"unknown step rule {step!r}")
    return _partner_by_factor(tokens, i, factor, band)


def _partner_by_factor(tokens, i: int, factor: float = STEP_FACTOR, band=None) -> int | None:
    """The recorded checkpoint nearest `factor` x later in tokens, or `None` for the tail.

    A30's step is "the change to the checkpoint nearest 1.5x later in tokens", so the partner is
    chosen by proximity to `1.5 * tokens[i]` among the checkpoints that are **later**, never
    earlier -- a nearest-in-absolute-distance search would pair the last few checkpoints
    backwards and invent steps that run against training.

    `band=(lo, hi)` refuses a partner whose realised factor falls outside it, returning `None`
    so the point is dropped rather than compared against whatever happens to be last. Left
    `None` by default so that A30's already-reported loss labels do not move under this module;
    curve 1 passes `FACTOR_BAND`. Whether curve 4's rule should take the band too is an
    amendment question -- `partner_factors` below is the diagnostic for it.
    """
    later = [j for j in range(len(tokens)) if tokens[j] > tokens[i]]
    if not later:
        return None
    want = tokens[i] * factor
    j = min(later, key=lambda j: abs(tokens[j] - want))
    if band is not None:
        f = tokens[j] / tokens[i]
        if not (band[0] <= f <= band[1]):
            return None
    return j


def partner_factors(tokens, factor: float = STEP_FACTOR) -> dict:
    """The realised partner factor at every point: the diagnostic for whether a branch has steps.

    Reported per branch so that a rule reading "the checkpoint nearest 1.5x later" is never
    applied to a branch that has no such checkpoint without that being visible.
    """
    import numpy as np
    out = []
    for i in range(len(tokens)):
        j = partner_index(tokens, i)
        if j is not None:
            out.append(tokens[j] / tokens[i])
    if not out:
        return {"n": 0}
    a = np.asarray(out, float)
    lo, hi = FACTOR_BAND
    return {"n": int(a.size), "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "in_band": int(((a >= lo) & (a <= hi)).sum()),
            "band": list(FACTOR_BAND)}


def at_or_above_untrained(level: float, unit: float, untrained: float | None) -> bool | None:
    """A30's level test, shared by flat and dated curves (A30 final form, addition 2).

    "At or above" carries a **one-unit tolerance** -- the resolution the rest of the rule uses --
    so a level within one noise unit of the untrained value counts as at it. `RecursionError`
    at 11.65 against 11.89 untrained is the case that fixed this: a 0.24-nat gap is not evidence
    of having learned anything. `None` when there is no untrained reading to compare against.
    """
    if untrained is None:
        return None
    return level >= untrained - unit


def flat_label(level: float, unit: float, untrained: float | None, forced: bool) -> dict:
    """A30's three sub-labels for a flat curve, by its level against the untrained reference.

    *"Competent before the record" was a floor claim, and flatness alone does not make it*
    (A30, ruling 3, 13 Oct). A flat curve is one of three different things:

    * `flat_at_floor` -- forced positions only, the level within one unit of zero.
    * `learned_before_record` -- below the untrained reference's loss at the same position: the
      fall happened before the first checkpoint and the floor is unknown.
    * `never_learned` -- at or above the untrained level: `BrokenPipeError` sits flat at 13.9
      nats against 11.5 untrained, which is not competence of any kind.

    A30 gives no tolerance on "at or above", and one noise unit is the resolution the rest of the
    rule uses, so **a level within one unit of the untrained value counts as "at" it** -- CC's
    reading, stated because it decides the boundary cases.
    """
    if forced and abs(level) <= unit:
        return {"flat": "flat_at_floor", "level": level}
    above = at_or_above_untrained(level, unit, untrained)
    if above is None:
        return {"flat": "flat_level_unknown", "level": level,
                "why": "no untrained reference for this position"}
    if above:
        return {"flat": "never_learned", "level": level, "untrained": untrained}
    return {"flat": "learned_before_record", "level": level, "untrained": untrained}


def label_curve(values, units, tokens, untrained=None, forced: bool = False,
                unit_stat: str = "median", step: str = "factor") -> dict:
    """A30's dating rule, one rule for every curve (`TODO-Settle` §5, approved 13 Oct).

    `R` is the range in noise units, the unit being the **median** of the per-checkpoint units
    over the curve (A30 ruling 1; `unit_stat="max"` gives the sensitivity that is reported
    beside it). Then:

    * `R <= 1` -- flat. The outcome is one of `flat_label`'s three sub-labels, never a bare
      "competent before the record".
    * `1 < R < 5` -- `unreadable`: the curve moves, by too little to date.
    * `R >= 5` -- datable. The date is the first checkpoint after which **every later 1.5x step**
      is within one noise unit. `censored` if the last step still exceeds one unit, or if the
      extreme is at the last checkpoint. A missing 1.5x partner imposes no constraint (A30 ruling
      2), and a date falling inside the last 1.5x window carries `at_edge: true`, since no later
      step can censor it.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2 or not np.all(np.isfinite(v)):
        return {"outcome": "unreadable", "why": "a checkpoint has no finite loss"}
    u = np.asarray([np.nan if x is None else x for x in units], dtype=float)
    if not np.any(np.isfinite(u)):
        return {"outcome": "unreadable", "why": "no checkpoint has a noise unit"}
    fin = u[np.isfinite(u)]
    unit = float(np.median(fin) if unit_stat == "median" else np.max(fin))
    rng_v = float(v.max() - v.min())
    if unit <= 0:
        return {"outcome": "unreadable", "why": "the noise unit is zero", "range": rng_v}
    R = rng_v / unit

    if R <= 1:
        out = {"outcome": "flat", "R": R, "noise_unit": unit, "range": rng_v}
        out.update(flat_label(float(v[0]), unit, untrained, forced))
        return out
    if R < 5:
        return {"outcome": "unreadable", "why": "the range is between one and five noise units",
                "R": R, "noise_unit": unit, "range": rng_v}

    steps = {i: (None if (j := partner_index(tokens, i, step=step)) is None
                 else abs(float(v[j] - v[i])))
             for i in range(v.size)}
    withpartner = [i for i in steps if steps[i] is not None]
    if withpartner:
        last = max(withpartner)
        if steps[last] > unit:
            name = "consecutive" if step == "consecutive" else "1.5x"
            return {"outcome": "censored",
                    "why": f"the last {name} step still exceeds one unit",
                    "R": R, "noise_unit": unit, "last_step": steps[last]}
    extreme = int(np.argmin(v)) if abs(v.min() - v[0]) > abs(v.max() - v[0]) else int(np.argmax(v))
    if extreme == v.size - 1:
        return {"outcome": "censored", "why": "the extreme is at the last checkpoint",
                "R": R, "noise_unit": unit}
    first = None
    for i in range(v.size):
        if all(steps[k] is None or steps[k] <= unit for k in range(i, v.size)):
            first = i
            break
    if first is None:
        return {"outcome": "censored", "why": "no checkpoint after which the steps settle",
                "R": R, "noise_unit": unit}
    out = {"outcome": "dated", "index": first, "R": R, "noise_unit": unit,
           "level": float(v[first]), "at_edge": steps[first] is None}

    # **A30 final form, addition 1: the level test applies to a dated curve as to a flat one.**
    # The settled level is the mean over the checkpoints AFTER the date -- not the value at the
    # date, which is the last point of the fall and sits above where the curve comes to rest.
    # A date whose settled level is at or above untrained still stands for the agreement question
    # in part 4, because the curve did stop moving; it is not a competence date and carries
    # `competence: false` so it cannot enter a competence claim by accident.
    after = v[first + 1:]
    if after.size:
        settled = float(after.mean())
        out["settled_from"] = "mean of the checkpoints after the date"
    else:
        settled = float(v[first])
        out["settled_from"] = "the value at the date; no checkpoint follows it"
    out["settled_level"] = settled
    above = at_or_above_untrained(settled, unit, untrained)
    if above is None:
        out["competence"] = None
        out["label"] = "dated_level_unknown"
    elif above:
        out["competence"] = False
        out["label"] = "dated_never_learned"
        out["untrained"] = untrained
    else:
        out["competence"] = True
        out["label"] = "dated"
        out["untrained"] = untrained
    return out


def trend(values, units, tokens) -> dict:
    """Descriptive trend, so that *flat and noisy* is distinguishable from *still falling*.

    **Why this is here and is not a second dating rule.** Curve 4's rule asks whether the loss
    stays within one noise unit of its *minimum over the branch*, and the minimum of a noisy
    sample is biased low by roughly the noise: over 35 checkpoints a flat curve's sample minimum
    sits about one unit below its level, so no flat-but-noisy curve can "stay within one unit of
    its minimum" and every one of them reads as `censored` -- as though still falling. These four
    numbers say which it is, and the rule's own outcome is left exactly as the rule computes it.

    * `first_above_floor_units` -- how far the first checkpoint is above the branch minimum, in
      noise units. Near zero means the floor was already reached at the first checkpoint.
    * `last_minus_first` and `last_minus_first_units` -- signed, so a falling curve is negative.
    * `spearman` -- rank correlation of the loss with token count; near zero is no trend.
    """
    v = np.asarray(values, dtype=float)
    u = np.asarray([np.nan if x is None else x for x in units], dtype=float)
    t = np.asarray(tokens, dtype=float)
    if v.size < 2 or not np.all(np.isfinite(v)):
        return {}
    unit = float(np.nanmedian(u)) if np.any(np.isfinite(u)) else float("nan")
    out = {"first_above_floor": float(v[0] - v.min()),
           "last_minus_first": float(v[-1] - v[0]),
           "median_noise_unit": unit}
    if np.isfinite(unit) and unit > 0:
        out["first_above_floor_units"] = out["first_above_floor"] / unit
        out["last_minus_first_units"] = out["last_minus_first"] / unit
    rv = np.argsort(np.argsort(v)).astype(float)
    rt = np.argsort(np.argsort(t)).astype(float)
    sd = rv.std() * rt.std()
    out["spearman"] = float(np.mean((rv - rv.mean()) * (rt - rt.mean())) / sd) if sd else 0.0
    return out


def build(per_ref: dict[str, dict], schedule: dict) -> dict:
    """The whole reading: one curve per construct per branch, dated, plus coverage and pieces."""
    bs = branches(schedule)
    constructs = sorted({c for r in per_ref.values() for c in r})
    out = {"n_constructs": len(constructs), "draws": DRAWS, "seed": SEED,
           "branches": {k: [r for r, _ in v] for k, v in bs.items()},
           "tokens": {k: [t for _, t in v] for k, v in bs.items()},
           "untrained": {}, "multi_piece": {}, "curves": {},
           "piece_class_note": "keyword | whole_name | first_piece -- see piece_class()"}

    for ref in ("untrained", "untrained_seed2"):
        if ref in per_ref:
            out["untrained"][ref] = {c: per_ref[ref][c]["mean"] for c in per_ref[ref]}

    for c in constructs:
        kinds = {per_ref[r][c]["piece_class"] for r in per_ref if c in per_ref[r]}
        entry = {"branch": {},
                 "piece_class": kinds.pop() if len(kinds) == 1 else sorted(kinds)}
        npieces = max((len(per_ref[r][c]["pieces"]) for r in per_ref if c in per_ref[r]),
                      default=0)
        if npieces > 1:
            out["multi_piece"][c] = npieces
        for b, seq in bs.items():
            refs = [r for r, _ in seq if r in per_ref and c in per_ref[r]]
            if not refs:
                continue
            vals = [per_ref[r][c]["mean"] for r in refs]
            ns = [per_ref[r][c]["n"] for r in refs]
            units = [bootstrap_se(per_ref[r][c]["core_means"]) for r in refs]
            toks = [dict(seq)[r] for r in refs]
            tr = trend(vals, units, toks)
            untr = None
            for ref in ("untrained", "untrained_seed2"):
                if ref in per_ref and c in per_ref[ref]:
                    untr = per_ref[ref][c]["mean"]
                    tr["fall_before_record"] = untr - vals[0]
                    break
            step = "consecutive" if b == "anneal" else "factor"      # A33
            d = label_curve(vals, units, toks, untrained=untr, step=step)
            alt = label_curve(vals, units, toks, untrained=untr, unit_stat="max", step=step)
            d["step_rule"] = step
            d["sensitivity_max_unit"] = {k: alt.get(k)
                                     for k in ("outcome", "flat", "label", "R", "index")}
            if "index" in d:
                i = d["index"]
                d["reference"] = refs[i]
                d["tokens"] = dict(seq)[refs[i]]
            entry["branch"][b] = {
                "n_checkpoints": len(refs), "references": refs,
                "mean": vals, "n": ns, "noise_unit": units,
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "date": d, "trend": tr,
            }
        out["curves"][c] = entry
    return out


def summarise(reading: dict, branch: str = "pretrain", sensitivity: bool = False) -> dict:
    """Counts by label on one branch -- a flat curve counts under its sub-label, not as `flat`."""
    by = collections.Counter()
    for c, e in reading["curves"].items():
        b = e["branch"].get(branch)
        if not b:
            by["absent"] += 1
            continue
        d = b["date"]["sensitivity_max_unit"] if sensitivity else b["date"]
        by[d.get("flat") or d.get("label") or d.get("outcome")] += 1
    return dict(by)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", type=int, default=0, help="first N references, for a smoke run")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from experiments.run_s15 import references

    token = os.environ.get("HF_TOKEN")
    refs = references(None)
    if a.only:
        refs = refs[:a.only]
    schedule = json.load(open(hf_hub_download(
        REPO, "validation/schedule/CHECKPOINTS.json", repo_type="dataset", token=token)))

    per_ref = {}
    for i, r in enumerate(refs, 1):
        p = hf_hub_download(REPO, f"record/{r}/LOSS.json", repo_type="dataset", token=token)
        per_ref[r] = read_one(json.load(open(p)))
        print(f"  [{i:2d}/{len(refs)}] {r:26s} {len(per_ref[r]):3d} A sets at offset 0",
              flush=True)

    sizes = {len(v) for v in per_ref.values()}
    if sizes != {188}:
        raise SystemExit(f"expected 188 A sets at offset 0 at every reference, saw {sizes}")

    reading = build(per_ref, schedule)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CONSTRUCT_TOKEN_LOSS.json").write_text(json.dumps(reading, indent=1))

    for b in ("pretrain", "anneal"):
        n = {e["branch"][b]["n_checkpoints"] for e in reading["curves"].values()
             if b in e["branch"]}
        print(f"\n{b}: {sorted(n)} checkpoints recorded per construct")
        print(f"  A30 labels (median unit) {summarise(reading, b)}")
        print(f"  sensitivity  (max unit)  {summarise(reading, b, sensitivity=True)}")
        edge = sum(1 for e in reading["curves"].values()
                   if b in e["branch"] and e["branch"][b]["date"].get("at_edge"))
        print(f"  dated at the edge: {edge}")
        tr = [e["branch"][b].get("trend", {}) for e in reading["curves"].values()
              if b in e["branch"]]
        fl = [x["first_above_floor_units"] for x in tr if "first_above_floor_units" in x]
        sp = [x["spearman"] for x in tr if "spearman" in x]
        fb = [x["fall_before_record"] for x in tr if "fall_before_record" in x]
        if fl:
            print(f"  first checkpoint above the floor, in noise units: "
                  f"median {np.median(fl):.1f}, within 1 unit for "
                  f"{sum(x <= 1 for x in fl)} of {len(fl)}")
        if sp:
            print(f"  spearman of loss against tokens: median {np.median(sp):+.2f}, "
                  f"negative for {sum(x < 0 for x in sp)} of {len(sp)}")
        if fb:
            print(f"  fall from untrained to the first checkpoint: "
                  f"median {np.median(fb):.2f} nats")
    print(f"\nconstructs whose construct token is more than one piece: "
          f"{len(reading['multi_piece'])}")
    kinds = collections.Counter(
        e["piece_class"] if isinstance(e["piece_class"], str) else "mixed"
        for e in reading["curves"].values())
    print("what offset 0 scores, by construct:", dict(kinds))
    print(f"\n-> {out / 'CONSTRUCT_TOKEN_LOSS.json'}")


if __name__ == "__main__":
    main()
