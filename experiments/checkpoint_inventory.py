"""Enumerate every OpenCoder checkpoint and verify it is actually usable.

The larger open risk: phase 1 dates formation events, and a date can only be resolved as
finely as the checkpoints allow. This asks three things of all 510 branches — does the
branch carry weights, where does it sit on the token axis, and which of the two
post-fork branches is it on — so that what phase 1 may promise is a measured fact rather
than an assumption about a release.

    python experiments/checkpoint_inventory.py -> calibration/checkpoints.json
"""

import concurrent.futures as cf
import json
import os
import pathlib
import time
import urllib.request
from profilebattery.paths import data_root

REPO = "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints"
TOKENS_PER_STEP = 4_194_304
FORK_AT = 373_000
OUT = data_root() / "calibration"
WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")


def get(url: str, tries: int = 5):
    """Exponential backoff. The endpoint rate-limits unauthenticated callers hard enough
    that three quick tries is not enough — an earlier run silently produced an inventory
    of zero branches because the refs call itself was throttled."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(2 ** i)


def branches() -> list[str]:
    d = get(f"https://huggingface.co/api/models/{REPO}/refs") or {}
    return [b["name"] for b in d.get("branches", []) if b["name"] != "main"]


def probe(name: str) -> dict:
    d = get(f"https://huggingface.co/api/models/{REPO}/revision/{name}?blobs=true")
    if d is None:
        return {"branch": name, "reachable": False}
    sizes = {s["rfilename"]: s.get("size") or 0 for s in d.get("siblings", [])}
    weights = {f: sizes[f] for f in WEIGHT_FILES if f in sizes}
    stage, _, it = name.partition("_iter_")
    step = int(it) if it.isdigit() else None
    return {"branch": name, "reachable": True, "stage": stage, "step": step,
            "has_weights": bool(weights),
            "weight_bytes": max(weights.values()) if weights else 0,
            "files": len(sizes),
            "has_config": "config.json" in sizes,
            "has_tokenizer": any("tokeniz" in f for f in sizes)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = branches()
    if len(names) < 400:
        raise SystemExit(f"refs call returned only {len(names)} branches — throttled; "
                         f"not writing a partial inventory")
    print(f"{len(names)} branches; probing…", flush=True)
    # Serial, with a pause. A concurrent probe against the unauthenticated endpoint
    # returns partial file lists under rate limiting, which reads as "this checkpoint has
    # no weights": the first run reported ten such branches and every one of them had the
    # full 3,811 MB file when re-probed by hand. A concurrent pass plus retries did not
    # fix it — the rate limit applies to the retry too — so the probe is simply slow, and
    # any branch that still looks empty is re-probed once more before it is believed.
    rows = []
    for i, name in enumerate(names, 1):
        rows.append(probe(name))
        if i % 100 == 0:
            print(f"  {i}/{len(names)}", flush=True)
        time.sleep(0.25)

    suspect = [r for r in rows if r.get("reachable") and not r.get("has_weights")]
    if suspect:
        print(f"re-probing {len(suspect)} apparent gaps…", flush=True)
        by_name = {r["branch"]: i for i, r in enumerate(rows)}
        for r in suspect:
            time.sleep(1.0)
            rows[by_name[r["branch"]]] = probe(r["branch"])

    ok = [r for r in rows if r.get("has_weights")]
    bad = [r for r in rows if not r.get("has_weights")]
    pre = sorted((r for r in ok if r["stage"] == "pretrain"), key=lambda r: r["step"])
    ann = sorted((r for r in ok if r["stage"] == "anneal"), key=lambda r: r["step"])

    def axis(rows_, offset=0):
        return [{"branch": r["branch"], "step": r["step"],
                 "tokens": (r["step"] + offset) * TOKENS_PER_STEP} for r in rows_]

    gaps = []
    for seq in (pre, ann):
        for a, b in zip(seq, seq[1:]):
            d = b["step"] - a["step"]
            if d > 1000:
                gaps.append({"from": a["branch"], "to": b["branch"], "steps": d})

    early = [r for r in pre if r["step"] <= 20_000]
    rec = {
        "repo": REPO, "tokens_per_step": TOKENS_PER_STEP,
        "branches": len(names), "with_weights": len(ok), "without_weights":
            [r["branch"] for r in bad],
        "unreachable": [r["branch"] for r in rows if not r.get("reachable")],
        "pretrain": {"n": len(pre), "first": pre[0]["branch"] if pre else None,
                     "last": pre[-1]["branch"] if pre else None,
                     "tokens_first": pre[0]["step"] * TOKENS_PER_STEP if pre else None,
                     "tokens_last": pre[-1]["step"] * TOKENS_PER_STEP if pre else None},
        "anneal": {"n": len(ann), "first": ann[0]["branch"] if ann else None,
                   "last": ann[-1]["branch"] if ann else None,
                   "branches_from_step": FORK_AT,
                   "tokens_last": (FORK_AT + ann[-1]["step"]) * TOKENS_PER_STEP
                   if ann else None},
        "fork": {"at_step": FORK_AT,
                 "pretrain_continues_past_fork":
                     sum(1 for r in pre if r["step"] > FORK_AT),
                 "note": "the two post-fork sequences are separate trajectories and "
                         "do not share a token axis"},
        "gaps_over_1000_steps": gaps,
        "earliest_10": axis(pre[:10]),
        "density_below_20B_tokens": len(early),
        "weight_bytes_distinct": sorted({r["weight_bytes"] for r in ok}),
        "all": rows,
    }
    (OUT / "checkpoints.json").write_text(json.dumps(rec, indent=2))
    print(f"\nwith weights: {len(ok)}/{len(names)}   without: {len(bad)}")
    print(f"pretrain {len(pre)}: {rec['pretrain']['first']} .. {rec['pretrain']['last']}")
    print(f"anneal   {len(ann)}: {rec['anneal']['first']} .. {rec['anneal']['last']}")
    print(f"pretrain branches past the fork: {rec['fork']['pretrain_continues_past_fork']}")
    print(f"gaps over 1000 steps: {len(gaps)}")
    print(f"checkpoints at or below 20k steps (~84B tokens): {len(early)}")
    print("earliest ten:")
    for r in rec["earliest_10"]:
        print(f"  {r['branch']:26} {r['tokens']/1e9:8.2f}B tokens")
    if bad:
        print(f"\nbranches WITHOUT weights: {[r['branch'] for r in bad][:20]}")


if __name__ == "__main__":
    main()
