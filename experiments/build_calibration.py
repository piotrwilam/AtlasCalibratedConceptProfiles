"""Build the §2 calibration tables.

    python experiments/build_calibration.py [--limit N]
        -> $ATLAS_DATA/calibration/

Produces the tables the generator draws its menu from, the covariates later reports
read, the exposure expectation per checkpoint, and a manifest of content hashes. The
corpus is codeparrot-clean train shards; the validation shard stays held out for §3's
natural sample.
"""

from profilebattery.paths import data_root
import argparse
import collections
import hashlib
import json
import os
import pathlib
import time

import numpy as np

from profilebattery.calibration.corpus import CALIBRATION_SHARDS, fetch_shard, read
from profilebattery.calibration.tables import (DECLARED_NOUNS, DOMAIN_SEEDS,
                                               MIN_VOCABULARY, Tables,
                                               add_file, distinctive)
from profilebattery.calibration.walk import inventory_constructs

OUT = (data_root() / "calibration")
TOKENS_PER_STEP = 4_194_304
CHECKPOINTS = {"pretrain_iter_0001000": 1_000, "pretrain_iter_0002000": 2_000,
               "pretrain_iter_0003000": 3_000, "pretrain_iter_0373000": 373_000,
               "anneal_iter_0023000": 373_000 + 23_000}
TOP_IDENTIFIERS = 400


def quantiles(res, qs=(0.25, 0.5, 0.75)) -> dict:
    if not res or not res.items:
        return {}
    a = np.asarray(res.items, dtype=float)
    lo, med, hi = (float(np.quantile(a, q)) for q in qs)
    return {"q25": lo, "median": med, "q75": hi, "iqr_band": [lo, hi],
            "n_sampled": len(res.items), "n_seen": res.n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="files per shard (smoke tests)")
    ap.add_argument("--shards", type=int, nargs="*", default=list(CALIBRATION_SHARDS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    constructs = inventory_constructs()
    t = Tables()
    t0 = time.time()
    for n in args.shards:
        path = fetch_shard(n)
        for i, doc in enumerate(read(path, limit=args.limit)):
            add_file(t, doc.content, constructs)
            if t.files % 20000 == 0:
                print(f"  {t.files} files, {t.parsed} parsed, "
                      f"{time.time()-t0:.0f}s", flush=True)
    wall = time.time() - t0
    print(f"{t.files} files, {t.parsed} parsed, {t.skipped} unparseable "
          f"({t.skipped/max(t.files,1):.1%}), {t.oversize} oversize, {wall:.0f}s")

    # chars -> model tokens, measured rather than assumed
    ratio, span_tokens = _tokenise(t)

    total_tokens = t.chars / ratio if ratio else 0
    rates = {}
    for c in sorted(constructs):
        n = t.occurrences.get(c, 0)
        rates[c] = {
            "occurrences": n,
            "files_with": t.files_with.get(c, 0),
            "per_file": n / max(t.parsed, 1),
            "per_kchar": 1000 * n / max(t.chars, 1),
            "per_ktoken": 1000 * n / max(total_tokens, 1),
            "file_share": t.files_with.get(c, 0) / max(t.parsed, 1),
        }

    tables = {
        "construct_rates": rates,
        "scope": {c: dict(v) for c, v in t.scope.items()},
        "depth": {c: {str(k): n for k, n in sorted(v.items())}
                  for c, v in t.depth.items()},
        "companions": {c: dict(v.most_common(25)) for c, v in t.companions.items()},
        "core_axis_rates": {c: dict(v) for c, v in t.axis.items()},
        "length_bands": {c: {"lines": quantiles(t.span_lines.get(c)),
                             "chars": quantiles(t.span_chars.get(c)),
                             "tokens": span_tokens.get(c, {})}
                         for c in sorted(t.span_lines)},
        "identifiers": _vocabulary(t),
        "identifiers_raw_top": {d: dict(v.most_common(60))
                                for d, v in t.identifiers.items()},
        "literals": {"types": dict(t.literals),
                     "int_values": dict(t.int_values.most_common(120)),
                     "str_lengths": dict(sorted(t.str_lengths.items())[:200])},
        "padding_pool": t.padding.items,
        "exposure": _exposure(rates, total_tokens, t),
        "corpus": {"source": "codeparrot-clean", "shards": args.shards,
                   "limit_per_shard": args.limit,
                   "files": t.files, "parsed": t.parsed, "skipped": t.skipped,
                   "oversize_skipped": t.oversize,
                   "skip_rate": t.skipped / max(t.files, 1),
                   "chars": t.chars, "lines": t.lines,
                   "chars_per_token": ratio, "tokens_est": total_tokens,
                   "domain_files": dict(t.domain_files),
                   "domain_seeds": {k: list(v) for k, v in DOMAIN_SEEDS.items()},
                   "held_out": "codeparrot-clean-valid (natural sample, §3)",
                   "wall_seconds": round(wall, 1)},
    }

    manifest = {}
    for name, payload in tables.items():
        f = OUT / f"{name}.json"
        blob = json.dumps(payload, indent=2, sort_keys=True, default=str)
        f.write_text(blob)
        manifest[f.name] = hashlib.sha256(blob.encode()).hexdigest()
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"\nwrote {len(manifest)} tables to {OUT}")
    top = sorted(rates.items(), key=lambda kv: -kv[1]["occurrences"])[:12]
    print(f"{'construct':14} {'occurrences':>12} {'per file':>9} {'file share':>11}")
    for c, r in top:
        print(f"{c:14} {r['occurrences']:12,} {r['per_file']:9.2f} {r['file_share']:11.1%}")


def _vocabulary(t: Tables) -> dict:
    """Corpus-derived domain vocabulary, with a declared fallback where it is thin.

    Which source each domain used is part of the output, never a silent mix: the corpus
    gives real domain nouns only where the domain has enough files behind it.
    """
    corpus = distinctive(t.identifiers, TOP_IDENTIFIERS, t.identifier_df)
    out = {}
    for dom in DECLARED_NOUNS:
        found = corpus.get(dom, {})
        if len(found) >= MIN_VOCABULARY:
            out[dom] = {"source": "corpus", "n": len(found), "names": list(found),
                        "detail": found}
        else:
            out[dom] = {"source": "declared (corpus yield thin)",
                        "n_corpus": len(found), "names": list(DECLARED_NOUNS[dom]),
                        "corpus_found": list(found)}
    return out


def _tokenise(t: Tables):
    """chars-per-token, and the token-length band per construct, from real spans."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints",
        revision="pretrain_iter_0008000", trust_remote_code=True)
    sample = [s for c in t.spans for s in t.spans[c].items[:20]]
    if not sample:
        return None, {}
    chars = sum(len(s) for s in sample)
    toks = sum(len(tok(s, add_special_tokens=False)["input_ids"]) for s in sample)
    ratio = chars / max(toks, 1)
    out = {}
    for c, res in t.spans.items():
        if not res.items:
            continue
        lens = [len(tok(s, add_special_tokens=False)["input_ids"])
                for s in res.items[:200]]
        a = np.asarray(lens, dtype=float)
        out[c] = {"q25": float(np.quantile(a, 0.25)),
                  "median": float(np.quantile(a, 0.5)),
                  "q75": float(np.quantile(a, 0.75)),
                  "iqr_band": [float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))],
                  "n": len(lens)}
    return ratio, out


def _exposure(rates: dict, total_tokens: float, t: Tables) -> dict:
    """Expected occurrences per construct by checkpoint, with Poisson SE.

    An expectation, not a count: OpenCoder released the corpus and the tokens-per-step,
    never the data order, so no interval's actual content is knowable. Assumes a constant
    mixture within the pretraining stage — which the annealing stage breaks by design, so
    anything past the branch point needs the stage-two rates.
    """
    out = {"assumption": "constant mixture within the pretraining stage; no data order "
                         "is released, so exposure is an expectation",
           "tokens_per_step": TOKENS_PER_STEP,
           "anneal_branches_from": "pretrain_iter_0373000",
           "final_base_model": "anneal_iter_0023000",
           "warning": "checkpoints form a tree after step 373000: the pretrain branch "
                      "continues to 0484865 while annealing branches from 0373000",
           "by_checkpoint": {}}
    for ckpt, step in CHECKPOINTS.items():
        seen = step * TOKENS_PER_STEP
        row = {}
        for c, r in rates.items():
            exp = r["per_ktoken"] / 1000 * seen
            row[c] = {"expected": exp, "poisson_se": exp ** 0.5 if exp > 0 else 0.0}
        out["by_checkpoint"][ckpt] = {"tokens_seen": seen, "constructs": row}
    return out


if __name__ == "__main__":
    main()
