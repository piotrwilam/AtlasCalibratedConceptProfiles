"""The natural sample — handout §3's last set.

Mined from `codeparrot-clean-valid`, the one corpus file held out from calibration so the
sample the generator is compared against has not shaped it.

Natural **A** is every inventory occurrence, windowed. Natural **B** is the construct's token
appearing *outside* every node of that construct's type — in a comment, a string, an
identifier or a sibling construct — which is the same "token present, construct absent"
definition the synthetic B uses, applied to real code instead of a generated carrier.

    python experiments/build_natural.py [--per-construct 100]
        -> $ATLAS_DATA/battery/natural/
"""

import argparse
import ast
import collections
import gzip
import json
import os
import pathlib
import time
import warnings

from transformers import AutoTokenizer

from profilebattery.calibration.corpus import fetch_valid, read
from profilebattery.generate.natural import (line_start_offsets, node_span, occurrences,
                                             parse_file, sites_of, text_hash, window_for)
from profilebattery.inventory.tokens import NODE_TOKENS, tokens_of
from profilebattery.paths import data_root

OUT = data_root() / "battery/natural"
BATT = data_root() / "battery"
CAL = data_root() / "calibration"
REPO = "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints"
REVISION = "pretrain_iter_0008000"


def synthetic_hashes() -> set[str]:
    """Texts in the *synthetic* sets, so the natural sample can exclude them.

    `natural/` lives under `battery/`, so a recursive glob picked up this script's own
    previous output and the second run de-duplicated against itself — `NamedExpr` went from
    12 accepted windows to 0, and 4,077 extra windows were dropped. Only the twin and
    alternative-menu directories are synthetic.
    """
    out = set()
    for sub in sorted(BATT.glob("twin*")) + [BATT / "alt"]:
        for f in sub.glob("*_A.jsonl.gz"):
            for line in gzip.open(f, "rt"):
                out.add(text_hash(json.loads(line)["text"]))
    return out


def natural_b(source: str, construct: str, token: str, bands, cpt) -> list[dict]:
    """Token occurrences outside every node of the construct's own type."""
    tree = parse_file(source)
    if tree is None:
        return []
    starts = line_start_offsets(source)
    cls = getattr(ast, construct, None)
    spans = [node_span(n, starts) for n in ast.walk(tree)
             if cls is not None and isinstance(n, cls)]
    out, pos = [], source.find(token)
    band = bands.get(construct, {}).get("tokens", {})
    target = int(band.get("median") or 60) + 40
    while pos != -1 and len(out) < 40:
        if not any(lo <= pos < hi for lo, hi in spans):
            lo, hi = window_for(source, pos, target, cpt)
            text = source[lo:hi]
            rel = pos - lo
            w = line_start_offsets(text)
            ln = max(i for i, o in enumerate(w, start=1) if o <= rel)
            out.append({"text": text, "anchor_lineno": ln,
                        "anchor_col": rel - w[ln - 1], "file_offset": pos,
                        "window_start": lo, "window_end": hi})
        pos = source.find(token, pos + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-construct", type=int, default=100)
    ap.add_argument("--limit-files", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    bands = json.loads((CAL / "length_bands.json").read_text())
    cpt = json.loads((CAL / "corpus.json").read_text())["chars_per_token"]
    manifest = json.loads((BATT / "MANIFEST.json").read_text())
    subset = manifest["subset"]
    seen = synthetic_hashes()
    n_synthetic = len(seen)
    dup_vs_synthetic = 0
    print(f"{len(seen)} synthetic texts to de-duplicate against")
    print(f"mining {len(subset)} constructs from the held-out split\n")

    want = {c: args.per_construct for c in subset}
    got_a: dict[str, list] = collections.defaultdict(list)
    got_b: dict[str, list] = collections.defaultdict(list)
    files = parsed = skipped = dup = 0
    t0 = time.time()
    for doc in read(fetch_valid(), limit=args.limit_files):
        files += 1
        if all(len(got_a[c]) >= want[c] and len(got_b[c]) >= want[c] for c in subset):
            break
        if parse_file(doc.content) is None:
            skipped += 1
            continue
        parsed += 1
        need = {c for c in subset if len(got_a[c]) < want[c]}
        if need:
            for occ in occurrences(doc.content, doc.repo, doc.path, need, bands, cpt,
                                   tokens_of):
                if len(got_a[occ.construct]) >= want[occ.construct]:
                    continue
                h = text_hash(occ.text)
                if h in seen:
                    dup += 1
                    continue
                seen.add(h)
                row = {"construct": occ.construct, "set_kind": "A", "source": "natural",
                       "repo": occ.repo, "path": occ.path,
                       "file_offset": occ.file_offset, "window_start": occ.window_start,
                       "window_end": occ.window_end, "text": occ.text,
                       "surface_token": occ.surface_token, "text_hash": h}
                row.update(sites_of(occ, tok))
                got_a[occ.construct].append(row)
        for c in subset:
            if len(got_b[c]) >= want[c]:
                continue
            for b in natural_b(doc.content, c, NODE_TOKENS[c], bands, cpt):
                if len(got_b[c]) >= want[c]:
                    break
                h = text_hash(b["text"])
                if h in seen:
                    dup += 1
                    continue
                seen.add(h)
                from profilebattery.generate.natural import Occurrence
                occ = Occurrence(construct=c, repo=doc.repo, path=doc.path,
                                 file_offset=b["file_offset"],
                                 window_start=b["window_start"],
                                 window_end=b["window_end"], text=b["text"],
                                 anchor_lineno=b["anchor_lineno"],
                                 anchor_col=b["anchor_col"], span_end_lineno=None,
                                 surface_token=NODE_TOKENS[c])
                row = {"construct": c, "set_kind": "B", "source": "natural",
                       "repo": doc.repo, "path": doc.path, "text": b["text"],
                       "file_offset": b["file_offset"], "text_hash": h,
                       "surface_token": NODE_TOKENS[c]}
                row.update(sites_of(occ, tok))
                got_b[c].append(row)
        if files % 5000 == 0:
            short = sum(1 for c in subset if len(got_a[c]) < want[c])
            print(f"  {files} files, {parsed} parsed, {short} constructs still short, "
                  f"{time.time()-t0:.0f}s", flush=True)

    summary = {}
    for kind, store in (("A", got_a), ("B", got_b)):
        for c, rows in store.items():
            p = OUT / f"{c}_{kind}.jsonl.gz"
            with gzip.open(p, "wt") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            summary[f"{c}/{kind}"] = len(rows)
    rec = {"source": "codeparrot-clean-valid (held out from calibration)",
           "files_seen": files, "files_parsed": parsed, "unparseable": skipped,
           "skip_rate": round(skipped / max(files, 1), 4),
           # `dup` covers both kinds, because `seen` grows as the mine proceeds: a window
           # already taken from another held-out file counts too, and corpus boilerplate
           # repeats a great deal. Labelling it "against synthetic" would overstate the
           # overlap between the two sets.
           "duplicates_dropped_total": dup,
           "synthetic_hashes_preloaded": n_synthetic,
           "per_construct_target": args.per_construct, "counts": summary,
           "wall_seconds": round(time.time() - t0, 1)}
    (OUT / "MANIFEST.json").write_text(json.dumps(rec, indent=2))
    short_a = {c: len(got_a[c]) for c in subset if len(got_a[c]) < args.per_construct}
    short_b = {c: len(got_b[c]) for c in subset if len(got_b[c]) < args.per_construct}
    print(f"\n{files} files, {parsed} parsed, {skipped} unparseable "
          f"({skipped/max(files,1):.1%}), {dup} duplicates dropped, "
          f"{time.time()-t0:.0f}s")
    print(f"natural A: {sum(len(v) for v in got_a.values()):,}  "
          f"natural B: {sum(len(v) for v in got_b.values()):,}")
    print(f"short of target — A: {short_a or 'none'}")
    print(f"short of target — B: {short_b or 'none'}")


if __name__ == "__main__":
    main()
