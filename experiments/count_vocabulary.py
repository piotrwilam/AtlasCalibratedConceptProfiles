"""Per-domain vocabulary the §2 walk did not collect: verbs, attributes, class names.

`identifiers.json` holds `Name` ids — the nouns. A core also needs *verbs* (function
names, for delegate bodies and every host), *attribute* names, and *class* names, and
those live in `FunctionDef.name`, `Attribute.attr` and `ClassDef.name`. Without them the
calibrated menu would still be falling back on declared words for three of its four
slots.

Same shape as the builtin pass: parse, walk for three node types, classify the file by
domain, filter by distinctiveness and document frequency.

    python experiments/count_vocabulary.py -> calibration/vocabulary.json
"""

import argparse
import ast
import collections
import json
import os
import pathlib
import time
import warnings

from profilebattery.calibration.corpus import fetch_shard, read
from profilebattery.calibration.tables import classify_domain, DOMAIN_SEEDS
from profilebattery.paths import data_root

OUT = data_root() / "calibration"
MIN_FILES = 8
MIN_COUNT = 15
RATIO = 3.0
TOP = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="*", default=[1, 18])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    kinds = ("verb", "attribute", "classname")
    count = {k: collections.defaultdict(collections.Counter) for k in kinds}
    dfreq = {k: collections.defaultdict(collections.Counter) for k in kinds}
    files = parsed = 0
    domain_files = collections.Counter()
    t0 = time.time()

    for n in args.shards:
        for doc in read(fetch_shard(n), limit=args.limit):
            files += 1
            dom = classify_domain(doc.content)
            if dom is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tree = ast.parse(doc.content)
            except (SyntaxError, ValueError, RecursionError):
                continue
            parsed += 1
            domain_files[dom] += 1
            here = {k: set() for k in kinds}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    k, v = "verb", node.name
                elif isinstance(node, ast.Attribute):
                    k, v = "attribute", node.attr
                elif isinstance(node, ast.ClassDef):
                    k, v = "classname", node.name
                else:
                    continue
                if v.startswith("_") or len(v) < 3 or not v.isidentifier():
                    continue
                count[k][dom][v] += 1
                here[k].add(v)
            for k in kinds:
                for v in here[k]:
                    dfreq[k][dom][v] += 1
            if files % 40000 == 0:
                print(f"  {files} files, {parsed} classified, "
                      f"{time.time()-t0:.0f}s", flush=True)

    out = {"corpus": {"shards": args.shards, "files": files,
                      "classified_parsed": parsed, "domain_files": dict(domain_files),
                      "seconds": round(time.time() - t0, 1)},
           "filters": {"min_count": MIN_COUNT, "min_files": MIN_FILES, "ratio": RATIO},
           "vocabulary": {}}
    for k in kinds:
        out["vocabulary"][k] = _distinct(count[k], dfreq[k])
        for dom, tbl in out["vocabulary"][k].items():
            print(f"{k:10} {dom:10} n={len(tbl):4}  {', '.join(list(tbl)[:10])}")
    (OUT / "vocabulary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'vocabulary.json'} in {time.time()-t0:.0f}s")


def _distinct(count: dict, dfreq: dict) -> dict:
    totals = {d: max(sum(c.values()), 1) for d, c in count.items()}
    res = {}
    for dom, counter in count.items():
        scored = []
        for name, n in counter.items():
            if n < MIN_COUNT or dfreq[dom].get(name, 0) < MIN_FILES:
                continue
            own = n / totals[dom]
            other = sum(count[d].get(name, 0) / totals[d]
                        for d in count if d != dom)
            r = own / (other + 1e-12)
            if r >= RATIO:
                scored.append((name, n, r, dfreq[dom][name]))
        scored.sort(key=lambda t: (-t[2], -t[1]))
        res[dom] = {nm: {"count": n, "ratio": round(r, 2), "files": f}
                    for nm, n, r, f in scored[:TOP]}
    return res


if __name__ == "__main__":
    main()
