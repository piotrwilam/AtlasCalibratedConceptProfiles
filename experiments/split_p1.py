r"""Item 19: count P1 on the phase battery, then make and hash the pre-registered split.

    python experiments/split_p1.py   ->  validation/battery/SPLIT.json

The contract (DESIGN §9): *"at the battery hash, if P1 >= 90, P0 minus the 30 pilot constructs is
split into two halves by a seeded draw stratified by frequency decile, complexity label and
token-piece class; the seed and both lists are hashed before recording; the analysis code refuses
to read the second half until the rules are fixed on the first, and the second half is then read
once. If P1 < 90, the whole record is read once."*

**P1 is counted with the SentencePiece marker stripped**, and that is not a detail: `|Base` and
`Base` are the same token in substance, and comparing the raw piece strings gave P1 = 1 of 183
rather than 138. The first figure was an artefact of the marker and nothing else.

**Two definitions the contract leaves open, declared here.** *Frequency* is the per-ktoken rate,
with the builtins' counts converted through the corpus denominator as `docs/FACTS.md` §4 sets out
-- `builtin_rates.json` has no `per_ktoken` field of its own. *Token-piece class* is read off the
tokeniser in three values: `keyword` (an identifier-shaped token that is one piece), `symbol` (a
token that is not identifier-shaped, so it merges with its neighbours -- ledger L5), and
`multi_piece` (a name the tokeniser splits, such as `ArithmeticError` into `Ar`+`ith`+`metic`+
`Error`). The class matters to the split because it is what P1 turns on, so the halves must be
balanced on it or one half would carry most of the unstable read sites.
"""

from profilebattery.paths import data_root
import collections
import hashlib
import gzip
import json
import math
import os
import pathlib
import statistics as st

import numpy as np
from transformers import AutoTokenizer

from experiments.build_battery import construct_of_stem, generable
from profilebattery.record.sweep import REPO

D = data_root()
BATTERY = D / "phase_battery"
OUT = D / "validation/battery/SPLIT.json"
TOK_REV = "pretrain_iter_0008000"
#: a NEW seed. The first draw was bound to the first battery's prompt hash and to a P1 count
#: computed under the wrong convention, so it is void (A19); redrawing under the old seed
#: would invite the suspicion that the halves were chosen after something had been seen.
SPLIT_SEED = 20261002
PILOT = ["AnnAssign", "Assert", "Assign", "Attribute", "AugAssign", "BinOp", "BoolOp", "Call",
         "Compare", "Delete", "Dict", "For", "FunctionDef", "GeneratorExp", "If", "IfExp",
         "ImportFrom", "List", "ListComp", "NamedExpr", "Pass", "Return", "Set", "Slice",
         "Starred", "Subscript", "Tuple", "UnaryOp", "While", "With"]


#: the scopes at which a statement-initial construct is indented, so its piece carries the
#: space marker. Imported rather than retyped: `twouse.py` has had this list since the
#: two-use sets were built, and the whole of A19a is that it was never applied elsewhere.
from profilebattery.generate.twouse import INDENTED_SCOPES  # noqa: E402


def read_piece(p: str | None) -> str:
    """The piece exactly as the tokeniser emitted it. **No normalisation** -- A19/A19a."""
    return p or ""


def piece_shares() -> dict[str, list[float]]:
    """Modal normalised anchor-piece share, per construct, over both twins and both sides."""
    out: dict[str, list[float]] = {}
    for twin in ("twin1", "twin2"):
        for f in sorted((BATTERY / twin).glob("*_A.jsonl.gz")):
            stem = f.name.rsplit("_", 1)[0]
            c = construct_of_stem(stem)
            for side in ("A", "B"):
                g = BATTERY / twin / f"{stem}_{side}.jsonl.gz"
                if not g.exists():
                    continue
                rows = [json.loads(l) for l in gzip.open(g, "rt")]
                if not rows:
                    out.setdefault(c, []).append(0.0)
                    continue
                rows = [r for r in rows if r.get("scope") in INDENTED_SCOPES]
                if not rows:
                    out.setdefault(c, []).append(0.0)
                    continue
                cnt = collections.Counter(read_piece(r.get("anchor_piece")) for r in rows)
                out.setdefault(c, []).append(cnt.most_common(1)[0][1] / len(rows))
    return out


def rates() -> dict[str, float]:
    """Per-ktoken rate for every construct, builtins converted (FACTS §4)."""
    cr = json.loads((D / "calibration/construct_rates.json").read_text())
    br = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    ks = [c for c in cr if cr[c].get("per_ktoken") and cr[c].get("occurrences")]
    ktok = st.median(cr[c]["occurrences"] / cr[c]["per_ktoken"] for c in ks)
    out = {}
    for c, v in cr.items():
        if v.get("per_ktoken"):
            out[c] = v["per_ktoken"]
    for c, v in br.items():
        if c not in out and v.get("occurrences"):
            out[c] = v["occurrences"] / ktok
    return out


def piece_class(construct: str, token: str, tok) -> str:
    n = len(tok(token, add_special_tokens=False)["input_ids"])
    if not token.strip().isidentifier():
        return "symbol"
    return "keyword" if n == 1 else "multi_piece"


def main():
    tok = AutoTokenizer.from_pretrained(REPO, revision=TOK_REV, trust_remote_code=True)
    inv = json.loads((D / "inventory/inventory_v2.json").read_text())
    labels = json.loads((D / "calibration/complexity_labels.json").read_text())["labels"]
    rate = rates()
    shares = piece_shares()

    p1 = sorted(c for c, s in shares.items() if len(s) == 4 and min(s) >= 0.80)
    failed = sorted(c for c in shares if c not in p1)

    # P0: a measurable rate on the proxy corpus. The split pool is P0 minus the pilot's 30.
    p0 = sorted(c for c in inv["constructs"] if rate.get(c))
    pool = sorted(set(p0) - set(PILOT))
    built = set(shares)

    strata: dict[tuple, list[str]] = collections.defaultdict(list)
    in_pool = [c for c in pool if c in rate]
    deciles = np.quantile([math.log10(rate[c]) for c in in_pool], np.linspace(0, 1, 11))
    for c in in_pool:
        dec = int(np.clip(np.searchsorted(deciles, math.log10(rate[c]), "right") - 1, 0, 9))
        lab = (labels.get(c) or {}).get("category", "unknown")
        token = inv["constructs"][c].get("token") or ""
        strata[(dec, lab, piece_class(c, token, tok))].append(c)

    rng = np.random.default_rng(SPLIT_SEED)
    first: list[str] = []
    second: list[str] = []
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        # alternate within each stratum, starting side chosen by the seeded draw, so a
        # stratum of odd size does not always give its extra member to the same half
        flip = bool(rng.integers(2))
        for i, c in enumerate(members):
            (first if (i % 2 == 0) != flip else second).append(c)

    def h(xs):
        return hashlib.sha256("\n".join(sorted(xs)).encode()).hexdigest()

    battery_hashes = json.loads((BATTERY / "CONTENT_HASHES.json").read_text())
    battery_hash = hashlib.sha256(
        "\n".join(f"{k}:{v['sha256']}" for k, v in sorted(battery_hashes.items())).encode()
    ).hexdigest()

    out = {
        "battery": str(BATTERY),
        "battery_prompt_hash": battery_hash,
        "battery_sets_hashed": len(battery_hashes),
        "p1_rule": "modal anchor piece >= 0.80 of prompts, MARKER KEPT, counted on the "
                   "indented scopes only (the primary reading), in both twins and on both A "
                   "and B. The earlier marker-stripped count of 138 is void -- A19/A19a.",
        "p1_count": len(p1), "p1": p1,
        "p1_failed_count": len(failed), "p1_failed": failed,
        "p1_threshold_for_split": 90,
        "split_fires": len(p1) >= 90,
        "p0_count": len(p0),
        "pool_rule": "P0 minus the pilot's 30 (DESIGN §9)",
        "pool_count": len(pool),
        "pool_not_in_battery": sorted(set(pool) - built),
        "strata": {"frequency": "log10 per-ktoken decile, builtins converted (FACTS §4)",
                   "complexity": "complexity_labels.json category",
                   "token_piece_class": "keyword | symbol | multi_piece, from the tokeniser",
                   "n_strata": len(strata)},
        "seed": SPLIT_SEED,
        "first_half": sorted(first), "first_half_sha256": h(first), "first_half_n": len(first),
        "second_half": sorted(second), "second_half_sha256": h(second),
        "second_half_n": len(second),
        "piece_class_counts": dict(collections.Counter(
            piece_class(c, inv["constructs"][c].get("token") or "", tok) for c in in_pool)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"P1 = {len(p1)} of {len(shares)} built   -> "
          f"{'SPLIT' if out['split_fires'] else 'READ ONCE'}")
    print(f"P0 = {len(p0)}; pool (P0 minus the pilot's 30) = {len(pool)}; "
          f"not in the battery: {out['pool_not_in_battery'] or 'none'}")
    print(f"strata: {len(strata)}; piece classes {out['piece_class_counts']}")
    print(f"first half  {len(first):3d}  sha256 {out['first_half_sha256'][:16]}")
    print(f"second half {len(second):3d}  sha256 {out['second_half_sha256'][:16]}")
    print(f"battery prompt hash {battery_hash[:16]} over {len(battery_hashes)} sets")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
