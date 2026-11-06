"""Generate and persist the validation-subset battery — handout §3, the artefacts.

Two twins, each drawing its own cores and its own C, plus the alternative-menu run. Every
prompt is written with its full metadata and its three read sites already resolved to token
indices, so nothing downstream recomputes an anchor.

    python experiments/build_battery.py [--per-set 720] [--constructs 30]
        -> $ATLAS_DATA/battery/
"""

from profilebattery.paths import data_root
import argparse
import collections
import itertools
import json
import os
import pathlib
import time

import numpy as np
from transformers import AutoTokenizer

from profilebattery.generate.battery import (match_offsets, offset_discrepancy,
                                             offset_target, record_from, write)
from profilebattery.generate.battery import control_from
from profilebattery.generate.floors import apply_floors, floors_for_set
from profilebattery.generate.bpool import build_b_cores
from profilebattery.generate.matching import corpus_profile
from profilebattery.generate.calibrated_menu import default_menu
from profilebattery.generate.seeds import stable_seed
from profilebattery.generate.cset import build_one as build_c
from profilebattery.generate.families import build_a_cores, family_of, sibling_b_cores
from profilebattery.generate.menu import DOMAINS, SCOPES, Cell
from profilebattery.generate.assemble import assemble, assemble_b
from profilebattery.inventory.tokens import NODE_TOKENS
from profilebattery.generate import weights as W

#: default output; --out stages a rebuild elsewhere so a relabel can be verified
#: text-identical before the live battery is touched
OUT = (data_root() / "battery")
CAPS = (data_root() / "probe/caps.json")
A_CAL = (data_root() / "calibration/a_calibration.json")
INV = (data_root() / "inventory/inventory_v2.json")
REPO = "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints"
REVISION = "pretrain_iter_0008000"
N_CORES, TWINS = 8, (1, 2)
CORE_SEED = {1: 11, 2: 23}
SURPLUS = 1.2


def content_hash(texts: list[str]) -> str:
    """One sha256 over a set's prompt texts, **newline-joined**.

    This is the hash in `CONTENT_HASHES.json` for the phase battery, and therefore the hash in
    every record's `prompt_sha256`. **It is NOT the same function `relabel_access` uses** --
    that one appends a NUL after each text -- and an earlier version of this docstring claimed
    it was. On 4 Oct that claim cost an hour: `run_validation`'s new battery check reimplemented
    the NUL form, reported all 964 sets of a correct battery as mismatching, and told Piotr his
    battery was the wrong one.

    The two forms are not reconciled here because the pilot battery's hashes are in the NUL
    form and rewriting them would break that record's provenance link. **Import this function
    rather than reimplementing it**, and when reading a hash, check which battery wrote it.
    """
    import hashlib
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


#: Constructs whose name collides with another construct's **case-insensitively**. macOS and
#: Windows filesystems fold case, so `Dict_A.jsonl.gz` and `dict_A.jsonl.gz` are one file: the
#: builtin `dict` silently overwrote the ast-node `Dict`, and the same for `list`/`List`,
#: `set`/`Set`, `slice`/`Slice`, `tuple`/`Tuple`. Five ast-node sets were destroyed by their
#: builtin namesakes and the loss was invisible -- the file existed, had 720 prompts, and held
#: the wrong construct. Four of the five are in the pilot's 30, and `Dict` and `Set` are two of
#: the three that fail P2, so the pilot comparison would have been against the wrong sets.
#:
#: The builtin side takes the suffix because the ast-node name is what every earlier artefact
#: and every pilot result already uses. `set_stem` is the single place filenames are formed and
#: `assert_distinct_stems` fails the build if any pair still folds together.
BUILTIN_STEM_SUFFIX = "__builtin"


def set_stem(construct: str, inventory: dict) -> str:
    """The filesystem stem for a construct's sets, unique under case folding."""
    entry = inventory["constructs"].get(construct, {})
    if entry.get("member_kind") != "builtin":
        return construct
    clash = any(c != construct and c.lower() == construct.lower()
                for c in inventory["constructs"])
    return f"{construct}{BUILTIN_STEM_SUFFIX}" if clash else construct


def construct_of_stem(stem: str) -> str:
    """The construct a stem names -- the inverse of `set_stem`."""
    return (stem[: -len(BUILTIN_STEM_SUFFIX)]
            if stem.endswith(BUILTIN_STEM_SUFFIX) else stem)


def assert_distinct_stems(subset: list[str], inventory: dict) -> None:
    """Fail before writing anything if two stems fold to the same filename."""
    seen: dict[str, str] = {}
    for c in subset:
        key = set_stem(c, inventory).lower()
        if key in seen:
            raise ValueError(f"set names collide under case folding: "
                             f"{seen[key]!r} and {c!r} both give {key!r}")
        seen[key] = c


def audit_read_sites(subset, inventory, menu, tok, cells, per_construct=8):
    """Refuse to build unless every construct reads ONE token piece on both sides.

    **Cowork's gate, and the mechanical form of a lesson that cost four rebuilds.** The read site
    is a token, and `▁for` (id 848) and `for` (2461) have input embeddings as unrelated as two
    random tokens, so a set that mixes them is not a set. The defect had four layers -- A's
    hosts, A's shapes, B's shapes, B's subtypes -- and each was found by rebuilding, measuring,
    and discovering the next. This audit works on token indices alone, so it runs on the
    generator's placements **before any prompt is written** and finds all four at once.

    **Scope, so a pass is not read as broader than it is.** It covers constructs whose surface
    token is identifier-shaped. A symbol token is exempt by `_piece_exact_positions`'s own
    declared policy -- a symbol merges with its neighbours whatever precedes it, which is ledger
    L5 and not something placement can repair -- so `BinOp`, `Call`, `Subscript` and the rest
    are skipped here and fail P₁ instead, which is where that fact belongs.

    It samples rather than enumerates: a placement that mixes pieces does so systematically --
    every one of these layers showed up as a clean 2:1 or 50/50 split, never as a rare case --
    so a handful of cells per construct is enough to catch it, and a full pass would cost as
    much as the build it is protecting.
    """
    from profilebattery.generate.align import line_start_offsets, token_spans
    from profilebattery.generate.assemble import _piece_exact_positions
    bad = {}
    for construct in subset:
        token = inventory["constructs"][construct].get("token")
        if not token:
            continue
        seen = {"A": collections.Counter(), "B": collections.Counter()}
        a_cores = build_a_cores(construct, 8, np.random.default_rng(11))
        b_cores = build_b_cores(construct, token, 3, np.random.default_rng(12))
        for side, cores in (("A", a_cores), ("B", b_cores)):
            for core in cores[:6]:
                for cell in cells[:per_construct]:
                    try:
                        lines = core.render(menu, cell, np.random.default_rng(13)).lines
                    except Exception:
                        continue
                    text = "\n".join(lines) + "\n"
                    cols = _piece_exact_positions(text, token, tok)
                    if not cols:
                        continue
                    ln, col = cols[0]
                    off = line_start_offsets(text)[ln] + col
                    pieces = [tok.convert_ids_to_tokens(i)
                              for i in tok(text, add_special_tokens=False)["input_ids"]]
                    spans = token_spans(pieces)
                    j = next((k for k, (lo, hi) in enumerate(spans) if lo <= off < hi), None)
                    if j is not None:
                        seen[side][pieces[j]] += 1
        for side in ("A", "B"):
            c = seen[side]
            if sum(c.values()) >= 4 and c.most_common(1)[0][1] / sum(c.values()) < 1.0:
                bad.setdefault(construct, {})[side] = dict(c.most_common(3))
    return bad


def _b_target(construct: str):
    """A's companion mix for this construct, or None where there is no corpus row for it.

    `corpus_profile` reads `construct_rates.json`, which covers the **63 syntactic constructs
    only** -- the 143 builtins have their own rates file with no companion table. Calling it
    unguarded raised `KeyError: 'ArithmeticError'` on the first builtin and killed the build
    after one set. `None` is the right fallback and not a silent one: it is the parameter's own
    documented default, and it means a builtin's B shapes are drawn for coverage rather than
    refitted -- which is what they were for every build before this, and which the companion
    cap does not test for builtins either, having no corpus row to test against.
    """
    try:
        return corpus_profile(construct) or None
    except KeyError:
        return None


def generable(inventory: dict) -> list[str]:
    """Every construct a family can build a set for -- the phase battery's population.

    `validation_subset` filters to `member_kind == "ast-node"`, which was right for a 30-construct
    validation subset and cannot reach the **143 builtins**: those are 139 of P0's 190, so a
    battery without them could not count P1 at all, let alone meet C3's power condition.

    Ungenerable, and therefore outside every gate that follows: 18 ast-nodes with no `a_family` or
    no entry in `NODE_TOKENS` -- the `Match*` family, the `Type*` family, `Await`, `Constant`,
    `FormattedValue`, `JoinedStr`, `Name`, `TryStar`, **`Yield` and `YieldFrom`**. The last two
    matter beyond their own rows: amendment A10 listed `yield` among the five piece-matched
    two-use groups and said none needed a new battery set. For `yield` that was wrong -- *neither*
    member is generable, so the group has no members in the battery and **A10's five groups are
    four**.
    """
    out = []
    for c, e in sorted(inventory["constructs"].items()):
        if not e.get("a_family"):
            continue
        if e["member_kind"] == "ast-node" and c not in NODE_TOKENS:
            continue
        out.append(c)
    return out


def validation_subset(inventory: dict, n: int) -> list[str]:
    """~n constructs, stratified by contrast class, from those a family can generate."""
    have = [c for c, e in inventory["constructs"].items()
            if e["member_kind"] == "ast-node" and e["a_family"] and c in NODE_TOKENS]
    by_class: dict[str, list[str]] = {}
    for c in sorted(have):
        by_class.setdefault(inventory["constructs"][c]["contrast_class"], []).append(c)
    # shuffled within class with a fixed seed: taking the head of each class alphabetically
    # would make the subset an alphabetical sample of it. Tier is not assigned yet, so
    # contrast class is the only stratum available and that is recorded as the limitation.
    rng = np.random.default_rng(4242)
    for cls in by_class:
        rng.shuffle(by_class[cls])
    out, i = [], 0
    while len(out) < n and any(by_class.values()):
        for cls in sorted(by_class):
            if by_class[cls] and len(out) < n:
                out.append(by_class[cls].pop(0))
        i += 1
        if i > 200:
            break
    return sorted(out)


def fitted_garnishes() -> tuple[dict, dict]:
    """The garnish probabilities already fitted for A and for B.

    Without these the battery is built from un-garnished sets, which is not what any audit
    measured — and it shows up immediately in the offset residual: a B carrier is one or two
    lines, so its read site sits near the end of the prompt while A's sits before a whole
    suite. The post-placement garnish is what gives B trailing material, so omitting it
    makes the read sites structurally unmatchable rather than merely unmatched.
    """
    a = {c: (v.get("pre", {}), v.get("post", {}))
         for c, v in (json.loads(A_CAL.read_text())["constructs"].items()
                      if A_CAL.exists() else [])
         if isinstance(v, dict) and "pre" in v}
    b = {c: (v.get("pre", {}), v.get("post", {}))
         for c, v in (json.loads(CAPS.read_text()).items() if CAPS.exists() else [])
         if isinstance(v, dict) and "pre" in v}
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-set", type=int, default=720)
    ap.add_argument("--constructs", type=int, default=30)
    ap.add_argument("--c-size", type=int, default=1000)
    ap.add_argument("--alt-per-set", type=int, default=60,
                    help="alternative-menu run size; its output is one cosine per construct")
    ap.add_argument("--only", default=None,
                    help="comma list of constructs instead of the validation subset -- for "
                         "the smoke build, where the control path has to be exercised and the "
                         "deterministic subset may contain no construct that needs one")
    ap.add_argument("--out", default=None,
                    help="write here instead of the live battery (staging a relabel)")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="build even though some construct mixes token pieces at the read "
                         "site -- right only for the operator families and the symbols, "
                         "whose surface token genuinely varies")
    ap.add_argument("--all", action="store_true",
                    help="every construct a family can generate -- the phase battery (item 11b). "
                         "`validation_subset` returns ast-nodes only and so cannot reach the 143 "
                         "builtins, which are 139 of P0's 190.")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = pathlib.Path(os.path.expanduser(args.out))
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    menu = default_menu()
    inventory = json.loads(INV.read_text())
    a_fit, b_fit = fitted_garnishes()
    print(f"fitted garnishes available: A for {len(a_fit)} constructs, "
          f"B for {len(b_fit)}")
    if args.only:
        subset = args.only.split(",")
    elif args.all:
        subset = generable(inventory)
    else:
        subset = validation_subset(inventory, args.constructs)
    assert_distinct_stems(subset, inventory)
    cells = [Cell(d, s, p, dr) for d, s, p, dr
             in itertools.product(DOMAINS, SCOPES, (0, 1, 2), (0, 1))]
    mixed = audit_read_sites(subset, inventory, menu, tok, cells)
    if mixed:
        print(f"\nread-site audit: {len(mixed)} construct(s) read more than one token piece")
        for c, d in list(mixed.items())[:12]:
            print(f"  {c:20} {d}")
        if not args.allow_mixed:
            raise SystemExit(
                f"refusing to build: {len(mixed)} construct(s) mix token pieces at the read "
                f"site. A profile cannot pool two tokens (A19/A19a). Pass --allow-mixed to "
                f"build anyway, which is only right for a construct whose surface token "
                f"genuinely varies -- the operator families and the symbols (ledger L5).")
    else:
        print("read-site audit: every construct reads one token piece on both sides")
    print(f"validation subset ({len(subset)}): {', '.join(subset)}")
    print(f"{len(cells)} cells x {N_CORES} cores = {len(cells)*N_CORES} combinations "
          f"per set; target {args.per_set}\n")

    manifest = {"per_set": args.per_set, "cells": len(cells), "cores": N_CORES,
                "garnish_fits": {"a": len(a_fit), "b": len(b_fit)},
                "twins": list(TWINS), "subset": subset, "menu": menu.provenance(),
                "sets": {}}
    t0 = time.time()
    for twin in TWINS:
        for construct in subset:
            cores = build_a_cores(construct, N_CORES,
                                  np.random.default_rng(CORE_SEED[twin]))
            a: list = []
            for core in cores:
                for cell in cells:
                    seed = stable_seed(twin, core.core_id, cell.key())
                    apre, apost = a_fit.get(construct, ({}, {}))
                    p = assemble(core, menu, cell, seed,
                                 pre_probs=apre or None, post_probs=apost or None)
                    if p.valid:
                        a.append(record_from(p, core, cell, seed, twin, "A", tok,
                                             inventory["constructs"]))
            rng = np.random.default_rng(1000 + twin)
            rng.shuffle(a)
            a = a[: args.per_set]
            tgt = offset_target(a)

            # the inventory's own `token`, not `NODE_TOKENS`, which covers ast-nodes only and
            # raised `KeyError: 'ArithmeticError'` on the first builtin of the phase build. The
            # two agree on all 60 shared entries (checked), and the only three constructs with
            # no token at all -- `Constant`, `Name`, `TypeVar` -- are ungenerable anyway.
            token = inventory["constructs"][construct].get("token") or NODE_TOKENS[construct]
            b_surplus: list = []
            # `target` is A's own companion mix, so B's shapes are chosen to match it rather
            # than drawn at random from the usable set. The parameter has existed since the
            # pool was written -- it ranks carriers by the *excess* they contribute, what a
            # shape brings that A rarely has, and keeps the lowest -- and it was never passed.
            # The companion cap is the check that B carries A's mix within a tolerance, so that
            # a companion's response cannot pass for the construct's in pi(A) - pi(B); it was
            # breached for 16 of 45 constructs at the token site, 12 of them inside P1 and so
            # read. The profile-method report's remedy for a breach is to give B A's mix --
            # a refit, not a looser cap (Cowork, 2 Oct).
            for bcore in (build_b_cores(construct, token, 5, np.random.default_rng(5 + twin),
                                        target=_b_target(construct))
                          + sibling_b_cores(construct, 4, np.random.default_rng(9 + twin))):
                for cell in cells:
                    seed = stable_seed(twin, bcore.core_id, cell.key())
                    bpre, bpost = b_fit.get(construct, ({}, {}))
                    p = assemble_b(bcore, token, menu, cell, seed, tokenizer=tok,
                                   pre_probs=bpre or None, post_probs=bpost or None)
                    if not p.valid:
                        continue
                    r = record_from(p, bcore, cell, seed, twin, "B", tok,
                                    inventory["constructs"], subtype=bcore.subtype)
                    if bcore.subtype.startswith("other_role:sibling:"):
                        r.pair_id = f"{construct}/sib/{bcore.core_id.split('/')[-1]}"
                    b_surplus.append(r)
            b = match_offsets(b_surplus, tgt, args.per_set,
                              rng=np.random.default_rng(2000 + twin))
            resid = offset_discrepancy(b, tgt)

            # The matched control set (G4, C2's role gap): one control per A prompt, for the
            # eleven constructs with a dependent class. Loss-only -- it carries no read site
            # and no yardstick -- and built *after* A is subsampled, so a control exists for
            # exactly the prompts that are kept. `control_from` also writes `control_index`
            # back onto the A record's positions, so the pair is navigable from either side.
            ctl = [c for c in (control_from(r, tok) for r in a) if c is not None]
            if ctl:
                info = write(ctl, OUT / f"twin{twin}" / f"{set_stem(construct, inventory)}_CTL.jsonl.gz")
                variants = collections.Counter(c.control_variant for c in ctl)
                info["variants"] = dict(variants)
                info["fell_back"] = sum(1 for c in ctl if c.control_fell_back)
                info["paired_positions"] = sum(
                    1 for r in a for m in r.positions if m.get("control_index") is not None)
                manifest["sets"][f"twin{twin}/{construct}/CTL"] = info

            # The floor at each obliged position: the conditional entropy of this set's own
            # continuations there (E10). Computed per set, because a floor is a property of the
            # distribution and a single prompt has no entropy; written onto the marks, so the
            # analysis reads it rather than re-deriving it from prompts that may be gone.
            def _pieces(text, _t=tok):
                return [_t.convert_ids_to_tokens(i)
                        for i in _t(text, add_special_tokens=False)["input_ids"]]
            fl = floors_for_set(a, _pieces)
            apply_floors(a, fl)
            manifest["sets"].setdefault(f"twin{twin}/{construct}/A", {})
            manifest["floors"] = manifest.get("floors", {})
            # `None` is a floor the slot cannot carry, not a missing number (A14): the
            # tokeniser merges the literal with the expression in front of it, so the slot is
            # several syntactic positions pooled. It is recorded as null, and C2 reads that
            # position on the loss alone.
            manifest["floors"][f"twin{twin}/{construct}"] = {
                k: (round(v, 4) if v is not None else None)
                for k, v in sorted(fl.items())}

            stem = set_stem(construct, inventory)
            for kind, recs in (("A", a), ("B", b)):
                info = write(recs, OUT / f"twin{twin}" / f"{stem}_{kind}.jsonl.gz")
                manifest["sets"][f"twin{twin}/{construct}/{kind}"] = info
            manifest["sets"][f"twin{twin}/{construct}/B"]["offset_residual"] = round(resid, 2)
            manifest["sets"][f"twin{twin}/{construct}/B"]["surplus"] = len(b_surplus)
            manifest["sets"][f"twin{twin}/{construct}/B"]["paired"] = sum(
                1 for r in b if r.pair_id)
            print(f"  twin{twin} {construct:16} A={len(a):4} B={len(b):4} "
                  f"(surplus {len(b_surplus):4}) CTL={len(ctl):4} "
                  f"offset residual {resid:6.2f} "
                  f"paired {sum(1 for r in b if r.pair_id):3}", flush=True)

        # each twin regenerates its own C
        c: list = []
        for i in range(args.c_size):
            cell = cells[i % len(cells)]
            seed = stable_seed(twin, "C", i)
            p = build_c(menu, cell, seed)
            if not p.valid:
                continue
            ids = tok(p.text, add_special_tokens=False)["input_ids"]
            c.append({"set_kind": "C", "text": p.text, "twin": twin, "seed": seed,
                      "domain": cell.domain, "scope": cell.scope,
                      "padding": cell.padding, "n_statements": p.n_statements,
                      "n_tokens": len(ids)})
        import gzip
        cpath = OUT / f"twin{twin}" / "C.jsonl.gz"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cpath, "wt") as fh:
            for r in c:
                fh.write(json.dumps(r) + "\n")
        manifest["sets"][f"twin{twin}/C"] = {"path": str(cpath), "n": len(c),
                                             "tokens": sum(r["n_tokens"] for r in c)}
        print(f"  twin{twin} C = {len(c)} prompts, "
              f"{sum(r['n_tokens'] for r in c):,} read positions")

    # the alternative-menu run: a second declared nu, not a second draw. Smaller by design —
    # its output is one cosine distribution per construct, and the reduced n is corrected for
    # at analysis (handout §6 check 4).
    from profilebattery.generate.alternative_menu import alternative_menu

    alt = alternative_menu()
    manifest["alternative_menu"] = alt.provenance()
    for construct in subset:
        cores = build_a_cores(construct, N_CORES, np.random.default_rng(CORE_SEED[1]))
        recs = []
        for core in cores:
            for cell in cells:
                if len(recs) >= args.alt_per_set * 2:
                    break
                seed = stable_seed("alt", core.core_id, cell.key())
                apre, apost = a_fit.get(construct, ({}, {}))
                p = assemble(core, alt, cell, seed,
                             pre_probs=apre or None, post_probs=apost or None)
                if p.valid:
                    recs.append(record_from(p, core, cell, seed, 0, "A", tok,
                                           inventory["constructs"]))
        rng = np.random.default_rng(3000)
        rng.shuffle(recs)
        recs = recs[: args.alt_per_set]
        info = write(recs, OUT / "alt" / f"{set_stem(construct, inventory)}_A.jsonl.gz")
        manifest["sets"][f"alt/{construct}/A"] = info
    print(f"\n  alternative menu: {len(subset)} sets of {args.alt_per_set}")

    # The prompt hash is written here, at generation, and not by a later script. The sweep
    # refuses to run without `CONTENT_HASHES.json` (G7, C0.1), so producing it in a separate
    # step meant a battery could exist without the hash that identifies it -- and a battery
    # whose hash was computed after an intervening edit identifies the wrong thing. One hash
    # over each set's prompt *texts*, which is what the record is of.
    hashes = {}
    for path in sorted(OUT.rglob("*.jsonl.gz")):
        texts = [json.loads(line)["text"] for line in gzip.open(path, "rt")]
        hashes[str(path.relative_to(OUT))] = {"sha256": content_hash(texts),
                                              "n": len(texts)}
    (OUT / "CONTENT_HASHES.json").write_text(json.dumps(hashes, indent=2))
    manifest["content_hashes"] = {"sets": len(hashes), "written_at": "generation"}
    print(f"  wrote CONTENT_HASHES.json over {len(hashes)} sets")

    manifest["wall_seconds"] = round(time.time() - t0, 1)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    tot = sum(v["n"] for v in manifest["sets"].values())
    print(f"\n{tot:,} prompts written to {OUT} in {manifest['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
