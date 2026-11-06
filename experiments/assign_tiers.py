"""Assign the deep/shallow tier and record how the union fell.

    python experiments/assign_tiers.py --compositional-level <the prior study's
        compositional_level.csv> [--n 40] [--dry-run]

Writes `tier` into every member of inventory_v2.json and a companion
calibration/tier.json holding the four input sets, the criterion (N) and the
outcome (|union|) separately — they answer different questions and PROVENANCE
quotes both.
"""
import argparse
import hashlib
import json
import pathlib

from profilebattery.paths import data_root
from profilebattery.generate.bpool import OTHER_ROLE
from profilebattery.inventory import tier
from profilebattery.inventory.tokens import sharers

DATA = data_root()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=tier.FREQUENCY_N)
    ap.add_argument("--dry-run", action="store_true")
    # the claim class comes from the earlier study's `compositional_level.csv` (its columns:
    # concept, language, compositional_level), which is not part of this release -- so it is
    # named on the command line rather than carried here as a path into another repository.
    ap.add_argument("--compositional-level", required=True, type=pathlib.Path,
                    help="the prior study's compositional_level.csv")
    args = ap.parse_args()

    rate = tier.rates(DATA / "calibration")
    claim = tier.claim_class(args.compositional_level)
    poly = tier.polysemous(sharers(), OTHER_ROLE)
    deep, sets = tier.assign(rate, claim, poly, n=args.n)

    inv_path = DATA / "inventory" / "inventory_v2.json"
    inv = json.loads(inv_path.read_text())
    members = set(inv["constructs"])

    unknown = {k: sorted(s - members) for k, s in sets.items() if s - members}
    print(f"{'input set':<14}{'n':>5}{'in top-N':>10}{'outside':>9}")
    top = sets["top_n"]
    for k, s in sets.items():
        print(f"{k:<14}{len(s):>5}{len(s & top):>10}{len(s - top):>9}")
    print(f"\ncriterion N = {args.n}   outcome |union| = {len(deep)}"
          f"   of {len(members)} inventory members")
    print(f"deep members outside the top {args.n} ({len(deep - top)}), "
          f"i.e. there because a claim needs them:\n  {', '.join(sorted(deep - top))}")
    if unknown:
        print("\nWARNING — set members absent from the inventory:")
        for k, v in unknown.items():
            print(f"  {k}: {', '.join(v)}")

    zero = sorted(c for c in deep if rate.get(c, 0) == 0)
    print(f"\ndeep members with zero corpus rate ({len(zero)}): {', '.join(zero)}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    for name, rec in inv["constructs"].items():
        rec["tier"] = "deep" if name in deep else "shallow"
    inv["counts"]["deep"] = len(deep & members)
    inv["counts"]["shallow"] = len(members - deep)
    inv_path.write_text(json.dumps(inv, indent=2))

    payload = {
        "criterion": {"frequency_n": args.n,
                      "note": "the knee of the cumulative-coverage curve, 97.5% of "
                              "mass; past it each member adds under 0.12%"},
        "outcome": {"deep": len(deep & members), "shallow": len(members - deep),
                    "union_before_inventory_filter": len(deep)},
        "sets": {k: sorted(s) for k, s in sets.items()},
        "bridge_note": "the programme documents call this the bridge-20, but no "
                       "enumeration of twenty was ever written down; these are the "
                       "v1 concepts carrying a published per-concept number "
                       "(probe_onset.json, and the concepts named by t1_riders.py "
                       "and isolated_sentinel.py), which come to 22",
        "deep_outside_top_n": sorted(deep - sets["top_n"]),
        "zero_corpus_rate": zero,
        "inventory_sha256": hashlib.sha256(inv_path.read_bytes()).hexdigest(),
    }
    out = DATA / "calibration" / "tier.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {inv_path} (deep {inv['counts']['deep']}, "
          f"shallow {inv['counts']['shallow']})\nwrote {out}")


if __name__ == "__main__":
    main()
