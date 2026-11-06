r"""Item 114: what this project cannot rebuild, why, and whether a second copy exists.

    python experiments/irreplaceable.py                  # the list, and what is where
    python experiments/irreplaceable.py --dataset        # also check the dataset (needs a token)
    python experiments/irreplaceable.py --write          # $ATLAS_DATA/IRREPLACEABLE.json

**Why this is a script and not a prose list.** `build_provenance.py` already states the house
rule -- generated rather than written, so the document cannot drift from the record it
describes. A hand-written list of irreplaceable artefacts would be out of date the first time
one moved, and the one thing this list exists to prevent is an artefact being lost while a
document says it is safe. So the reasons are prose and the locations are measured.

**The rebuild classes, which are the whole point.** An artefact is not irreplaceable because it
is valuable; it is irreplaceable because of *how* it was made:

* `impossible` -- the generator cannot produce it again at all. Every one of these is the same
  defect: seeds derived from `abs(hash(...))`, which Python salts per process (see
  `profilebattery/generate/seeds.py`). `stable_seed` fixed it, and the fix **postdates the
  artefacts**, so the recorded seeds cannot regenerate what they label.
* `gpu` -- rebuildable, but only with a paid A100 pass over released checkpoints.
* `cpu` -- rebuildable from something else in this list, so its safety is that thing's safety.
* `deterministic` -- rebuildable exactly, from a seed in the file itself; listed so that it is
  visibly *not* a worry.

A `cpu` artefact whose input is `impossible` is itself effectively impossible, and the report
says so rather than leaving the reader to chain it.

**What this check can and cannot see.** Two places: this machine's `$ATLAS_DATA` and
the dataset. **Not the shared Drive**, which has no listing from here -- so "one copy only"
means one copy *this check can see*, and several artefacts (`phase_battery`, `battery`,
`calibration`, `inventory`, `validation`) were also on the Drive before 7 Oct, under Piotr's own
listing of 6 Oct. What they lacked was a copy in the dataset, which is the one store that
outlives every laptop and every reclaimed session -- and the only one of the three this script
can verify rather than be told about.

**What `verified` means.** True where the generator was read and the class checked against it;
false where the class is CC's inference from the artefact's position in the pipeline. An
unverified `impossible` costs a backup nobody needed. An unverified `cpu` or `deterministic` is
the dangerous direction -- it is a promise that something can be remade -- so those are printed
as questions, not as answers.
"""

from __future__ import annotations
from profilebattery.paths import data_root, dataset_repo

import argparse
import hashlib
import json
import os
import pathlib

D = data_root()
REPO = dataset_repo()
#: hashing is for identifying a copy, not for auditing bytes; above this a tree is described by
#: its size and file count instead, because 198 GB through sha256 is an hour to say "still 198 GB"
HASH_CAP = 200 * 1024 * 1024

#: path, class, why, verified
ARTEFACTS: list[tuple[str, str, str, bool]] = [
    ("phase_battery", "impossible",
     "the battery the whole sweep is recorded against. Its per-prompt seeds came from "
     "`abs(hash((twin, core_id, cell_key)))`, salted per process, so the seeds in "
     "CONTENT_HASHES.json cannot regenerate the prompts they name. 188 constructs, 964 sets, "
     "seven builds of amendment work behind it.", True),
    ("battery", "impossible",
     "the pilot battery, 30 constructs. Same seed defect, and every pilot figure quoted before "
     "2 Oct -- the ablation pilot, the first twin units, C0.5's first reading -- is against it.",
     True),
    ("probe/companions.json", "impossible",
     "the companion table check 10 reads. `audit_companions.py` assembles its prompts with "
     "`seed=abs(hash((core.core_id, cell.key())))`, so a rebuild gives a different table. Was "
     "on the laptop alone until 6 Oct.", True),
    ("probe/bpool.json", "impossible",
     "the B pool every A/B contrast is read against. `probe_bpool.py` assembles it with "
     "`seed=abs(hash((core.core_id, cell.key())))`, salted per process.", True),
    ("probe/access.json", "impossible",
     "the access labelling. `relabel_access.py` built it from `abs(hash(tuple_of_strings))`, "
     "salted per process, and says in its own docstring that a rebuild produces a different "
     "labelling.", True),
    ("probe/caps.json", "cpu",
     "the companion cap, fitted per read site on one twin and audited on the other. "
     "`check_caps.py` rebuilds it from the battery, so it is as safe as the battery is -- which "
     "is to say not at all. Whether the fit itself is deterministic was not checked.", False),
    ("probe/probe.json", "impossible",
     "the machine probe: tokens/s with the 24 pre-hooks live, peak memory per batch size, disk "
     "throughput. Its own docstring says it measures the machine and not the design, and that "
     "machine is gone, so it cannot be re-measured -- but **nothing reads it**. Kept as the "
     "schedule's provenance, not as an input to any reading.", True),
    ("probe/probe_a100.json", "impossible",
     "the same probe on the A100 the sweep was costed against, and the source of the 6.85 "
     "units/h the budget rests on. That session is gone and a new one is a different machine; "
     "nothing downstream reads it either.", True),
    ("calibration", "impossible",
     "the A calibration, carried through four amendments (A15, A16 and their two corrections). "
     "Depends on the battery, so it inherits the battery's class whatever its own generator "
     "does.", False),
    ("inventory", "cpu",
     "the checkpoint inventory, from the released repository's listing. Rebuildable while "
     "OpenCoder's checkpoints stay published -- which is a dependency on somebody else.", False),
    ("validation/_archive_pilot_suites", "cpu",
     "the eight pilot SUITE.json files, archived 6 Oct before the closure run could overwrite "
     "one. `run_validation` is CPU over the pilot record, so these are rebuildable **only while "
     "the pilot record exists** -- which item 33 is deciding.", True),
    ("validation/battery/SPLIT.json", "deterministic",
     "the P1 split. Seed 20261002 is in the file and `split_p1.py` is deterministic given the "
     "battery -- so this is safe exactly as far as the battery is.", True),
    ("validation/schedule/CHECKPOINTS.json", "deterministic",
     "the 61 references in D5's order, with the dense block, the tail and the annealing branch "
     "-- `checkpoint_schedule.py` rebuilds it from the inventory, and the sweep, `hf_status "
     "--plan` and item 111 all read it as the definition of what the record should contain.",
     True),
    ("PROVENANCE.md", "cpu",
     "generated by `build_provenance.py` from the artefacts themselves, so it is rebuildable "
     "while they are. Note item 36: its `edf38a8f...` is the hash of this file rather than of "
     "the record it claims to identify.", True),
    ("smoke", "impossible",
     "a small battery build in its own right -- twin1, twin2, alt, CONTENT_HASHES.json and the "
     "SMOKE_COUNTS.json the first scope-use counts were read from. Built by the generator, so it "
     "carries the same salted-`hash()` seeds and cannot be rebuilt either; small enough that "
     "keeping it costs nothing.", True),
    ("smoke2", "impossible",
     "the second smoke build, same shape and same seed defect. Kept because the amendments were "
     "argued against the differences between these builds, so a figure quoted from one cannot be "
     "checked against a rebuild of it.", True),
    ("smoke95", "impossible",
     "the smoke build that measured amendment A16's scope-use repair: the `FunctionDef` 0 -> 525 "
     "and 18-of-37 calibration figures are quoted from it, and it carries the generator's salted "
     "seeds like every other build.", True),
    ("kernel", "gpu",
     "the kernel pass. An A100 pass over released checkpoints; `verify_kernel.py` checks it but "
     "cannot remake it without paying for it again.", False),
    ("profiles", "cpu",
     "derived profiles for the eight pilot-era references. Rebuildable from the pilot record, "
     "and item 52 is about to write the sweep's own -- but check 9 reads these.", True),
    ("record", "gpu",
     "the pilot record, 198 GB, 8 references. A paid A100 pass to remake, and item 33 decides "
     "whether it is kept at all. Listed because 'irreplaceable' and 'to be deleted' are "
     "compatible: the decision must be made knowing it cannot be undone.", True),
]


def tree_digest(p: pathlib.Path) -> tuple[str, int, int]:
    """(sha256 over the tree, bytes, files) -- or an empty digest above `HASH_CAP`.

    Names are hashed with the contents, so a renamed file is a different tree: the list is for
    telling one copy from another, and two copies with the same bytes under different names are
    not the same artefact.
    """
    if not p.exists():
        # `rglob` on an absent directory yields nothing, so the loop below would return the
        # sha256 of the empty string -- a stable, valid-looking digest for an artefact that is
        # not there. A list whose job is to notice a missing artefact must not do that.
        raise FileNotFoundError(p)
    if p.is_file():
        n = p.stat().st_size
        if n > HASH_CAP:
            return "", n, 1
        return hashlib.sha256(p.read_bytes()).hexdigest(), n, 1
    files = sorted(q for q in p.rglob("*") if q.is_file())
    total = sum(q.stat().st_size for q in files)
    if total > HASH_CAP:
        return "", total, len(files)
    h = hashlib.sha256()
    for q in files:
        h.update(str(q.relative_to(p)).encode("utf-8"))
        h.update(q.read_bytes())
    return h.hexdigest(), total, len(files)


def in_dataset(token) -> set[str]:
    """Every file path in the dataset. One listing; ~118,000 paths is nothing to hold.

    **Paths, not prefixes.** The first version collected top-level prefixes and asked whether
    an artefact's name was among them, which reported the laptop's `record/` -- the 8-reference
    *pilot* record, 212.9 GB, recorded against the 30-construct battery -- as safely in the
    dataset, because the dataset has a `record/` of its own: the 61-reference *sweep*, against
    the 188-construct battery. Same name, different contents, and the check said "safe" about
    the one artefact whose safety is the 198 GiB question item 33 is deciding. A name match is
    not a copy.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    return {e.path for e in api.list_repo_tree(REPO, recursive=True, repo_type="dataset",
                                               token=token)
            if getattr(e, "size", None) is not None}


#: how many of an artefact's files to look for upstream. A sample, because `record` holds 3,401
#: and the answer is the same after ten; all of them when there are fewer.
SAMPLE = 10

#: **where an artefact's second copy may live, besides its own relative path.** The pilot's three
#: big trees went up under `pilot/` rather than at their own names, because the pilot and the
#: sweep share the prefix `record/` and eight reference names (`upload_pilot.py`). This script did
#: not know that and reported `kernel` and `profiles` as single-copy on 7 Oct while both were
#: fully duplicated -- two of CC's own tools disagreeing about where a copy is, which is exactly
#: the kind of false alarm a safety list must not raise.
PREFIXES = ("", "pilot/")


def copy_upstream(p: pathlib.Path, paths: set[str]) -> bool | None:
    """Is this artefact in the dataset -- judged by its own files' paths, not by its name.

    Returns None when the artefact is not here to compare. Every sampled file must be present:
    a partial copy is not a copy, and saying so is the whole point of the column.
    """
    if not p.exists():
        return None
    # zero-byte files are not data and are never uploaded -- `record/.gitkeep` and
    # `profiles/.gitkeep` exist to keep an empty directory in git. They sort first, so they
    # landed in every sample and made two fully-duplicated trees read as single-copy.
    local = ([p] if p.is_file()
             else sorted(q for q in p.rglob("*") if q.is_file() and q.stat().st_size))
    if not local:
        return False
    step = max(1, len(local) // SAMPLE)
    sample = local[::step][:SAMPLE]
    return any(all(f"{pre}{q.relative_to(D)}" in paths for q in sample) for pre in PREFIXES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="store_true", help="also check the dataset (needs HF_TOKEN)")
    ap.add_argument("--write", action="store_true", help="write IRREPLACEABLE.json beside the data")
    a = ap.parse_args()

    upstream = None
    if a.dataset:
        from profilebattery.record.remote import hub_token, quiet_hub
        token = hub_token()
        if token is None:
            raise SystemExit("--dataset needs a token: `hf auth login` once on this machine, "
                             "or HF_TOKEN in the environment (Colab).")
        quiet_hub()
        upstream = in_dataset(token)

    rows, at_risk, unverified = [], [], []
    print(f"{'artefact':38} {'class':14} {'GB':>8} {'files':>7} {'local':>6} "
          + (f"{'dataset':>8}" if upstream is not None else ""))
    for path, cls, why, verified in ARTEFACTS:
        p = D / path
        here = p.exists()
        digest, size, files = tree_digest(p) if here else ("", 0, 0)
        up = None if upstream is None else copy_upstream(p, upstream)
        rows.append({"path": path, "class": cls, "why": why, "verified": verified,
                     "local": here, "bytes": size, "files": files,
                     "sha256": digest or None, "in_dataset": up})
        print(f"{path:38} {cls:14} {size/1e9:8.3f} {files:7} {'yes' if here else 'NO':>6} "
              + ("" if up is None else f"{'yes' if up else 'NO':>8}"))
        # the thing the list exists to find: one copy of something that cannot be remade
        if cls in ("impossible", "gpu") and here and up is False:
            at_risk.append(path)
        if not verified and cls in ("cpu", "deterministic"):
            unverified.append(path)

    total = sum(r["bytes"] for r in rows)
    print(f"\n{len(rows)} artefacts, {total/1e9:.1f} GB on this machine")

    if upstream is not None:
        if at_risk:
            print(f"\n**one copy only, and not rebuildable** -- these are item 114's whole "
                  f"point:\n  " + "\n  ".join(at_risk))
        else:
            print("\nevery impossible and gpu artefact has a second copy in the dataset")
    else:
        print("\nrun with --dataset (and a token) to say which of these have a second copy; "
              "without it this is a list and not a check")

    if unverified:
        print(f"\n**unverified in the dangerous direction** -- classed rebuildable by inference "
              f"rather than by reading the generator, so each is a promise that something can be "
              f"remade:\n  " + "\n  ".join(unverified))

    chained = [r["path"] for r in rows if r["class"] == "cpu"
               and ("pilot record" in r["why"] or "battery" in r["why"])]
    if chained:
        print(f"\nrebuildable only as far as their input is: {', '.join(chained)}")

    if a.write:
        out = D / "IRREPLACEABLE.json"
        out.write_text(json.dumps({"artefacts": rows, "hash_cap_bytes": HASH_CAP,
                                   "note": "classes: impossible | gpu | cpu | deterministic; "
                                           "see experiments/irreplaceable.py"}, indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
