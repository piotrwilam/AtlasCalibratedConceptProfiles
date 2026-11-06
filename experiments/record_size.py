r"""RECORD_SIZE.json -- the record's true size, per reference, in all, and by kind of file.

Paper 1 quotes four storage figures (\S2.4 artefact line, \S3.3): 27.6 GB of accumulators a
reference, 107 GB for each of the four per-prompt references, 2,022 GB in all, and a
counterfactual 872 GB for storing the neurons per prompt for every construct. Before this
script only the total was stored anywhere (`record_GATE_111.json :: dataset_bytes_gb`), so
three of the four numbers had no artefact. This writes them.

**It is not CPU-only.** The sizes are read from the dataset repository's file tree, because
that is the record: the local tree holds only what has been pulled. Re-running it needs
network and read access to the dataset.

**The per-prompt bytes are split by construct.** A file is an accumulator if its name ends
`_acc.npz`, a C cell if `C_cells` is in its name, and per-prompt otherwise. The per-prompt
files are then split by whether their construct is in the 32-construct per-prompt subset
(`validation/battery/PER_PROMPT_SUBSET.json`), because those hold the MLP neurons and the
rest hold the residual stream -- which is the split \S3.3 quotes as "36 GB for the neurons of
31 constructs, 43 GB for the residual stream of 164", and the basis of its 144 GB and 872 GB.

    python experiments/record_size.py
"""
from __future__ import annotations

import collections
import json
import re
import time

from experiments.atlas import D, provenance
from profilebattery.paths import dataset_repo

REPO = dataset_repo()
OUT = D / "validation/paper1"


def main() -> None:
    from huggingface_hub import list_repo_tree
    t0 = time.time()
    per, nf, kind, kb = (collections.Counter() for _ in range(4))
    subset = set(json.loads(
        (D / "validation/battery/PER_PROMPT_SUBSET.json").read_text())["subset"])
    pp = collections.Counter()      # per-prompt bytes, split by construct
    pp_n = collections.Counter()
    for e in list_repo_tree(REPO, "record", repo_type="dataset", recursive=True):
        if getattr(e, "size", None) is None:
            continue
        parts = e.path.split("/")
        if len(parts) < 3:
            continue
        per[parts[1]] += e.size
        nf[parts[1]] += 1
        name = parts[-1]
        k = ("accumulators" if name.endswith("_acc.npz") else
             "C cells" if "C_cells" in name else
             "per-prompt primary" if name.endswith(".npz") else "other")
        kind[k] += e.size
        kb[k] += 1
        if k == "per-prompt primary":
            # twin1__For_A.npz, alt__Assert_B.npz -> the construct between __ and the set
            m = re.match(r"[^_]+__(.+)_[ABC][^.]*\.npz$", name)
            c = m.group(1) if m else None
            side = ("neurons (the per-prompt subset)" if c in subset
                    else "residual stream (the rest)" if c else "unattributed")
            pp[side] += e.size
            pp_n[side] += 1
    gb = {r: per[r] / 1e9 for r in sorted(per)}
    pre = [v for r, v in gb.items() if r.startswith("pretrain")]
    doc = {
        "what": "the record's size, per reference and by kind of file, from the dataset tree",
        "definition": {
            "accumulators_gb_per_reference": "the accumulator bytes over ALL references "
                                             "divided by the number of references -- the "
                                             "figure paper 1 quotes as 27.6",
            "per_prompt_reference_gb": "the largest per-reference total; the four per-prompt "
                                       "references carry the per-prompt store as well",
            "kind": "accumulators = *_acc.npz; C cells = *C_cells*; per-prompt primary = any "
                    "other .npz, which mixes MLP activations with per-offset losses"},
        "not_cpu_only": "sizes come from the dataset repository tree, not the local copy",
        "n_references": len(per),
        "n_files": sum(nf.values()),
        "total_GB": sum(per.values()) / 1e9,
        "total_bytes": sum(per.values()),
        "accumulators_gb_per_reference": kind["accumulators"] / 1e9 / max(1, len(per)),
        "per_prompt_reference_gb": max(gb.values()) if gb else None,
        "median_gb_per_reference": sorted(gb.values())[len(gb) // 2] if gb else None,
        "median_gb_per_pretraining_reference": sorted(pre)[len(pre) // 2] if pre else None,
        "by_kind_GB": {k: v / 1e9 for k, v in kind.items()},
        "by_kind_files": dict(kb),
        "per_prompt_split_GB": {k: v / 1e9 for k, v in pp.items()},
        "per_prompt_split_files": dict(pp_n),
        "per_prompt_split_per_reference_GB": {
            k: v / 1e9 / max(1, sum(1 for r in gb if gb[r] > 60)) for k, v in pp.items()},
        "n_per_prompt_references": sum(1 for r in gb if gb[r] > 60),
        "per_prompt_subset_size": len(subset),
        "per_reference_GB": gb,
        "minutes": (time.time() - t0) / 60,
        "provenance": provenance([f"hf://{REPO}/record (file tree, sizes only)"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RECORD_SIZE.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"  {doc['n_references']} references, {doc['n_files']:,} files, "
          f"{doc['total_GB']:.1f} GB in all")
    print(f"  accumulators {doc['accumulators_gb_per_reference']:.2f} GB a reference; "
          f"largest reference {doc['per_prompt_reference_gb']:.1f} GB")
    for k, v in doc["per_prompt_split_per_reference_GB"].items():
        print(f"  per-prompt, {k}: {v:.1f} GB a reference "
              f"({v * doc['n_per_prompt_references']:.0f} GB over "
              f"{doc['n_per_prompt_references']} references)")
    print(f"-> {OUT / 'RECORD_SIZE.json'}")


if __name__ == "__main__":
    main()
