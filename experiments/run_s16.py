"""S16, rebuilt: the position-keyed loss pass at all 61 references — TODO-Settle S16.

    python experiments/run_s16.py --timed            # one reference, timed, nothing uploaded
    python experiments/run_s16.py --go               # all 61, resumable

**What S16 is now, and why it is wider than the item said.** The item asked for the loss probe
on the 11 matched-control sets. That could not work: `_CTL` rows carry
`site_construct_token = None` by construction, so an offset-anchored read produces nothing, and
the first pass wrote 11 empty sets at 54 references. Worse, `slot` is `None` for the governed
and scope positions of the **A** rows too, so the A/B pass never read the dependent positions
either -- neither half of the role gap existed.

So the pass reads at a position identified by class and token (S3's rule) over the **188 A sets
and the 11 controls**: 8,366,284 tokens a reference, 53% of the A+B pass by tokens. Cowork's
recommendation of 10 Oct, and the point of the wider scope is that every loss reading in the
plan -- obliged and dependent -- then comes from one artefact and the offset join is retired for
the primary readings.

**Three calls, decided by Cowork on 10 Oct and checked against the battery:** the gap is read at
dependent positions only (a control's header is a plain assignment, so A's obliged positions
have no counterpart); governed and scope split and never pooled (`class` carries it: 5,430 and
1,725 positions in the controls); `floor_nats` enters the obliged arm only.

**It writes a new artefact.** `record/<ref>/POSITIONS.json`, never a merge into `LOSS.json`:
that file is offset-keyed, this reading is position-keyed, and keeping them apart removes the
fetch-modify-reupload step that nearly cost the 376-set A and B pass.

**The timed reference exists because every inherited figure for this pass has been wrong.**
S16's own "about a minute a reference and 7 units" omitted 233 GB of checkpoint downloads; the
~14 h and ~100 units halves a 22-minute figure measured on another configuration; my own S15
estimate was out threefold and my float32 saving fivefold. `--timed` measures the rate on the
host that will run it and prints the projection; `--go` refuses until it has.
"""
from __future__ import annotations
from profilebattery.paths import data_root, dataset_repo

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

REPO = dataset_repo()
D = data_root()
#: where a reference's `POSITIONS.json` is written before it is uploaded and deleted. The
#: default is beside the rest of the data and works on any host; the Colab notebook points it
#: at `/content/positions` so nothing goes through the Drive.
OUT = pathlib.Path(os.path.expanduser(
    os.environ.get("PB_POSITIONS_DIR", str(D / "positions"))))
#: the checkpoint repo a different revision of which is loaded at every reference
MODEL_TARGET = "model/OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints"
#: tokens a full reference reads, counted from the battery: the **27** A sets that name
#: positions plus the 11 controls. The other 161 A sets are builtins and exception names with no
#: obliged, governed or scope position by design -- reading them was 74% of the tokens and
#: produced nothing, which the first timed reference showed as a column of `0 class/role`.
TOKENS_PER_REFERENCE = 2_135_922
TIMED_RECORD = D / "validation" / "S16_TIMED.json"
#: sets for the timed reference: enough to measure a rate, few enough to be quick
TIMED_SETS = 20
#: a real checkpoint, so the timed reference pays the download and load every other one does
TIMED_REFERENCE = "pretrain_iter_0484865"


def host_kind() -> str:
    """What the rate is a measurement *of*: the kind of host, not the machine.

    Colab hands out a fresh node name every session, so gating on the name would refuse to
    resume a run it was half way through -- the lesson from `run_s15`'s gate on 10 Oct.
    """
    import platform
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:                           # noqa: BLE001
        gpu = "unknown"
    return f"{platform.system().lower()}-{gpu}"


def free_gb(path="/") -> float:
    return shutil.disk_usage(path).free / 2 ** 30


def purge_model_cache() -> float:
    """Drop the cached checkpoint revisions. Returns the GiB freed, measured.

    **The failure this exists for, which S16's own cost note predicted and then ignored.** A
    revision is 3.81 GB and a different one is loaded at every reference; nothing dropped them,
    so 54 accumulated to about 206 GB on a runtime with roughly 110 and the pass died at
    reference 55 with `No space left on device`. `cache rm` detaches the revisions and
    `cache prune` reclaims the shared blobs -- one without the other frees nothing.
    """
    from profilebattery.record.remote import purge_repo_cache
    before = free_gb()
    worked, how = purge_repo_cache(MODEL_TARGET)
    if worked == 0:
        raise SystemExit(
            f"the cache purge ran nothing: {how}. This runtime's huggingface_hub may not expose "
            f"`hf cache`, and without the purge a 3.81 GB revision accumulates every reference "
            f"-- which filled the disk at reference 55 of 61. Stopping rather than discovering "
            f"it fifty references in.")
    return free_gb() - before


def stream(cmd: list[str]):
    """Run a child and put its output in this cell as it arrives.

    `subprocess.run` buffers until exit and its output never reaches a Colab cell, which once
    made a 90-minute fetch indistinguishable from a hung one.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    return proc.wait()


def run_pass(ref: str, out: pathlib.Path, batch: int, limit_sets=None) -> int:
    cmd = [sys.executable, "-m", "experiments.position_loss", "--reference", ref,
           "--batch", str(batch), "--out", str(out)]
    if limit_sets:
        cmd += ["--limit-sets", str(limit_sets)]
    print(f"$ {' '.join(cmd)}", flush=True)
    return stream(cmd)


def verify(doc: dict, want_controls: set) -> None:
    """Every set read something, and every control produced a gap.

    **Counting keys is what failed on 10 Oct.** S16's first pass wrote 11 `_CTL` entries with
    empty contents at 54 references and the check passed every one, because it counted that
    eleven keys had arrived and never looked inside them.
    """
    sets = doc.get("sets", {})
    if not sets:
        raise SystemExit("the pass produced no sets at all")
    empty = [k for k, v in sets.items() if not v.get("by_class_role")]
    if empty:
        raise SystemExit(f"{len(empty)} of {len(sets)} sets read nothing at any position: "
                         f"{empty[:4]}. Not uploading.")
    nofinite = [k for k, v in sets.items()
                if not any(s.get("mean") is not None for s in v["by_class_role"].values())]
    if nofinite:
        raise SystemExit(f"{len(nofinite)} set(s) have no finite mean: {nofinite[:4]}")
    gaps = doc.get("role_gap", {})
    missing = set(want_controls) - set(gaps)
    if missing:
        raise SystemExit(f"no role gap for {sorted(missing)}; not uploading a partial pass")
    dead = [c for c, g in gaps.items()
            if not any(s.get("mean") is not None for s in g["by_class_role"].values())]
    if dead:
        raise SystemExit(f"the role gap is empty for {dead}; the pairing read nothing")


def upload(ref: str, path: pathlib.Path, token) -> None:
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=str(path), path_in_repo=f"record/{ref}/POSITIONS.json",
        repo_id=REPO, repo_type="dataset",
        commit_message=f"S16: position-keyed losses and the role gap for {ref}")


def done_upstream(token) -> set:
    """References whose `POSITIONS.json` is already in the dataset.

    Read from the dataset, not from local disk, which is empty after a Colab restart.
    """
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    from experiments.run_s15 import references
    api = HfApi(token=token)

    def has(ref):
        try:
            return ref, api.file_exists(REPO, f"record/{ref}/POSITIONS.json",
                                        repo_type="dataset")
        except Exception:                       # noqa: BLE001 -- unreadable means not done
            return ref, False
    try:
        refs = references(token)
    except Exception as e:                      # noqa: BLE001
        print(f"  cannot list the record ({type(e).__name__}: {e}); nothing treated as done")
        return set()
    out = set()
    with ThreadPoolExecutor(16) as ex:
        for ref, ok in ex.map(has, refs):
            if ok:
                out.add(ref)
    return out


def controls() -> set:
    from experiments.position_loss import battery_dir
    return {q.name[:-len("_CTL.jsonl.gz")]
            for q in (battery_dir() / "twin1").glob("*_CTL.jsonl.gz")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timed", action="store_true",
                    help="one reference, a subset of sets, timed; nothing uploaded")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-free", type=float, default=15.0)
    args = ap.parse_args()
    if not (args.timed or args.go or args.reference):
        raise SystemExit("one of --timed, --go or --reference")

    from experiments.run_s15 import references
    from profilebattery.record.remote import hub_token
    token = hub_token()
    OUT.mkdir(parents=True, exist_ok=True)
    want = controls()
    if not want:
        raise SystemExit("no _CTL sets in the battery; the role gap cannot be read")
    print(f"host: {host_kind()} | {len(want)} control sets | {free_gb():.0f} GiB free\n")

    if args.timed:
        # **a real checkpoint, not `untrained_seed2`.** `references()[-1]` is sorted last and
        # is an untrained reference, which `load_lm` builds from config with a seed -- no
        # download and no real load. The tok/s would still be right, but the projection would
        # omit the 25-40 s every other reference spends fetching and loading a 3.81 GB
        # revision, understating a 3.4-minute reference by 15-20%.
        ref = args.reference or TIMED_REFERENCE
        out = OUT / f"TIMED_{ref}.json"
        if run_pass(ref, out, args.batch, limit_sets=TIMED_SETS) != 0:
            raise SystemExit("the timed reference failed; nothing recorded")
        doc = json.loads(out.read_text())
        rate = doc["tokens"] / max(doc["seconds"]["forward"], 1e-9)
        per = TOKENS_PER_REFERENCE / rate / 60 + doc["seconds"]["load"] / 60
        TIMED_RECORD.parent.mkdir(parents=True, exist_ok=True)
        TIMED_RECORD.write_text(json.dumps(
            {"host_kind": host_kind(), "reference": ref, "sets": TIMED_SETS,
             "tokens": doc["tokens"], "forward_s": doc["seconds"]["forward"],
             "load_s": doc["seconds"]["load"], "tok_per_s": round(rate, 1),
             "minutes_per_reference": round(per, 2),
             "hours_for_61": round(per * 61 / 60, 2),
             "when": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
        print(f"\nMEASURED on {host_kind()}: {rate/1000:.1f}k tok/s, "
              f"load {doc['seconds']['load']:.0f}s")
        print(f"a full reference is {TOKENS_PER_REFERENCE:,} tokens -> "
              f"{per:.1f} min a reference -> {per*61/60:.1f} h for 61")
        print(f"at 6.85 units an hour that is about {per*61/60*6.85:.0f} compute units")
        print(f"recorded to {TIMED_RECORD}; compare with the approved figure, then --go")
        return

    if args.go:
        try:
            rec = json.loads(TIMED_RECORD.read_text())
        except (FileNotFoundError, ValueError):
            raise SystemExit(f"no timed reference at {TIMED_RECORD}: run --timed first. Every "
                             f"inherited figure for this pass has been wrong, so the rate is "
                             f"measured on the host that will run it.")
        if rec.get("host_kind") != host_kind():
            raise SystemExit(f"the timed reference was measured on {rec.get('host_kind')!r} "
                             f"and this is {host_kind()!r}. The rate is the host's.")
        print(f"timed: {rec['tok_per_s']/1000:.1f}k tok/s, "
              f"{rec['minutes_per_reference']:.1f} min a reference, "
              f"{rec['hours_for_61']:.1f} h for 61\n")

    refs = [args.reference] if args.reference else references(token)
    have = set() if args.reference else done_upstream(token)
    todo = [r for r in refs if r not in have]
    print(f"{len(refs)} references, {len(have)} already upstream, {len(todo)} to do\n")

    t_run = time.time()
    for i, ref in enumerate(todo, 1):
        if free_gb() < args.min_free:
            raise SystemExit(f"only {free_gb():.1f} GiB free, need {args.min_free}: a "
                             f"checkpoint revision is 3.81 GB and the next reference needs "
                             f"another one.")
        print(f"\n=== [{i}/{len(todo)}] {ref} ({free_gb():.0f} GiB free) ===", flush=True)
        out = OUT / f"{ref}.json"
        t0 = time.time()
        if run_pass(ref, out, args.batch) != 0:
            raise SystemExit(f"{ref}: the pass exited non-zero; nothing uploaded")
        doc = json.loads(out.read_text())
        verify(doc, want)
        upload(ref, out, token)
        out.unlink(missing_ok=True)
        freed = purge_model_cache()
        left = (len(todo) - i) * (time.time() - t_run) / i
        print(f"--- {ref}: {doc['n_sets']} sets, {len(doc['role_gap'])} gaps, "
              f"{(time.time()-t0)/60:.1f} min, freed {freed:.1f} GiB. "
              f"{left/3600:.1f} h left on the mean of {i}", flush=True)
    print(f"\n{len(todo)} references in {(time.time()-t_run)/3600:.1f} h")


if __name__ == "__main__":
    main()
