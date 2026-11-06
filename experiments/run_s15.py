"""S15's driver: profiles for all 61 references, one reference at a time — TODO-Settle S15.

    python experiments/run_s15.py --trial                  # one reference, then stop
    python experiments/run_s15.py --go                     # the full run, resumable

**Why a driver and not a notebook cell.** Three of the four things that can go wrong here are
things that already went wrong once in this project, and all three want a guard that a test can
reach:

* **the disk fills.** Each reference needs about 25 GB of accumulators cached plus about 7 GB of
  output, and Colab gives ~100 GB. S1 filled it to 0.8 GB because `RecordStore` fetches into
  `~/.cache` and the cell deleted `/content/record`, a directory the files were never in. So the
  cache is purged **after each reference**, by path, and the free space is measured before and
  after; a purge that frees nothing stops the run instead of continuing into a full disk.
* **the upload is the only durable copy.** The output is uploaded per reference and the local
  copy freed, so a disconnect loses one reference, not the run.
* **resume must read the dataset, not the local disk.** The local disk is empty after a
  disconnect, and a resume that trusted it would redo all 61. What is upstream is the authority
  -- the same rule item 101 learned when resume-by-file-size called 22 of 962 files done before
  any had been slimmed.

The fourth is new: **the rate must be measured before the run is committed to.** S15 moves about
1.5 TB. From the Mac that is 1.66 MB/s and nine days; `--trial` does one reference and prints
the projection, and `--go` refuses to start until a trial has been recorded for this machine.
"""
from __future__ import annotations
from profilebattery.paths import data_root, dataset_repo

import argparse
import json
import platform
import os
import pathlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = dataset_repo()
D = data_root()
PROF = pathlib.Path(os.path.expanduser(os.environ.get("PB_PROFILES_DIR", str(D / "profiles"))))
#: the cache target for `hf cache rm`. **Not a directory to delete.** `huggingface_hub` 1.32
#: keeps file contents in a *shared, content-addressed* `hub/blobs/` store used by every repo,
#: and `hub/datasets--<owner>--<name>/` holds only pointers into it. So removing that directory
#: frees essentially nothing and merely orphans the blobs -- measured on this laptop: the
#: directory went away, the cache stayed at 33 GB, and 11.3 GB sat detached until
#: `hf cache prune` reclaimed it. A purge by `shutil.rmtree` would therefore have freed ~0 on
#: the first reference and stopped the run on its own guard, which is the guard working but the
#: run not. The CLI understands the layout; the free-space measurement stays as the check.
CACHE_TARGET = f"dataset/{REPO}"
TRIAL_RECORD = D / "validation" / "S15_TRIAL.json"
#: a trial that names both classes: two scope-varying constructs and two plain ones. A trial
#: over the first N alphabetically contains none of the 24 and under-reports the size by half.
TRIAL_ONLY = "For,If,AnnAssign,Return"


def host_kind() -> str:
    """What the trial's rate is a measurement *of* -- the kind of host, not the machine.

    **Not the hostname.** Colab allocates a fresh node with a random name every session, so a
    perfectly good Colab trial would stop matching the moment the runtime reconnected, and
    `--go` would refuse to resume a run it was half way through. The system and the core count
    are what the rate depends on, and they are stable across Colab sessions while still telling
    a 10-core darwin laptop apart from a Linux VM.
    """
    return f"{platform.system().lower()}-{os.cpu_count()}cpu"


def references(token=None) -> list[str]:
    """The 61 references, **from the record**, cross-checked against the schedule.

    Not derived from `CHECKPOINTS.json`: its reference list is `reference_set` with `dense`,
    `tail` and `anneal` and a count of 59, and the record holds 61 -- those 59 plus `untrained`
    and `untrained_seed2`. My first version of this read two top-level keys the file does not
    have and returned a single reference, and the test passed because it fed a schedule of my
    own invented shape. The record is the authority on what was recorded; the schedule is the
    authority on how many there should be, so both are consulted and a disagreement stops the
    run rather than quietly computing 1 reference or 60.
    """
    from huggingface_hub import HfApi
    # non-recursive: `record/` holds 117,732 files and enumerating them to learn 61 names is
    # the defect this module's sibling commit removed.
    entries = HfApi(token=token).list_repo_tree(
        REPO, path_in_repo="record", recursive=False, repo_type="dataset")
    refs = sorted(e.path.split("/")[-1] for e in entries
                  if getattr(e, "size", None) is None)       # directories only
    expect = expected_count()
    if expect is not None and len(refs) != expect:
        raise SystemExit(
            f"the record holds {len(refs)} references and the schedule expects {expect} "
            f"({sorted(refs)[:3]}...). Refusing to run against a set neither agrees on.")
    return refs


def expected_count() -> int | None:
    """59 checkpoints from the schedule, plus `untrained` and `untrained_seed2`."""
    try:
        sched = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    except (FileNotFoundError, ValueError):
        return None
    rs_ = sched.get("reference_set")
    if not rs_:
        return None
    n = rs_.get("n") or sum(len(rs_.get(k, [])) for k in ("dense", "tail", "anneal"))
    return n + 2


def done_upstream(token, arrays: str = "s15") -> set[str]:
    """References already written **in this array set**, read from the dataset.

    Two conditions, and the second is the one that matters here. `twin1/SUMMARY.json` must be
    present -- it is written last, so a reference interrupted half way has npz files without it
    and must be redone. And the summary must say `arrays: "s15"` with a partner twin recorded.

    **Presence alone is wrong for this run.** S1 left eight references upstream carrying the
    pre-S15 twelve arrays and a `twin1/SUMMARY.json` beside them. A resume that trusted the
    file would have skipped exactly the eight that need recomputing, and the tree would have
    ended half in one array set and half in the other -- the mixed state the whole ordering was
    arranged to avoid, reported as 53 successful references and 8 correctly skipped.
    """
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    have = set()
    try:
        for f in api.list_repo_tree(REPO, path_in_repo="profiles", recursive=True,
                                    repo_type="dataset", expand=True):
            parts = f.path.split("/")
            if len(parts) == 4 and parts[3] == "SUMMARY.json" and parts[2] == "twin1":
                have.add(parts[1])
    except Exception as e:                      # noqa: BLE001
        print(f"  cannot list profiles/ ({type(e).__name__}: {e}); treating nothing as done")
        return set()
    out, stale = set(), []
    for ref in sorted(have):
        try:
            q = hf_hub_download(REPO, f"profiles/{ref}/twin1/SUMMARY.json",
                                repo_type="dataset", token=token)
            d = json.loads(pathlib.Path(q).read_text())
        except Exception:                       # noqa: BLE001 -- unreadable means not done
            stale.append(ref)
            continue
        if d.get("arrays") == arrays and d.get("partner"):
            out.add(ref)
        else:
            stale.append(ref)
    if stale:
        print(f"  {len(stale)} reference(s) upstream in an older array set, to be redone: "
              f"{', '.join(stale[:4])}{'...' if len(stale) > 4 else ''}")
    return out


def already_done(ref: str, token, arrays: str = "s15") -> bool:
    """Whether this one reference is complete upstream, checked now rather than at startup.

    **Two hosts sharing the work is why this exists.** The todo list is computed once, so a
    second host working from the other end of it would be invisible: the first host would
    plough straight through everything the second had finished, redoing every one. Checking
    immediately before each reference costs one HEAD and one small GET, and bounds the overlap
    to the single reference each host happens to start at the meeting point.

    The condition is the same as `done_upstream`'s -- the summary present *and* in this array
    set -- because S1 left eight references upstream with a summary and the pre-S15 arrays.
    """
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    path = f"profiles/{ref}/twin1/SUMMARY.json"
    try:
        if not api.file_exists(REPO, path, repo_type="dataset"):
            return False
        q = hf_hub_download(REPO, path, repo_type="dataset", token=token)
        d = json.loads(pathlib.Path(q).read_text())
    except Exception:                           # noqa: BLE001 -- unreadable means not done
        return False
    return d.get("arrays") == arrays and bool(d.get("partner"))


def free_gb(p: pathlib.Path) -> float:
    return shutil.disk_usage(p).free / 2 ** 30


def purge_cache() -> float:
    """Drop this dataset's cached blobs. Returns the GiB freed, measured, not assumed.

    Two steps, because one is not enough on the shared-blob layout: `cache rm` detaches the
    repo's revisions, and `cache prune` then reclaims blobs nothing references any more. The
    free space either side is measured, so a purge that silently frees nothing is visible to
    the caller -- S1 deleted `/content/record`, a directory the files were never in, and ran
    the disk to 0.8 GB while reporting success at every step.
    """
    from profilebattery.record.remote import purge_repo_cache
    before = free_gb(pathlib.Path.home())
    worked, how = purge_repo_cache(CACHE_TARGET)
    if worked == 0:
        print(f"    the cache purge ran nothing: {how}", flush=True)
    return free_gb(pathlib.Path.home()) - before


def stream(cmd: list[str]) -> int:
    """Run a child and put its output in this cell's output as it arrives.

    `subprocess.run` buffers until exit and its output never reaches a Colab cell, which on
    4 Oct made a 90-minute fetch indistinguishable from a hung one.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    return proc.wait()


def compute(ref: str, extra: list[str]) -> int:
    """Run the writer as a **module**, from the repository root.

    Not by script path. Running `python experiments/compute_profiles.py` puts
    `experiments/` on `sys.path` and not the repository root, so `import profilebattery` works
    only where the package happens to be pip-installed -- true in this venv, false in Colab,
    which is the one place S15 can actually run. `-m` puts the working directory on the path,
    which is what the S1 notebook did and why it worked there.
    """
    return stream([sys.executable, "-m", "experiments.compute_profiles",
                   "--reference", ref, "--repo", REPO, "--accumulators-only"] + extra)


def upload(ref: str, token) -> None:
    """Upload one reference through the project's own Hub retry policy.

    **Called, not reimplemented -- and that is the whole point of this function.**
    `upload_pilot.wait_out_rate_limit` already handles the three failures that are not the
    job's fault, each on its own timescale: a stale xet shard, which is cured on disk by
    clearing staging; a bare `ConnectionError` out of the xet layer, which clears when the
    network returns and is counted against a separate budget so an outage does not spend the
    allowance for real failures; and a 429, which is the repository's 256-commits-an-hour
    budget and clears in an hour, not in seconds.

    I wrote a retry here that handled the **first of the three**. The run then died at
    reference 8 of 55 on the second -- `ConnectionError` to the xet CAS server -- having
    already died at reference 7 of 61 on the first before that handler existed. Both were
    already solved in a function twenty lines long that takes a callable.
    """
    from experiments.upload_pilot import wait_out_rate_limit
    wait_out_rate_limit(f"upload {ref}", lambda: _upload_once(ref, token))


def _upload_once(ref: str, token) -> None:
    from huggingface_hub import HfApi
    src = PROF / ref
    if not src.exists():
        raise SystemExit(f"nothing to upload for {ref}: {src} does not exist")
    HfApi(token=token).upload_folder(
        repo_id=REPO, repo_type="dataset", folder_path=str(src),
        path_in_repo=f"profiles/{ref}",
        commit_message=f"S15 profiles for {ref}")


def claim_output_dir(ref: str) -> None:
    """Refuse to write where something else already has.

    `upload_folder` sends **everything** under the directory, so a stale or foreign tree there
    is published along with this run's work. The default profiles directory on the laptop holds
    the 30-construct *pilot's* profiles under the same reference names -- 31 files under
    `pretrain_iter_0001000/twin1` -- and uploading that mixture to `profiles/<ref>` would put
    pilot-battery profiles into the sweep's tree, where every later reading consumes them.
    """
    d = PROF / ref
    if d.exists() and any(d.rglob("*")):
        raise SystemExit(
            f"{d} already holds files and this run has not written them. `upload_folder` would "
            f"publish all of it to profiles/{ref}. Point PB_PROFILES_DIR at an empty directory, "
            f"or remove that one if it is spent.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true",
                    help="one reference, four named constructs, then stop and print the "
                         "projection. Writes S15_TRIAL.json, which --go requires.")
    ap.add_argument("--go", action="store_true", help="the full run over every reference")
    ap.add_argument("--reference", default=None, help="just this one")
    ap.add_argument("--no-scope-cores", action="store_true")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes over constructs; 0 (the default) means one per cpu. "
                         "Serial, a reference is about 95 min on Colab's cpu and 61 are 4.5 "
                         "days; constructs are independent, so this is the whole difference.")
    ap.add_argument("--no-upload", action="store_true",
                    help="compute without uploading -- for a trial, never for the run")
    ap.add_argument("--reverse", action="store_true",
                    help="work the reference list from the end. Two hosts, one forward and one "
                         "reversed, share the run and meet in the middle -- and it balances "
                         "itself, since whichever is faster covers more. Each reference is "
                         "re-checked upstream immediately before it starts, so the overlap is "
                         "at most the one reference each host begins at the meeting point.")
    ap.add_argument("--pipeline", action="store_true",
                    help="overlap a reference's upload with the next reference's work. Saves "
                         "about 12 min of 50 a reference and needs about 45 GiB free, because "
                         "the prefetch happens inside the compute subprocess: there is no way "
                         "to overlap the upload with only the fetch, so two outputs (13.2 GiB) "
                         "and one reference's accumulators (25 GiB) are on disk at once. Off "
                         "by default -- on 11 Oct it ran a 44 GiB disk down to 6.3 GiB.")
    ap.add_argument("--min-free", type=float, default=45.0,
                    help="GiB that must be free before a reference starts")
    args = ap.parse_args()
    if not (args.trial or args.go or args.reference):
        raise SystemExit("one of --trial, --go or --reference")

    from profilebattery.record.remote import hub_token
    token = hub_token()
    extra = ["--no-scope-cores"] if args.no_scope_cores else []
    extra += ["--jobs", str(args.jobs)]
    print(f"{os.cpu_count()} cpu(s) on this host; --jobs {args.jobs} "
          f"({'one per cpu' if args.jobs == 0 else 'as given'})")

    if args.trial:
        t0 = time.time()
        ref = args.reference or 'pretrain_iter_0484865'
        print(f"TRIAL on {ref}: {TRIAL_ONLY}\n")
        rc = compute(ref, extra + ["--only", TRIAL_ONLY])
        dt = time.time() - t0
        if rc != 0:
            raise SystemExit(f"trial failed with {rc}; nothing recorded")
        TRIAL_RECORD.parent.mkdir(parents=True, exist_ok=True)
        TRIAL_RECORD.write_text(json.dumps(
            {"reference": ref, "only": TRIAL_ONLY, "seconds": round(dt, 1),
             "host": os.uname().nodename, "host_kind": host_kind(),
             "when": time.strftime("%Y-%m-%d %H:%M"),
             "no_scope_cores": args.no_scope_cores,
             "cpu_count": os.cpu_count(), "jobs": args.jobs}, indent=1))
        print(f"\ntrial took {dt/60:.1f} min for 4 constructs on {ref}.")
        print("The size per reference is printed above, weighted by the real mix.")
        print("A reference is 188 constructs and about 25 GB of accumulators; the four above "
              "fetched far less, so scale the RATE, not this wall time.")
        print("The size figure above is SAMPLED from four constructs and drifts between 6.74 "
              "and 6.93 GiB with which scope-varying ones it measures -- their B core sets run "
              "from 11 to 21. The exact figure, over all 188 with their real core counts, is "
              "6.50 GiB a reference and 426 GB for 61. Use the sampled number as a check on "
              "that, not as the figure.")
        print(f"Recorded to {TRIAL_RECORD}. Compare the size with the approved figure, then "
              f"re-run with --go.")
        return

    if args.go:
        # **the trial must be this host's.** The record lives on the Drive so it survives a
        # Colab restart, which means a trial run on the laptop would otherwise unlock the run
        # in Colab -- and the host is the whole variable: 1.66 MB/s here against whatever
        # Colab gives. A gate that any host can satisfy for any other is not a gate.
        here = host_kind()
        try:
            rec = json.loads(TRIAL_RECORD.read_text())
        except (FileNotFoundError, ValueError):
            rec = None
        if rec is None:
            raise SystemExit(
                f"no trial at {TRIAL_RECORD}: run --trial first. S15 moves about 1.5 TB and "
                f"the rate differs by more than an order of magnitude between hosts -- "
                f"1.66 MB/s measured from the Mac is nine days. Measure before committing.")
        if rec.get("host_kind") != here:
            raise SystemExit(
                f"the trial on record was measured on {rec.get('host_kind')!r} and this is "
                f"{here!r}. The rate and the worker count are the host's, so re-run --trial "
                f"here before --go.")
        if bool(rec.get("no_scope_cores")) != bool(args.no_scope_cores):
            raise SystemExit(
                f"the trial recorded no_scope_cores={rec.get('no_scope_cores')} and this run "
                f"asks for {args.no_scope_cores}. The size per reference differs by about "
                f"125 GB over 61 references, so the figure you compared is not this run's.")

    refs = [args.reference] if args.reference else references(token)
    done = set() if args.reference else done_upstream(token)
    todo = [r for r in refs if r not in done]
    if args.reverse:
        todo.reverse()
    print(f"{len(refs)} references, {len(done)} already upstream, {len(todo)} to do"
          + (" (from the end)" if args.reverse else ""))
    print(f"free disk {free_gb(pathlib.Path.home()):.1f} GiB; purging {CACHE_TARGET} after each reference\n")

    # **the upload overlaps the next reference's work only under `--pipeline`.** This comment
    # once said it was joined before the next compute when the code joined it before the next
    # upload, and anyone reading the loop would have believed it; the default is now inline.
    # Measured on reference 1: 12 min to
    # prefetch 26.9 GB at 36 MB/s down, 3.5 min to compute 188 constructs on four workers, and
    # **19 min to upload 6.5 GiB at 6.2 MB/s up** -- this line is asymmetric, so the upload is
    # the cost and the compute is a tenth of it. Strictly sequential that is 34 min a reference
    # and about 35 hours for 61.
    #
    # The upload is therefore handed to a thread and the next reference's prefetch starts at
    # once. It is joined **before the next compute**, not before the next upload, and that is a
    # disk constraint rather than caution: holding one reference's output (6.6 GiB) while the
    # next one's accumulators (26 GiB) are on disk peaks at 32.6 GiB, which is what reference 1
    # already peaked at; letting it run into the next compute would add a second output and ask
    # for 39 GiB of a 37 GiB disk. So the overlap is upload-against-prefetch, worth 12 of the
    # 19 minutes: about 22 min a reference, roughly 22 hours.
    pending: dict = {}
    up_rates: list = []

    def join_upload():
        """Finish the upload in flight, if any, and free its output. Raises what it raised."""
        if not pending:
            return
        ref_done, fut = pending.pop("ref"), pending.pop("fut")
        nbytes = pending.pop("bytes", 0)
        t_wait = time.time()
        fut.result()                    # re-raises inside the main thread
        shutil.rmtree(PROF / ref_done, ignore_errors=True)
        waited = time.time() - t_wait
        # **the upload's own rate, reported.** It is the whole cost on an asymmetric line --
        # 3.3 to 3.9 min of compute against 15 to 50 of upload, measured over seven references
        # -- so comparing this host with another needs the rate and not just the total.
        if nbytes and waited > 1:
            up_rates.append(nbytes / waited)
            print(f"    upload {nbytes/2**30:.2f} GiB in {waited/60:.1f} min "
                  f"({nbytes/waited/1e6:.1f} MB/s)", flush=True)
        print(f"    upload of {ref_done} finished"
              + (f" (waited {waited/60:.1f} min)" if waited > 5 else " (already done)"),
              flush=True)

    if args.pipeline and args.min_free < 45.0:
        raise SystemExit(
            f"--pipeline holds two outputs and one reference's accumulators at once, about "
            f"38 GiB, so it needs --min-free 45 or more; got {args.min_free}. Without the "
            f"headroom it runs the disk down to a few GiB, which is what happened on 11 Oct.")
    print(f"upload: {'overlapped with the next reference' if args.pipeline else 'inline'}\n")

    t_run, sizes = time.time(), []
    with ThreadPoolExecutor(max_workers=1) as up_pool:
        for i, ref in enumerate(todo, 1):
            # the in-flight upload holds one reference's output, so the floor is checked
            # against what this reference will add on top of it
            if free_gb(pathlib.Path.home()) < args.min_free:
                join_upload()           # its output is freed on the way out
            if free_gb(pathlib.Path.home()) < args.min_free:
                raise SystemExit(
                    f"only {free_gb(pathlib.Path.home()):.1f} GiB free, need {args.min_free}. "
                    f"A reference caches ~25 GB and writes ~7 GB. Stopping before the disk "
                    f"fills rather than after -- S1 ran it to 0.8 GB.")
            t0 = time.time()
            print(f"\n=== [{i}/{len(todo)}] {ref} "
                  f"({free_gb(pathlib.Path.home()):.0f} GiB free) ===", flush=True)
            # re-checked here, not trusted from the startup list: another host may have done it
            if already_done(ref, token):
                print("  done upstream since this run started; skipping", flush=True)
                continue
            claim_output_dir(ref)
            rc = compute(ref, extra)
            if rc != 0:
                join_upload()           # do not abandon a reference that is nearly up
                raise SystemExit(f"{ref} failed with {rc}; stopping. Nothing uploaded for it, "
                                 f"so a re-run redoes this reference and no other.")
            got = sum(f.stat().st_size for f in (PROF / ref).rglob("*") if f.is_file())
            sizes.append(got)
            # the cache goes now, not after the upload: the upload reads `PROF`, not the cache,
            # and the next prefetch needs the room
            freed = purge_cache()
            if freed < 1.0:
                join_upload()
                raise SystemExit(
                    f"the cache purge freed {freed:.1f} GiB, which means it did not purge the "
                    f"accumulators. Stopping: the next reference would run into a full disk, "
                    f"and a guard that reports success while changing nothing is the defect "
                    f"this project keeps meeting. Expected ~25 GB back from {CACHE_TARGET}.")
            if not args.no_upload:
                join_upload()           # one in flight at a time
                pending["ref"] = ref
                pending["bytes"] = got
                pending["fut"] = up_pool.submit(upload, ref, token)
                # **and joined again straight away unless pipelining is asked for.** The
                # overlap is upload-against-everything, not upload-against-prefetch: the
                # prefetch runs inside the compute subprocess, so an upload left in flight
                # spans the next fetch *and* the next compute, and two outputs plus one
                # reference's accumulators is about 38 GiB. The comment this loop carried
                # claimed the join happened before the next compute; it did not, and on
                # 11 Oct that took a 44 GiB disk to 6.3 GiB in two references.
                if not args.pipeline:
                    join_upload()
            dt, mean = time.time() - t0, sum(sizes) / len(sizes)
            left = (len(todo) - i) * (time.time() - t_run) / i
            print(f"--- {ref}: {dt/60:.1f} min to compute, {got/2**30:.2f} GiB, "
                  f"freed {freed:.1f} GiB; uploaded. "
                  f"mean {mean/2**30:.2f} GiB/ref -> {mean*61/2**30:.0f} GiB for 61. "
                  f"{left/3600:.1f} h left on the mean of {i} so far.", flush=True)
        join_upload()                   # the last one, before the summary claims it is done
    if up_rates:
        import statistics
        print(f"upload rate over {len(up_rates)} reference(s): "
              f"median {statistics.median(up_rates)/1e6:.1f} MB/s, "
              f"min {min(up_rates)/1e6:.1f}, max {max(up_rates)/1e6:.1f}")
    print(f"\n{len(todo)} references in {(time.time()-t_run)/3600:.1f} h")


if __name__ == "__main__":
    main()
