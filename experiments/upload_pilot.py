r"""Items 114 and 33: the pilot's 236 GB into the dataset, under its own prefix.

    python experiments/upload_pilot.py                 # the plan, the hash, the provenance
    python experiments/upload_pilot.py --apply         # upload, resumably
    python experiments/upload_pilot.py --verify        # check all of it from the listing

**The decision this carries out** (Piotr, 7 Oct, with the arithmetic). The pilot record is kept
rather than deleted: every pilot number in the STORY rests on it -- Facts 1 to 6, item 108's
twin unit, the pilot check 9 medians, P5's +0.540 -- and those will be cited in the papers as
the pilot. A cited number whose record no longer exists cannot be checked. Keeping it is
**$3.83 a month** at $18 per TB-month; remaking it is eight references at about 2 h of A100
each, ~16 h and ~110 units, **and the remade record would not be bit-identical to the one the
numbers came from**. `kernel` is $0.21 a month against an A100 pass, `profiles` $0.21 against a
rebuild it would need the pilot record for anyway. All three go up: ~236 GB, ~$4.25 a month,
and the at-risk list goes to zero.

**Why `pilot/` and not `record/`.** The pilot and the sweep both name their references
`untrained`, `pretrain_iter_0001000` and six more, and both would sit under `record/<ref>/` --
but one is 8 references against the **30**-construct battery and the other 61 against the
**188**-construct one. Uploading the pilot to `record/` would land it on top of the sweep,
file by file, with no error anywhere. So the pilot takes its own prefix with `record/`,
`kernel/` and `profiles/` beneath it, and a `PROVENANCE.md` of its own naming the battery hash
and the eight references, so that the thing a paper cites is identified inside the store that
holds it.

**Resumable, because 236 GB is not one session -- and resumable by *reading the dataset*, not
by trusting a library.** The unit of work is one reference: `pilot/record/<ref>`, about 26 GB
and 425 files, one commit. Before uploading anything the script reads the dataset's listing and
**skips every unit already complete upstream**, so a re-run after an interruption costs a
listing and finishes the rest. `upload_large_folder` was the obvious tool and is the wrong one:
it has no `path_in_repo`, so it uploads to the repository root -- it would have put 236 GB of
pilot beside the sweep's `record/` with no error anywhere, which is the exact collision this
prefix exists to avoid.

**The laptop's copy stays until the listing says otherwise.** `--verify` checks every file of
all three trees against the dataset's own listing -- not a sample, as the 96.9 MB pass used,
because this is the copy that is meant to let the laptop's go.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import pathlib
import time

from experiments.irreplaceable import D, REPO

PREFIX = "pilot"
#: smallest first, so throughput is measured on 11.6 GB before 212.9 GB is committed to
TREES = ("kernel", "profiles", "record")
#: the pilot's eight references, in the order `build_provenance.py` lists them
REFERENCES = ["untrained", "pretrain_iter_0001000", "pretrain_iter_0002000",
              "pretrain_iter_0003000", "pretrain_iter_0005000", "pretrain_iter_0010000",
              "pretrain_iter_0015000", "pretrain_iter_0020000"]


def battery_hash(battery: pathlib.Path) -> tuple[str, int]:
    """The battery's prompt hash, by the one derivation the project already uses.

    sha256 over `name:sha256` lines of CONTENT_HASHES.json, sorted -- the same function as in
    `split_p1.py`, `read_site_overlap.py`, `scope_schedule.py` and `run_validation.py`, so the
    number written here is comparable with the one in every SUITE.json and MANIFEST.json.
    """
    h = json.loads((battery / "CONTENT_HASHES.json").read_text())
    digest = hashlib.sha256(
        "\n".join(f"{k}:{v['sha256']}" for k, v in sorted(h.items())).encode()).hexdigest()
    return digest, len(h)


def provenance(bh: str, sets: int, sizes: dict[str, tuple[int, int]]) -> str:
    total = sum(n for n, _f in sizes.values())
    lines = [
        "# The pilot record",
        "",
        f"*Placed in the dataset {time.strftime('%d %B %Y')} under `{PREFIX}/`, by "
        f"`experiments/upload_pilot.py`. Generated, not written.*",
        "",
        "The **30-construct pilot battery**, eight references. This is the record every pilot "
        "number in the STORY was read from, and those numbers are cited in the papers as the "
        "pilot: Facts 1 to 6, item 108's twin unit, the pilot check 9 medians, P5's +0.540. "
        "It is kept so that a cited number can be checked against the record it came from.",
        "",
        f"- **battery** `{PREFIX}`'s own: `{bh}`, {sets} prompt sets, 30 constructs. "
        f"**Not the sweep's battery**, which is `570b42db13f8fe57…` with 188 constructs and "
        f"964 sets.",
        f"- **references ({len(REFERENCES)})** " + ", ".join(f"`{r}`" for r in REFERENCES) + ".",
        "",
        "| tree | bytes | files | what it is |",
        "|---|---|---|---|",
    ]
    what = {"record": "the recorded activations: per-prompt arrays and per-core accumulators",
            "kernel": "the kernel pass, one `.npz` a reference — an A100 pass to remake",
            "profiles": "derived profiles, `A_mean`/`A_se`/`concept`/`concept_se`; the pilot's "
                        "check 9 was read from these"}
    for t in TREES:
        n, f = sizes.get(t, (0, 0))
        lines.append(f"| `{PREFIX}/{t}/` | {n/1e9:.1f} GB | {f} | {what[t]} |")
    lines += [
        f"| **total** | **{total/1e9:.1f} GB** | **{sum(f for _n, f in sizes.values())}** | |",
        "",
        "**Eight of these reference names are also sweep reference names**, where under "
        "`record/` they mean the 188-construct battery and 964 sets. That collision is why the "
        f"pilot has its own prefix: nothing of it is under `record/`, all of it under "
        f"`{PREFIX}/`.",
        "",
        "The pilot battery itself is at `battery/` in this dataset, placed the same day under "
        "item 114 — it cannot be regenerated, its per-prompt seeds having come from "
        "`abs(hash(...))`, which Python salts per process.",
    ]
    return "\n".join(lines) + "\n"


#: how long to wait when the Hub's commit limit is spent. Its own message says "about 1 hour";
#: five minutes over is cheaper than waking into the same refusal.
RATE_LIMIT_WAIT = 65 * 60


#: a dropped connection is the likeliest failure over twenty unattended hours on a laptop, and
#: it is the one that costs most: the unit in flight is up to 26 GB. Thirty tries a minute apart
#: rides out half an hour of outage.
NETWORK_WAIT, NETWORK_TRIES = 60, 30


def clear_xet_staging() -> int:
    """Drop the xet client's shard cache and staging area; return the bytes freed.

    **Derived state, and the only cure for one error.** A killed upload leaves shards in
    `staging/` that reference xorbs it never finished sending, and the next run then fails with
    `Internal error: invalid shard failed to verify file ...; xorb not found` -- which no amount
    of retrying inside the same process will fix, because the bad reference is on disk. Deleting
    it costs the dedup benefit of whatever was half-sent and nothing else: committed files are
    in the dataset and the source files are on this machine.
    """
    import shutil
    root = pathlib.Path(os.environ.get(
        "HF_XET_CACHE", os.path.expanduser("~/.cache/huggingface/xet")))
    freed = 0
    for parent in root.glob("https___*"):
        for name in ("staging", "shard-cache"):
            d = parent / name
            if not d.exists():
                continue
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    return freed


def wait_out_rate_limit(what: str, fn, tries: int = 4):
    """Run an upload; wait out the three failures that are not the job's fault.

    Three different things wear two exception types, and each clears on its own timescale.
    A **collision** -- the other writer committed first -- clears in seconds. A **429** is the
    repository's 256-commits-an-hour budget and clears in an hour; retrying that in seconds is
    not retrying, which is how item 101's first run died after three attempts five, ten and
    fifteen seconds apart. A **dropped connection** clears when the network comes back, which is
    minutes, and it arrives as a bare `ConnectionError` out of the xet layer -- not an
    `HfHubHTTPError` at all, which is why the first version of this caught none of it and the
    7 Oct upload died at `pilot/profiles` with the laptop briefly offline.
    """
    from huggingface_hub.errors import HfHubHTTPError
    attempt, net, shards = 0, 0, 0
    while True:
        attempt += 1
        try:
            return fn()
        except RuntimeError as e:
            # **a stale shard, and the fix is on disk rather than in time.** Only the xet
            # consistency failures are handled here; any other RuntimeError is a real bug and
            # must not be swallowed by a retry loop.
            msg = str(e)
            if not ("xorb not found" in msg or "invalid shard" in msg):
                raise
            shards += 1
            if shards > 2:
                raise
            freed = clear_xet_staging()
            print(f"    {what}: stale xet shard ({shards}/2). Cleared {freed/1e6:.0f} MB of "
                  f"shard cache and staging, which a killed upload leaves behind, and retrying.",
                  flush=True)
            attempt -= 1
        except (ConnectionError, TimeoutError) as e:
            # not an HTTP error and not the job's fault: wait for the network, and count these
            # separately so a long outage does not exhaust the budget for real failures
            net += 1
            if net > NETWORK_TRIES:
                raise
            print(f"    {what}: {type(e).__name__} ({net}/{NETWORK_TRIES}), the network is "
                  f"down or flapping; waiting {NETWORK_WAIT}s", flush=True)
            time.sleep(NETWORK_WAIT)
            attempt -= 1                         # a network outage is not a failed attempt
        except HfHubHTTPError as e:
            rate = "429" in str(e) or getattr(
                getattr(e, "response", None), "status_code", 0) == 429
            if attempt >= tries:
                raise
            wait = RATE_LIMIT_WAIT if rate else 5 * attempt
            print(f"    {what}: {'rate limit spent' if rate else type(e).__name__} on attempt "
                  f"{attempt}, waiting {wait/60:.0f} min", flush=True)
            time.sleep(wait)


def units(tree: pathlib.Path) -> list[pathlib.Path]:
    """One commit's worth of work: a reference directory, or the tree itself if it is flat.

    `record/` and `profiles/` hold one directory per reference, so each is ~26 GB and ~425
    files -- a commit large enough to be efficient and small enough that losing one to a
    dropped connection costs minutes. `kernel/` is nine files at the top level and goes as one.
    """
    subs = sorted(q for q in tree.iterdir() if q.is_dir()) if tree.exists() else []
    return subs or ([tree] if tree.exists() else [])


def sizes_of(trees=TREES) -> dict[str, tuple[int, int]]:
    out = {}
    for t in trees:
        p = D / t
        files = [q for q in p.rglob("*") if q.is_file()] if p.exists() else []
        out[t] = (sum(q.stat().st_size for q in files), len(files))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify", action="store_true", help="every file, against the listing")
    ap.add_argument("--only", default=None, help="one tree: record, kernel or profiles")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    from profilebattery.record.remote import can_write, hub_token, quiet_hub
    quiet_hub()
    token = hub_token()
    if token is None:
        raise SystemExit("no token: `hf auth login`, or HF_TOKEN in the environment")
    api = HfApi(token=token)

    trees = (a.only,) if a.only else TREES
    sizes = sizes_of(trees)
    bh, sets = battery_hash(D / "battery")
    print(f"pilot battery {bh[:16]}…, {sets} sets, {len(REFERENCES)} references")
    for t in trees:
        n, f = sizes[t]
        print(f"  {t:10} -> {PREFIX}/{t:10} {n/1e9:8.1f} GB {f:6} files")
    print(f"  {'total':10}    {' ':20} {sum(n for n, _f in sizes.values())/1e9:8.1f} GB "
          f"{sum(f for _n, f in sizes.values()):6} files")

    if a.verify:
        print("\nreading the dataset's listing in full:")
        upstream = {e.path for e in api.list_repo_tree(REPO, recursive=True,
                                                       repo_type="dataset", token=token)
                    if getattr(e, "size", None) is not None}
        bad = 0
        for t in trees:
            # **zero-byte files are not data and are not uploaded.** `profiles/.gitkeep` and
            # `record/.gitkeep` exist to keep an empty directory in git and hold nothing; the
            # Hub does not list them, and counting them as missing turns a complete copy into
            # "2 files missing -- re-run --apply", which is a false alarm in the one check that
            # decides whether 236 GB may be released from the laptop.
            local = sorted(q for q in (D / t).rglob("*") if q.is_file() and q.stat().st_size)
            missing = [q for q in local
                       if f"{PREFIX}/{q.relative_to(D)}" not in upstream]
            print(f"  {PREFIX}/{t:10} {len(local)-len(missing):6}/{len(local):<6} present"
                  + (f"   MISSING {len(missing)}" if missing else ""))
            for q in missing[:5]:
                print(f"      missing: {PREFIX}/{q.relative_to(D)}")
            bad += len(missing)
        print(f"\n{'every file of all three trees is in the dataset' if not bad else f'{bad} files missing — re-run --apply'}")
        print("The laptop's copy may be released only on the first of those two lines."
              if not bad else "")
        return

    if not a.apply:
        print("\nreport only. Re-run with --apply to upload (resumable; re-run until done).")
        print("\nthe PROVENANCE.md that would be written:\n")
        print(provenance(bh, sets, sizes))
        return

    ok, who = can_write(token)
    print(f"\ntoken: {who}")
    if not ok:
        raise SystemExit("this token cannot write to the Hub")

    print("\nreading the dataset once, to skip what is already there:")
    upstream = {e.path for e in api.list_repo_tree(REPO, recursive=True, repo_type="dataset",
                                                   token=token)
                if getattr(e, "size", None) is not None}
    print(f"  {len(upstream)} files upstream")

    for t in trees:
        for unit in units(D / t):
            rel = unit.relative_to(D)
            local = sorted(q for q in unit.rglob("*") if q.is_file())
            missing = [q for q in local if f"{PREFIX}/{q.relative_to(D)}" not in upstream]
            n = sum(q.stat().st_size for q in missing)
            if not missing:
                print(f"  {PREFIX}/{rel}: all {len(local)} files already upstream, skipped")
                continue
            print(f"  {PREFIX}/{rel}: {len(missing)}/{len(local)} files, {n/1e9:.1f} GB ...",
                  flush=True)
            t0 = time.time()
            # **waits out a rate limit rather than dying on it.** The Hub allows 256 commits an
            # hour per repository; this job makes 17 over 32 hours and so is nowhere near the
            # limit by itself, but it shares the repository with item 101's batches and can
            # arrive inside a window those have spent. Unlike a Colab cell this process is
            # detached and long-lived, so sleeping an hour is cheaper than stopping: stopping
            # needs a person to notice and relaunch.
            wait_out_rate_limit(f"{PREFIX}/{rel}", lambda: api.upload_folder(
                folder_path=str(unit), path_in_repo=f"{PREFIX}/{rel}",
                repo_id=REPO, repo_type="dataset", token=token,
                commit_message=f"items 114 and 33: the pilot's {rel}"))
            dt = max(time.time() - t0, 1e-9)
            print(f"    done in {dt/60:.1f} min, {n/1e6/dt:.1f} MB/s", flush=True)

    # the provenance last, so it never describes an upload that did not finish
    text = provenance(bh, sets, sizes_of())
    (D / "PILOT_PROVENANCE.md").write_text(text)
    api.upload_file(path_or_fileobj=text.encode(), path_in_repo=f"{PREFIX}/PROVENANCE.md",
                    repo_id=REPO, repo_type="dataset", token=token,
                    commit_message="items 114 and 33: the pilot record's own provenance")
    print(f"\nwrote {PREFIX}/PROVENANCE.md. Now run --verify before releasing the laptop's copy.")


if __name__ == "__main__":
    main()
