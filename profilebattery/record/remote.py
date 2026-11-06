"""Reading a record that lives on HuggingFace rather than on local disk.

**Why this exists.** Until 4 Oct the record was written to local disk and analysed from there.
It is now written to Colab's local disk, flushed to a HuggingFace dataset every fifty sets,
and the local copy freed -- because a reference is ~107 GB against a disk of 110 to 166 GB and
the sweep is 57 more references. So the record's home is the dataset, and every analysis script
that opened `$ATLAS_DATA/record/<ref>/<set>.npz` now opens nothing. `run_validation`
was the first to hit it, with the licence pass recorded and its own checks unable to read it.

**The manifest drives the fetching, and that is the whole point.** A naive "download the
reference" costs 107 GB and sixteen minutes. Worse, a naive per-file fetch is no better: a set
in P1 but outside the 32-construct hidden subset has a shard holding an *empty* `primary` and
a 77 MB `primary_resid`, so `np.load(shard)["primary"]` downloads 77 MB to discover there is
nothing in it -- 52 GB of that across the 671 such sets. The manifest records each set's
`shape`, so a caller can know the array is empty without moving a byte.

Local disk wins where it exists: a reference still on disk is read directly, so nothing changes
for a run that analyses as it records.
"""

from __future__ import annotations

import json
import os
import pathlib


def quiet_hub() -> bool:
    """Turn the Hub's per-file progress bars off, and say whether they were already off.

    **Process-wide and inherited by children.** Setting the environment variable as well as
    calling the function matters: `run_colab` spawns `run_sweep` and `loss_probe`, and the
    noisiest bars of all -- a 3.81 GB `pytorch_model.bin` redrawn a hundred times, and an
    upload redrawn once per file per second for ninety-two files -- come from those children.
    The variable crosses the `subprocess` boundary; the function call does not.

    They are not merely noise. Colab truncates a cell to its last 5,000 lines, so the bars
    push out the `[k/964]` lines, the flush summaries and any warning -- the three things a
    human reads a sweep by -- and a paste of the cell can be byte-identical between two
    different moments of the same run. Twice on 4 Oct that made a fetch that was progressing
    indistinguishable from one that had stalled.

    Returns the prior state, so a caller wanting them back can restore it rather than force
    them on -- which would undo a deliberate process-wide choice, and would also warn, since
    `enable_progress_bars` refuses to override the variable.
    """
    from huggingface_hub.utils import disable_progress_bars
    try:
        from huggingface_hub.utils import are_progress_bars_disabled
        was_off = bool(are_progress_bars_disabled())
    except ImportError:                         # an older hub: assume they were on
        was_off = False
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    disable_progress_bars()
    return was_off


def hub_token() -> str | None:
    """The token to use, from the environment or from the machine's own store.

    Two places, because the two machines store it differently and a script that reads only
    `HF_TOKEN` fails on the laptop in a way that looks like a credential problem and is not.
    In Colab the token comes from `userdata.get("HF_TOKEN")` into the environment; on a
    workstation `hf auth login` writes it under the Hub's cache and nothing exports it. The
    Hub itself falls back to the stored token when passed `token=None`, so the only thing
    needed here is to *report* which one is in play -- an error that says "HF_TOKEN is not
    set" on a machine that is perfectly well logged in sends the reader to the wrong fix.
    """
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    try:
        from huggingface_hub import get_token
        return get_token()
    except ImportError:
        return None


def can_write(token: str | None) -> tuple[bool, str]:
    """Whether this token may write to the Hub, and what it is -- checked before a long job.

    A read-only token is the right one to keep on a laptop, and the wrong one to discover
    half-way through rewriting a 121 GB reference. One call, no data.
    """
    from huggingface_hub import HfApi
    try:
        me = HfApi(token=token).whoami()
    except Exception as e:                       # noqa: BLE001 -- no token, or a bad one
        return False, f"the token could not be identified ({type(e).__name__})"
    auth = (me.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role") or ""
    name = me.get("name") or "?"
    if role in ("write", "fineGrained") or not role:
        # `fineGrained` does not say in `whoami` which scopes it holds, so it is allowed
        # through here and the write itself reports the refusal
        return True, f"{name}, token role {role or 'unknown'}"
    return False, f"{name}, token role {role}: read-only"


def _download(repo: str, filename: str, token: str | None) -> str:
    """The one place the Hub is called for a file. A seam, so a test need not reach the network.

    The import is local because `huggingface_hub` is not needed to read a record from disk, and
    the analysis scripts should not require it to be installed in order to do that.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename, token=token)


def _snapshot(repo: str, patterns: list[str], token: str | None, workers: int) -> str:
    """Fetch, with the per-file progress bars off.

    **They are not merely noise.** `huggingface_hub` redraws three bars per file, so an 856-file
    fetch emits tens of thousands of lines; Colab truncates the cell to the last 5,000, and a
    paste of it is byte-identical between two different moments of the same run. Twice on 4 Oct
    that made it impossible to tell a fetch that was progressing from one that had stalled. One
    line before and one after says more.
    """
    from huggingface_hub import snapshot_download

    was_off = quiet_hub()
    try:
        return snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns,
                                 token=token, max_workers=workers)
    finally:
        if not was_off:
            from huggingface_hub.utils import enable_progress_bars
            enable_progress_bars()


def _sizes(repo: str, token: str | None) -> dict[str, int]:
    """Every file's size, in one call, so a fetch can say up front how big it is."""
    from huggingface_hub import HfApi

    info = HfApi(token=token).repo_info(repo, repo_type="dataset", files_metadata=True)
    return {f.rfilename: (f.size or 0) for f in info.siblings}


def _tree(repo: str, prefix: str, token: str | None) -> dict[str, int]:
    """`{name relative to prefix: size}` for one reference, in **one scoped call**.

    **Why this exists rather than filtering a whole-repo listing.** `upstream()` used
    `list_repo_files`, which enumerates every file in the repository and then threw away all
    but one reference's; `prefetch` then called `repo_info(files_metadata=True)`, which does
    the same again *with metadata*, only to print an ETA. On this dataset that is 124,528
    files twice, and it cost **1,437 seconds for a six-construct trial** -- the listing took
    an order of magnitude longer than the download it was describing, and would have added
    about twenty hours of pure enumeration over 61 references. `list_repo_tree` scoped to the
    reference returns the same names *and* their sizes in one call over ~1,900 files.
    """
    from huggingface_hub import HfApi

    out = {}
    for f in HfApi(token=token).list_repo_tree(
            repo, path_in_repo=prefix.rstrip("/"), recursive=True,
            repo_type="dataset", expand=True):
        size = getattr(f, "size", None)
        if size is None:                        # a directory entry, not a file
            continue
        out[f.path[len(prefix):] if f.path.startswith(prefix) else f.path] = size
    return out


def _list_files(repo: str, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi(token=token).list_repo_files(repo, repo_type="dataset")


class RecordStore:
    """A reference's files, from local disk or the dataset, whichever has them.

    `path(name)` returns a local path, downloading on demand, or None when the file is in
    neither place. Downloads go to the HuggingFace cache, so a second call is free and a
    re-run after a disconnect re-reads rather than re-fetches.
    """

    def __init__(self, root: pathlib.Path, reference: str, repo: str | None = None,
                 token: str | None = None):
        self.root = pathlib.Path(root)
        self.reference = reference
        self.repo = repo
        self.token = token or os.environ.get("HF_TOKEN")
        self._manifest: dict | None = None
        self._cache_root: pathlib.Path | None = None
        self._sizes: dict[str, int] | None = None
        self.fetched, self.fetched_bytes = 0, 0

    # ---------------------------------------------------------------- locating

    @property
    def local_dir(self) -> pathlib.Path:
        return self.root / self.reference

    def _local(self, name: str) -> pathlib.Path | None:
        p = self.local_dir / name
        return p if p.exists() else None

    def upstream(self) -> set[str]:
        return set(self.upstream_sizes())

    def upstream_sizes(self) -> dict[str, int]:
        """The reference's files and their sizes, listed once and remembered."""
        if self._sizes is None:
            if not self.repo:
                self._sizes = {}
            else:
                pre = f"record/{self.reference}/"
                try:
                    self._sizes = _tree(self.repo, pre, self.token)
                except Exception as e:          # noqa: BLE001 -- any Hub failure
                    print(f"  cannot list {self.repo} ({type(e).__name__}: {e}); "
                          f"local disk only")
                    self._sizes = {}
        return self._sizes

    def path(self, name: str) -> pathlib.Path | None:
        p = self._local(name)
        if p is not None:
            return p
        if self._cache_root is not None:            # a prefetch put it here
            q = self._cache_root / name
            if q.exists():
                return q
        if name not in self.upstream():
            return None
        got = pathlib.Path(_download(self.repo, f"record/{self.reference}/{name}",
                                     self.token))
        self.fetched += 1
        self.fetched_bytes += got.stat().st_size
        return got

    def use_cache(self, root) -> None:
        """Point this store at a snapshot a *different* process already fetched.

        `path()` checks the cache root before it lists anything, so a pool worker given the
        root the parent's `prefetch` returned never calls the Hub at all. Without this each
        worker's first lookup pays the scoped listing -- about 12s on this dataset -- and with
        eight workers that is most of a small reference's wall clock, for an answer the parent
        already had.
        """
        self._cache_root = pathlib.Path(root)

    def exists(self, name: str) -> bool:
        return self._local(name) is not None or name in self.upstream()

    def prefetch(self, names: list[str], workers: int = 8) -> dict:
        """Fetch many files at once, into the same cache `path()` reads.

        **One file at a time is the wrong shape for this record.** `hf_hub_download` is serial
        and pays per-file setup, and the licence pass's validation measured 4-20 MB/s a file
        against the 115 MB/s the same connection uploads at -- about 90 minutes to move 54 GB
        that should take ten. `snapshot_download` fetches in parallel into the same cache, so
        every later `path()` is a cache hit and nothing else has to change.

        Files already local, or absent upstream, are left out rather than requested: asking for
        a pattern that matches nothing is not an error on the Hub, which means a typo would
        otherwise download nothing and report success.
        """
        if not self.repo:
            return {"fetched": 0, "why": "no repo"}
        up = self.upstream()
        want = [n for n in dict.fromkeys(names)
                if n in up and self._local(n) is None]
        if not want:
            return {"fetched": 0, "why": "all present locally or absent upstream"}
        import time
        # **Say how big this is before it starts.** A silent seven minutes is fine; seven
        # minutes with no idea whether it is seven or seventy is not. The sizes come from the
        # listing this store already did, not from a second whole-repo call: that second call
        # was costing more than the fetch it described.
        sz = self.upstream_sizes()
        total = sum(sz.get(n, 0) for n in want)
        if total:
            print(f"  prefetching {len(want)} files, {total/1e9:.1f} GB "
                  f"(~{total/1e9/0.1/60:.0f} min at 100 MB/s); progress bars off on purpose",
                  flush=True)
        else:
            print(f"  prefetching {len(want)} files; progress bars off on purpose", flush=True)
        t0 = time.time()
        root = pathlib.Path(_snapshot(
            self.repo, [f"record/{self.reference}/{n}" for n in want], self.token, workers))
        got = [f for f in (root / "record" / self.reference).glob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in got)
        dt = max(time.time() - t0, 1e-6)
        self._cache_root = root / "record" / self.reference
        self.fetched += len(want)
        self.fetched_bytes += size
        print(f"  prefetched {len(want)} files, {size/1e9:.1f} GB in {dt:.0f}s "
              f"({size/dt/1e6:.0f} MB/s, {workers} workers)", flush=True)
        return {"fetched": len(want), "bytes": size, "seconds": round(dt, 1)}

    # ---------------------------------------------------------------- manifest

    def manifest(self) -> dict:
        if self._manifest is None:
            p = self.path("MANIFEST.json")
            self._manifest = json.loads(p.read_text()) if p else {}
        return self._manifest

    def sets(self) -> list[str]:
        """Set names, from the manifest rather than a directory listing."""
        m = self.manifest().get("sets") or {}
        if m:
            return sorted(m)
        return sorted({f[:-4] for f in self.upstream()
                       if f.endswith(".npz") and not f.endswith(("_acc.npz", "_cells.npz"))}
                      | {f[:-len(".npz")] for f in
                         (q.name for q in self.local_dir.glob("*.npz"))
                         if not f.endswith(("_acc", "_cells"))})

    def per_prompt_rows(self, set_name: str) -> int:
        """How many per-prompt rows this set's shard holds, **without fetching it**.

        `shape` is `info`'s own key, written by `ShardWriter.write`. Returns -1 when the
        manifest does not say, so a caller can fall back to opening the file rather than
        assume it is empty -- the one reading that must not be guessed at.
        """
        entry = (self.manifest().get("sets") or {}).get(set_name)
        if not entry:
            return -1
        shape = entry.get("shape")
        if isinstance(shape, list) and shape:
            return int(shape[0])
        return -1

    def report(self) -> str:
        return (f"{self.fetched} files fetched from {self.repo} "
                f"({self.fetched_bytes/1e9:.1f} GB)" if self.fetched
                else "read entirely from local disk")

def purge_repo_cache(target: str) -> tuple[int, str]:
    """`hf cache rm <target>` then `hf cache prune`. Returns (commands that worked, how).

    **One place, and the return code checked.** Two drivers need this and both had their own
    copy that threw the result away. Two risks that silence covers:

    * the module path. `python -m huggingface_hub.cli.hf` is right on 1.32; the Colab runtime
      carries 2.2.0, and if the path moved there the purge would do nothing and say nothing.
      A reference caches a 3.81 GB checkpoint, so that fills the disk around reference 50 of
      61 -- which is how one pass already died, at 55.
    * the shared blob store. `cache rm` only detaches the revisions; `cache prune` reclaims the
      blobs. One without the other frees nothing on the 1.32+ layout, measured: the directory
      went away and the cache stayed at 33 GB.

    Both invocations are tried, the module first and the console script second, and the caller
    is told which worked. Freeing nothing is still legitimate -- there may be nothing cached --
    so this reports rather than raises; the caller measures the disk.
    """
    import subprocess
    import sys
    worked, how = 0, []
    for argv in (["cache", "rm", target, "-y"], ["cache", "prune", "-y"]):
        for cmd in ([sys.executable, "-m", "huggingface_hub.cli.hf", *argv], ["hf", *argv]):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.SubprocessError) as e:
                how.append(f"{cmd[0]} {argv[1]}: {type(e).__name__}")
                continue
            if r.returncode == 0:
                worked += 1
                how.append(f"{argv[1]} via {'module' if '-m' in cmd else 'script'}")
                break
            how.append(f"{argv[1]} via {'module' if '-m' in cmd else 'script'}: "
                       f"rc={r.returncode} {(r.stderr or '').strip()[:80]}")
    return worked, "; ".join(how)
