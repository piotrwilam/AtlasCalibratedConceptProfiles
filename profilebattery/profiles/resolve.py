r"""One resolver for every reader of the derived profiles, and a hard stop on local copies.

Three times now a pass has read the **pilot** recording where it meant the **phase** recording,
because a local directory held the pilot's eight references under the phase recording's name and
the reader preferred whatever was on disk:

  A50  `record/`   -- accumulators; A48's C0.5 departure came from comparing the two
  A51  `record/`   -- verified against `pilot/record/` and retired, 154 GiB freed
  A52  `profiles/` -- the derived tree; caught before test 6 wrote anything, at cosine 0.87-0.91
                      against the dataset at the same reference, twin and construct

The pattern each time: eight references -- `untrained`, `0001000`, `0002000`, `0003000`,
`0005000`, `0010000`, `0015000`, `0020000` -- content that differs from the dataset, and a
reader that prefers the local copy silently. The guard below does two things a per-module fix
would not. It reads the dataset and nothing else, so a correct result cannot depend on what
happens to be on disk; and it **fails** when a local tree exists at those references, so the
next person to put one there is told rather than quietly served from it.
"""

from __future__ import annotations

import io
import os
import pathlib

import numpy as np

from profilebattery.paths import dataset_repo

REPO = dataset_repo()
#: the pilot's references, which is the set that keeps appearing under the phase name
PILOT_REFERENCES = ("untrained", "pretrain_iter_0001000", "pretrain_iter_0002000",
                    "pretrain_iter_0003000", "pretrain_iter_0005000", "pretrain_iter_0010000",
                    "pretrain_iter_0015000", "pretrain_iter_0020000")
_FS = None


def _token():
    t = os.environ.get("HF_TOKEN")
    if t is None:
        from huggingface_hub import get_token
        t = get_token()
    return t


def refuse_local_tree(root, what: str = "profiles") -> None:
    """Fail if a local tree holds the pilot's references under the phase recording's name."""
    p = pathlib.Path(os.path.expanduser(str(root)))
    if not p.is_dir():
        return
    here = {d.name for d in p.iterdir() if d.is_dir()}
    overlap = sorted(here & set(PILOT_REFERENCES))
    if len(overlap) >= len(PILOT_REFERENCES) - 1 and not here - set(PILOT_REFERENCES):
        raise SystemExit(
            f"{p} holds {len(overlap)} of the pilot's eight references and nothing else.\n"
            f"  That is the shape of the pilot {what} sitting where the phase {what} is "
            "expected (A50/A51/A52). Reading it would compare two different recordings.\n"
            "  Move it aside, or read from the dataset: "
            "`profilebattery.profiles.resolve.profile(...)`.")


def profile(ref: str, twin: str, construct: str, keys=None):
    """A derived profile: the local dataset if it has one, else the Hub.

    A52 made this dataset-only because the *private* tree's `profiles/` held the pilot
    recording at the phase recording's path, so a local read silently compared two different
    recordings. That hazard does not exist here: the local tree is the released dataset
    itself, named by `$ATLAS_DATA`. So a reader who has the dataset reads it from disk and
    needs no network, and anyone without it still falls back to the Hub.
    """
    global _FS
    from profilebattery.paths import data_root, dataset_repo
    local = data_root() / "profiles" / ref / twin / f"{construct}.npz"
    if local.exists():
        with np.load(local) as z:
            want = z.files if keys is None else [k for k in keys if k in z.files]
            return {k: np.asarray(z[k]) for k in want}
    if _FS is None:
        from huggingface_hub import HfFileSystem
        _FS = HfFileSystem(token=_token())
    path = f"datasets/{REPO}/profiles/{ref}/{twin}/{construct}.npz"
    if keys is None:
        # no key list: the caller wants the file, so read it whole
        with _FS.open(path, "rb") as f:
            buf = io.BytesIO(f.read())
        with np.load(buf) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    # **Named members only.** A profile npz is about 99 MB and `concept` is 0.6 MB of it; the
    # cores arrays are the rest. Reading the whole file is invisible single-threaded, where the
    # cost is round-trip latency, but with eight workers it saturates the link: 2.02 s a file
    # against 0.57 s when only the member is fetched. np.load on the open handle seeks.
    with _FS.open(path, "rb") as f:
        with np.load(f) as z:
            return {k: np.asarray(z[k]) for k in keys if k in z.files}


def concept(ref: str, twin: str, construct: str) -> np.ndarray:
    z = profile(ref, twin, construct, ("concept", "A_mean", "B_mean"))
    v = z.get("concept")
    return np.asarray(v if v is not None else z["A_mean"] - z["B_mean"], np.float64)
