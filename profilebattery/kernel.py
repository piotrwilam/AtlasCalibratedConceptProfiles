"""Read the analysis kernel — the small object the 199 GB record reduces to.

The kernel holds per-group standardised means at the construct-token read site, one shard per
reference (`experiments/build_kernel.py` says what it keeps and what it drops, with reasons).
A profile is the unweighted mean of a set's group vectors and its reported SE is their
between-group spread, so every statistic in the study is a function of those vectors plus a
choice of which groups to include:

    from profilebattery.kernel import Kernel

    k = Kernel()
    v = k.contrast("pretrain_iter_0020000", "twin1", "For", layer=22)   # sibling-free A - B
    z = k.groups("pretrain_iter_0020000", "twin1", "For", "B", drop=":sibling:")

`drop` is how the sibling-free B set is rebuilt: the two-use readings needed a B set carrying
no use of the other kind, which in the record is the subtype groups whose name lacks
`:sibling:`. The earlier `_cores_all.npz` cache baked that filter in and could not undo it;
here the group names travel with the vectors, so any subset can be taken after the fact.
"""
from profilebattery.paths import data_root
import functools
import json
import os
import pathlib

import numpy as np

DEFAULT_ROOT = (data_root() / "kernel")


class Kernel:
    """Lazy reader over the kernel shards. One shard is opened per reference, on demand."""

    def __init__(self, root=DEFAULT_ROOT):
        self.root = pathlib.Path(root)
        man = self.root / "MANIFEST.json"
        if not man.exists():
            raise FileNotFoundError(f"no kernel at {self.root}; run experiments/build_kernel.py")
        self.manifest = json.loads(man.read_text())
        self.acts = self.manifest["acts"]
        self.site = self.manifest["site"]

    @property
    def references(self):
        return [r["reference"] for r in self.manifest["references"]]

    @functools.lru_cache(maxsize=4)
    def _shard(self, ref):
        return np.load(self.root / f"{ref}.npz")

    def sets(self, ref):
        """(stream, construct, kind, act) present at this reference."""
        z = self._shard(ref)
        return sorted({tuple(k.split("|")[:4]) for k in z.files if k.endswith("|z")})

    def group_names(self, ref, stream, construct, kind, act="h"):
        return list(self._shard(ref)[f"{stream}|{construct}|{kind}|{act}|groups"])

    def groups(self, ref, stream, construct, kind, act="h", drop=None, keep=None):
        """Group vectors [n_groups, n_layers, width], optionally filtered by name.

        `drop` excludes groups whose name contains it; `keep` restricts to those that do.
        Filtering by name is exact — the names are the accumulator's own group keys.
        """
        z = self._shard(ref)
        tag = f"{stream}|{construct}|{kind}|{act}"
        arr = z[f"{tag}|z"]
        names = [str(g) for g in z[f"{tag}|groups"]]
        idx = [i for i, g in enumerate(names)
               if (drop is None or drop not in g) and (keep is None or keep in g)]
        if not idx:
            raise ValueError(f"no groups left in {tag} after drop={drop!r} keep={keep!r}")
        return arr[idx]

    def profile(self, ref, stream, construct, kind, act="h", drop=None, keep=None):
        """The set's profile: the unweighted mean over its groups, [n_layers, width]."""
        return self.groups(ref, stream, construct, kind, act, drop, keep).mean(0)

    def se(self, ref, stream, construct, kind, act="h", drop=None, keep=None):
        """Between-group SE, the study's reported interval unit."""
        g = self.groups(ref, stream, construct, kind, act, drop, keep)
        return g.std(0, ddof=1) / np.sqrt(len(g))

    def contrast(self, ref, stream, construct, layer=None, act="h",
                 b_drop=":sibling:"):
        """The concept profile A - B. B is sibling-free by default, as the two-use readings used it."""
        a = self.profile(ref, stream, construct, "A", act)
        b = self.profile(ref, stream, construct, "B", act, drop=b_drop)
        v = a - b
        return v if layer is None else v[layer]

    def raw(self, ref, stream, construct, kind, act="h"):
        """The unstandardised set mean, [n_layers, width]."""
        return self._shard(ref)[f"{stream}|{construct}|{kind}|{act}|raw"]
