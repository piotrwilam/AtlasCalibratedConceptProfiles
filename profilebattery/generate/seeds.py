"""Stable per-prompt seeds.

The battery derived its seeds with `abs(hash((twin, core_id, cell_key))) % 2**31`.
Python salts `hash()` for str and bytes with a per-process random seed, so the same
call returns a different number in every interpreter — the battery was never
reproducible across runs, and the seeds recorded in PROVENANCE could not regenerate it.
Nothing downstream noticed, because a battery is normally built once.

`stable_seed` replaces it with a keyed digest: same inputs, same seed, any process, any
platform, any Python version.
"""
import hashlib

#: seeds are recorded as positive ints below 2**31, as the generator's rng expects
SEED_MODULUS = 2 ** 31


def stable_seed(*parts) -> int:
    """A reproducible seed from any tuple of values with a stable str()."""
    key = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % SEED_MODULUS
