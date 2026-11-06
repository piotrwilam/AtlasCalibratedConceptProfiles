r"""The three watched numbers — item 103.

**These have never been computed.** `run_colab` has a gate that reads `manifest["watched"]` and
aborts the sweep when a number is out of range, with the comment "Aborting at reference one
rather than at fifty-nine". `run_sweep` writes no `watched` key, so `d.get("watched")` has
always been `None`, the loop over it has always been empty, and the gate has always returned
`ok=True` — on every reference of the pilot and on both references of the licence pass. It
looked like a check and was an empty loop. Found on 4 Oct only because a *different* defect
made the missing-manifest branch fire first.

The three numbers, and what each is for. They are a smoke test on a reference, not a reading:
the question is whether the recording apparatus produced something usable, cheaply enough to
ask after every reference.

* **twin agreement** — the median over sampled constructs of twin1 against twin2 at the
  construct-token site, as `checks.twins`'s median cosine. Two batteries sample one declared
  distribution, so this is the noise floor; if it collapses, the reference is not measuring
  what the earlier ones measured. Floor 0.30.

* **concept norm** — the median magnitude of pi(A) - pi(B). A reference whose activations came
  out zero, or whose A and B sets were somehow recorded from the same prompts, has a concept
  profile of no magnitude at all. Floor 1e-3, which is far below any real value and is there to
  catch a dead record rather than a weak one.

* **carrier overlap** — the fraction of twin1's carrier set that twin2 also carries, where the
  carrier set is the top 5% of the squared mass of pi(A) - pi(B) over (layer, unit), which is
  C0's own definition and the one `ablation_pilot` uses. Bounded **both ways**, 0.02 to 0.995.
  Below, the twins agree on nothing and the carrier is noise. Above, they agree too perfectly
  to be two samples — which is what a reference recorded twice under two names, or a profile
  that is constant across units, would look like.

**Each number is a median over a sample, and the sample is recorded in the manifest.** A mean
would let one pathological construct abort a good reference; a median over twelve will not,
while a broken *reference* moves all twelve together, which is the failure this is for.
"""

from __future__ import annotations

import numpy as np

from .checks import twins

#: the carrier set is the top this fraction of the concept profile's squared mass, over
#: (layer, unit) jointly -- the product space, as `ablation_pilot.K_FRAC` has it
K_FRAC = 0.05


def carrier_mask(delta: np.ndarray, k_frac: float = K_FRAC) -> np.ndarray:
    """The top `k_frac` of the squared mass of a concept profile, as a boolean mask."""
    mass = np.asarray(delta, dtype=np.float64) ** 2
    k = max(1, int(round(k_frac * mass.size)))
    flat = mass.ravel()
    cut = np.partition(flat, -k)[-k]
    mask = (mass >= cut)
    # `>=` can admit more than k on ties, which a constant profile makes certain; trim so the
    # overlap denominator is the size asked for rather than the whole array
    if mask.sum() > k:
        idx = np.argsort(flat)[::-1][:k]
        mask = np.zeros(flat.shape, dtype=bool)
        mask[idx] = True
        mask = mask.reshape(mass.shape)
    return mask


def carrier_overlap(delta1: np.ndarray, delta2: np.ndarray,
                    k_frac: float = K_FRAC) -> float:
    """The fraction of twin1's carrier set that twin2 also carries."""
    m1 = carrier_mask(delta1, k_frac)
    m2 = carrier_mask(delta2, k_frac)
    n = int(m1.sum())
    return float((m1 & m2).sum() / n) if n else 0.0


def watched_numbers(profiles, constructs: list[str]) -> dict:
    """The three numbers for one reference, plus the sample they were taken over.

    `profiles(construct, twin, kind)` returns a `Profile` or None — the caller decides where
    the accumulators come from, so this is testable without a record and works whether the
    reference is still on local disk or already in the dataset.

    A construct missing any of its three profiles is skipped and named. **If none of the sample
    can be built the result is `None` for every number and `n_sampled` is zero**, which the
    caller must treat as a failure rather than a pass: that distinction is the whole lesson of
    this item, and of the battery check that printed "matches the record on 0 prompt hashes".
    """
    agree, norms, overlaps, used, skipped = [], [], [], [], []
    for c in constructs:
        a1 = profiles(c, "twin1", "A")
        b1 = profiles(c, "twin1", "B")
        a2 = profiles(c, "twin2", "A")
        b2 = profiles(c, "twin2", "B")
        if a1 is None or b1 is None or a2 is None or b2 is None:
            skipped.append(c)
            continue
        agree.append(twins(a1, a2)["median_cosine"])
        d1 = a1.mean - b1.mean
        d2 = a2.mean - b2.mean
        norms.append(float(np.median(np.linalg.norm(d1, axis=1))))
        overlaps.append(carrier_overlap(d1, d2))
        used.append(c)

    med = lambda xs: float(np.median(xs)) if xs else None   # noqa: E731
    return {"twin_agreement": med(agree),
            "concept_norm": med(norms),
            "carrier_overlap": med(overlaps),
            "n_sampled": len(used),
            "sampled": used,
            "unavailable": skipped,
            "k_frac": K_FRAC,
            "per_construct": {c: {"twin_agreement": a, "concept_norm": n,
                                  "carrier_overlap": o}
                              for c, a, n, o in zip(used, agree, norms, overlaps)}}
