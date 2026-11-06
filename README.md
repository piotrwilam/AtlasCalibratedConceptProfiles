# AtlasCalibratedConceptProfiles — v1.0

The **calibrated concept profile**: a way of reading a language model's MLP neurons that is a
measurement rather than a number. It states the population of stimuli it reads over, gives
every number a unit, carries a licence of nine checks, and declares the resolution below which
it cannot separate two checkpoints.

## The papers

This repository is the tool and the scripts of **"Calibrated Concept Profiles: Making the
Reading of a Model's Neurons a Measurement"** (Wilam, 2026; arXiv:[id]).

A second paper reads the same dataset with this instrument — **"Representational Drift in a
Code Model: Behaviour Settles, Which Neurons Carry It Does Not"** (Wilam, 2026;
arXiv:[id]) — and its own scripts are in `AtlasRepresentationalDrift`, which depends on this
repository and not the other way round.

## The dataset this reads

Every script here reads **`AtlasOpenCoder` at tag `v1`** — the 20-cell yardstick. Point
`$ATLAS_DATA` at a checkout of that tag:

```
hf download piotrwilam/AtlasOpenCoder --repo-type dataset --revision v1 \
    --local-dir ./AtlasOpenCoder
export ATLAS_DATA=./AtlasOpenCoder
```

A later tag is a different yardstick and names different neurons for the same
constructs, so the numbers in the paper belong to `v1` and only to `v1`.

## Install

```sh
pip install -e .
export ATLAS_DATA=/path/to/AtlasOpenCoder     # the dataset, cloned beside or anywhere
```

Unset, `ATLAS_DATA` falls back to `./AtlasOpenCoder`. Every module resolves the dataset
through `profilebattery.paths.data_root()`; nothing hardcodes a path.

## Layout

| | |
|---|---|
| `profilebattery/` | the tool: the generator, the record, the profiles and yardstick, the inventory, the nine checks |
| `experiments/` | the scripts that turn the record into the paper's artefacts |
| `tests/` | 23 test modules, 425 tests |

## The scripts

50 scripts in `experiments/`. `atlas.py` carries the method paper's computations as subcommands
`b1`–`b8b`; the rest are one script per artefact. Added since the first draft of this README:
`record_size.py` (the record's size, by reference and by kind of file), `backfill_paper1.py`
(`SET_GRIDS.json`), `check4_twin_units.py`, `band_held.py`, `b9b10_trial.py`,
`count_vocabulary.py`, `audit_recordings.py`, `b10v.py`, `build_natural.py`,
`check6_reweight.py`, `check6_turnover.py`, `checkpoint_inventory.py`, `compute_profiles.py`,
`r2_bootstrap.py`, `r2_twin2.py`, `recheck_per_prompt.py`, `ablation_splits.py`.

## What the scripts can and cannot do

**These rebuild a result or a figure from the dataset alone**, and are what a reader runs:
the `figure_*`, `check*`, `ablation_*`, `b*` and `licence_table*` scripts.

**These made the inputs and cannot be re-run from the release**: `build_battery.py` needs the
generator and, because the per-prompt seeds are salted, could not reproduce the battery it
built; `run_s15.py` and `run_s16.py` need OpenCoder's checkpoints and a GPU. They are published
as the record of how the inputs were made.

## Tests

`pytest tests` — 425 pass. The suite covers the released code; the private repository's other
44 test modules exercise scripts this release does not contain, so they are not shipped.

## Licence

Apache-2.0 for this code. The **dataset** is CC BY 4.0 for its own content, but its
padding snippets are third-party and keep their original licences: see
`AtlasOpenCoder/ATTRIBUTION.md`.

## Papers

The first paper is the method; the second is the finding, and reads the same record. Both are named in full above.
