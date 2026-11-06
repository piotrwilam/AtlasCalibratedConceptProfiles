"""Where the released data lives, and which repository it comes from.

The working tree hardcoded one author's home directory in the package and in the shared
`experiments` module every script imports, so an installed copy reached for a directory only
its author had. Two functions and two environment variables let a stranger point the tool at
their own copy:

    export ATLAS_DATA=/path/to/AtlasOpenCoder          # the local tree
    export ATLAS_DATASET=some-account/some-dataset     # the repository to fetch from

`ATLAS_DATA` unset falls back to `./AtlasOpenCoder`, which is what a reader has after cloning
the dataset beside the code. `ATLAS_DATASET` unset falls back to the published dataset.

**Why the repository is a variable and not a literal.** The scripts fetch what the local tree
does not hold, and the account name in a fetch is the author's name. A double-blind
submission ships an anonymous mirror of this code beside an anonymous copy of the data, and a
literal would deanonymise it from inside the source. With the variable, the mirror is the
same files with one value changed.
"""
from __future__ import annotations

import os
import pathlib

ENV = "ATLAS_DATA"
DATASET_ENV = "ATLAS_DATASET"
#: the published dataset, used when ATLAS_DATASET is unset
DEFAULT_DATASET = "piotrwilam/AtlasOpenCoder"


def data_root() -> pathlib.Path:
    """The dataset root: $ATLAS_DATA, else ./AtlasOpenCoder."""
    return pathlib.Path(os.environ.get(ENV, "AtlasOpenCoder")).expanduser()


def dataset_repo() -> str:
    """The Hugging Face dataset to fetch from: $ATLAS_DATASET, else the published one."""
    return os.environ.get(DATASET_ENV, DEFAULT_DATASET)
