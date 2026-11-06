"""Reading the calibration corpus — handout §2.

codeparrot-clean, by the 20 Sept decision: a **sample of the pretraining population**,
not the corpus the model read. File overlap is irrelevant; what calibration needs is the
population's marginals, and the model at any checkpoint has itself consumed a large
sample of the same mixture. RefineCode reconstruction from the Stack-V2 manifest is the
registered upgrade — the released `RefineCode-code-corpus-meta` carries no file content.

The **train** shards are the calibration sample. The **validation** shard is held out for
the natural sample of §3, so nothing calibrated here has seen it.
"""

from __future__ import annotations

import gzip
import json
import pathlib
from dataclasses import dataclass
from typing import Iterator

TRAIN_REPO = "codeparrot/codeparrot-clean"
VALID_REPO = "codeparrot/codeparrot-clean-valid"
VALID_FILE = "file-000000000054.json.gz"

#: shards spread across the range rather than the first three: the shards are ordered, so
#: taking a contiguous prefix would be a cluster sample of whatever that ordering is.
CALIBRATION_SHARDS = (1, 18, 36)


@dataclass(frozen=True)
class Doc:
    repo: str
    path: str
    content: str


def read(path: str | pathlib.Path, limit: int | None = None) -> Iterator[Doc]:
    with gzip.open(path, "rt") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield Doc(rec.get("repo_name", ""), rec.get("path", ""),
                      rec.get("content", ""))


def fetch_shard(n: int) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(TRAIN_REPO, f"file-{n:012d}.json.gz", repo_type="dataset")


def fetch_valid() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(VALID_REPO, VALID_FILE, repo_type="dataset")
