"""Every notebook must ask for the secrets that actually exist.

Colab secrets are named by the user, once, in the sidebar: **`GH_Lazy`** for the GitHub token
and **`HF_TOKEN`** for the HuggingFace one, as cell 2 of `colab_sweep.ipynb` establishes.
A notebook that asks for `GH_TOKEN` instead fails with `SecretNotFoundError` on its first
line of real work -- which is a wasted session start, and happened on 6 Oct to a notebook
written the day before by copying the name from memory rather than from the source notebook.

The names are few and stable, so the guard is a whitelist: anything else is either a typo or a
new secret that the user has to be told to create, and both should fail here rather than in a
GPU session.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

NB = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
#: what the user has actually stored. `HF_TOKEN` is also the name `huggingface_hub` reads from
#: the environment unaided, which is why it is not called `HF_Lazy`.
KNOWN = {"GH_Lazy", "HF_TOKEN"}
PATTERN = re.compile(r"""(?:userdata\.get|secret)\(\s*["']([A-Za-z_0-9]+)["']""")


def _secret_names(path: pathlib.Path) -> set[str]:
    nb = json.loads(path.read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    return set(PATTERN.findall(src))


@pytest.mark.parametrize("path", sorted(NB.glob("*.ipynb")), ids=lambda p: p.name)
def test_only_secrets_that_exist_are_requested(path):
    unknown = _secret_names(path) - KNOWN
    assert not unknown, (
        f"{path.name} asks Colab for {sorted(unknown)}, which is not stored. "
        f"The names are {sorted(KNOWN)} -- see cell 2 of colab_sweep.ipynb.")


def test_the_guard_would_catch_the_wrong_name(tmp_path):
    """A whitelist that admitted anything would pass the whole suite silently."""
    p = tmp_path / "bad.ipynb"
    p.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ['GH = userdata.get("GH_TOKEN")\n']}]}))
    assert _secret_names(p) - KNOWN == {"GH_TOKEN"}
