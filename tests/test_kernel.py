"""The kernel reader, on a synthetic shard — no data root needed."""
import json

import numpy as np
import pytest

from profilebattery.kernel import Kernel


@pytest.fixture
def root(tmp_path):
    ref = "ref0"
    rng = np.random.default_rng(0)
    blob = {}
    for kind, names in (("A", ["For/c0", "For/c1"]),
                        ("B", ["For/B:comment/s1", "For/B:sibling:ListComp/ListComp/c0"])):
        tag = f"twin1|For|{kind}|h"
        blob[f"{tag}|z"] = rng.normal(size=(len(names), 3, 5)).astype(np.float32)
        blob[f"{tag}|groups"] = np.asarray(names)
        blob[f"{tag}|raw"] = rng.normal(size=(3, 5)).astype(np.float32)
        blob[f"{tag}|n"] = np.asarray([720])
    np.savez_compressed(tmp_path / f"{ref}.npz", **blob)
    (tmp_path / "MANIFEST.json").write_text(json.dumps(
        {"acts": ["h"], "site": "token", "references": [{"reference": ref}]}))
    return tmp_path


def test_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        Kernel(tmp_path / "nope")


def test_sets_and_names(root):
    k = Kernel(root)
    assert k.references == ["ref0"]
    assert ("twin1", "For", "A", "h") in k.sets("ref0")
    assert len(k.group_names("ref0", "twin1", "For", "B")) == 2


def test_profile_is_unweighted_mean(root):
    k = Kernel(root)
    g = k.groups("ref0", "twin1", "For", "A")
    assert np.allclose(k.profile("ref0", "twin1", "For", "A"), g.mean(0))


def test_sibling_filter_selects_one_group(root):
    """The filter is what makes group-level storage sufficient: it can be applied after the fact."""
    k = Kernel(root)
    kept = k.groups("ref0", "twin1", "For", "B", drop=":sibling:")
    assert len(kept) == 1
    assert "comment" in k.group_names("ref0", "twin1", "For", "B")[0]


def test_contrast_uses_sibling_free_b(root):
    k = Kernel(root)
    a = k.profile("ref0", "twin1", "For", "A")
    b = k.profile("ref0", "twin1", "For", "B", drop=":sibling:")
    assert np.allclose(k.contrast("ref0", "twin1", "For"), a - b)
    assert np.allclose(k.contrast("ref0", "twin1", "For", layer=1), (a - b)[1])


def test_empty_filter_raises(root):
    with pytest.raises(ValueError):
        Kernel(root).groups("ref0", "twin1", "For", "A", keep="nosuchgroup")


def test_se_is_between_group(root):
    k = Kernel(root)
    g = k.groups("ref0", "twin1", "For", "A")
    assert np.allclose(k.se("ref0", "twin1", "For", "A"),
                       g.std(0, ddof=1) / np.sqrt(len(g)))
