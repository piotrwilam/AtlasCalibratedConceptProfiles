"""The verdict layer rule (go/no-go handout, DECIDED 1).

Read at a layer fixed *before* the run: the construct's v1 home layer if it lies in
1-22, else layer 12. Layer 0 is excluded because it carries the token-embedding floor —
the concept norm there is about three times the deeper layers even untrained, and
`Assert`, whose v1 home is 0, passed two of the three criteria there on a model that had
learned nothing.

This lives here rather than in one experiment because two of them need the same answer:
the verdict itself, and check 9, whose smoothness must be read at the layer the verdict
uses. It was hardcoded to 12 in check 9 at first, which reported smoothness at a layer
eight of the nine verdict constructs do not use.
"""
import json
import os
import pathlib

V1_ATLAS = pathlib.Path(os.path.expanduser(
    "~/Code/No-Deeper-No-Later_private/data/atlas/anneal_iter_0023000.json"))

FALLBACK_LAYER = 12
#: layer 0 is the token-embedding floor; the last layer is excluded for symmetry
VALID_HOME = range(1, 23)


def home_layers(path=V1_ATLAS) -> dict:
    """v1's published specificity-home layer per concept, where one exists."""
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text())["python"]
    return {k.split(":", 1)[1]: v["home"]
            for k, v in d.items() if v.get("home") is not None}


def verdict_layer(construct: str, homes: dict | None = None,
                  fallback: int = FALLBACK_LAYER) -> int:
    homes = home_layers() if homes is None else homes
    home = homes.get(construct)
    return home if home is not None and home in VALID_HOME else fallback
