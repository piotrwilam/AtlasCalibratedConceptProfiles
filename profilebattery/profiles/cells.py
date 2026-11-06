"""Yardstick cells: (relative-position band x token class).

Decision 2 makes the yardstick per (checkpoint, layer, cell) rather than per site, because
C has no construct token and standardising A's construct-token vectors against C's
*last-token* statistics compares different positions. Bands control for context length;
the token class controls for what the token *is*, which is what actually drives a unit's
ordinary behaviour — a keyword mid-snippet and a trailing dedent are not the same ordinary.

Four classes, because the alternative was to match C's read positions to each construct's
anchor distribution, and that reintroduces a per-construct yardstick — losing the shared-C
property that makes the sigma0 error common across constructs and so cancelling in
construct-to-construct comparisons.
"""

from __future__ import annotations

import keyword

N_BANDS = 5
CLASSES = ("keyword", "identifier", "symbol", "newline-indent")

#: SentencePiece marks a word boundary with U+2581; a piece may be pure whitespace
_SP_SPACE = "▁"


def token_class(piece: str) -> str:
    """Classify one tokeniser piece.

    Works on the decoded piece rather than the id, so it is tokeniser-agnostic: OpenCoder's
    SentencePiece marks boundaries with U+2581 and emits pure-whitespace pieces for
    indentation, which are exactly the trailing-dedent positions the yardstick must not
    confuse with a keyword.
    """
    raw = piece.replace(_SP_SPACE, " ")
    stripped = raw.strip()
    if not stripped:
        return "newline-indent"
    if "\n" in raw and not stripped:
        return "newline-indent"
    if keyword.iskeyword(stripped) or stripped in ("match", "case", "type"):
        return "keyword"
    if stripped.isidentifier():
        return "identifier"
    return "symbol"


def band(position: int, n_tokens: int, n_bands: int = N_BANDS) -> int:
    """Relative-position band, 0-indexed, of a read position in a prompt."""
    if n_tokens <= 1:
        return 0
    return min(n_bands - 1, int(position * n_bands / n_tokens))


def label(pieces: list[str]) -> list[tuple[int, str]]:
    """(band, class) per position for one tokenised prompt."""
    n = len(pieces)
    return [(band(i, n), token_class(p)) for i, p in enumerate(pieces)]


def cell_key(b: int, cls: str) -> str:
    return f"b{b}:{cls}"


def all_cells() -> list[str]:
    return [cell_key(b, c) for b in range(N_BANDS) for c in CLASSES]
