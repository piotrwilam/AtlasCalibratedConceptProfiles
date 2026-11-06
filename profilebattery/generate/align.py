"""AST offsets to token indices — the aligner the handout's BPE policy describes.

The AST gives a construct's position in *characters*; the model reads at a *token* index.
Nothing bridged the two, and the naive bridge is badly wrong: taking the token count of the
lines before the anchor lands on SentencePiece's indentation piece rather than on the
construct, because indentation is split across pieces (`'    total'` tokenises as
`'▁▁▁'` + `'▁total'`). Measured on a sample of A prompts, that put **76% of
construct-token reads on whitespace** — the profiles would have been profiles of indents.

OpenCoder's tokeniser is not a fast tokeniser, so it offers no offset mapping. It does
reconstruct exactly piece by piece (`''.join(p.replace('▁', ' ')) == src`), so spans are
built by accumulating decoded piece lengths.

Policy, per the handout: the anchor token is the token **containing** the construct's first
character, and an exactness flag records whether the construct starts on a token boundary.
No 100%-alignment assertion — BPE merges a keyword with preceding whitespace often enough
that the assertion would simply fire.
"""

from __future__ import annotations

from dataclasses import dataclass

SP_SPACE = "▁"


def piece_text(piece: str) -> str:
    return piece.replace(SP_SPACE, " ")


def token_spans(pieces: list[str]) -> list[tuple[int, int]]:
    """[start, end) character span of every token."""
    spans, pos = [], 0
    for p in pieces:
        n = len(piece_text(p))
        spans.append((pos, pos + n))
        pos += n
    return spans


def reconstructs(pieces: list[str], text: str) -> bool:
    """Whether the pieces rebuild the text exactly — the aligner's precondition."""
    return "".join(piece_text(p) for p in pieces) == text


def token_at(offset: int, spans: list[tuple[int, int]]) -> int:
    """Index of the token containing `offset`; the last token if past the end."""
    for i, (lo, hi) in enumerate(spans):
        if lo <= offset < hi:
            return i
    return len(spans) - 1


def char_offset(text: str, lineno: int, col: int) -> int:
    """Absolute character offset of a (1-indexed line, 0-indexed column)."""
    offs = line_start_offsets(text)
    i = min(max(lineno, 1), len(offs)) - 1
    return min(offs[i] + col, len(text))


@dataclass
class Sites:
    construct_token: int
    span_end: int
    last_token: int
    exact: bool
    n_tokens: int
    reconstructed: bool
    anchor_piece: str


def resolve(text: str, pieces: list[str], anchor_lineno: int, anchor_col: int,
            span_end_lineno: int | None) -> Sites:
    """The three read sites as token indices."""
    spans = token_spans(pieces)
    off = char_offset(text, anchor_lineno, anchor_col)
    i = token_at(off, spans)
    if span_end_lineno is None:
        j = i
    else:
        offs = line_start_offsets(text)
        end_off = (offs[min(span_end_lineno, len(offs)) - 1]
                   + len(lines_of(text)[min(span_end_lineno, len(offs)) - 1]))
        j = token_at(max(end_off - 1, off), spans)
    return Sites(construct_token=i, span_end=j, last_token=len(pieces) - 1,
                 exact=bool(spans[i][0] == off), n_tokens=len(pieces),
                 reconstructed=reconstructs(pieces, text),
                 anchor_piece=pieces[i])


def node_span(node, line_starts: list[int]) -> tuple[int, int]:
    lo = line_starts[node.lineno - 1] + node.col_offset
    end_line = getattr(node, "end_lineno", None) or node.lineno
    hi = line_starts[min(end_line, len(line_starts)) - 1] + \
        (getattr(node, "end_col_offset", None) or node.col_offset)
    return lo, max(hi, lo)


def line_start_offsets(text: str) -> list[int]:
    """Character offset of each line start, counting lines the way Python does.

    `str.splitlines()` breaks on form feed, vertical tab, U+2028 and several other
    characters that are *not* line breaks in Python source — they are whitespace. A padding
    block containing one of them therefore produced a line table one longer than the AST's
    `lineno` numbering, shifting every subsequent anchor. It happened in one prompt of 960,
    which is exactly the rate at which a silent anchor corruption never gets noticed.
    """
    offs, pos = [0], 0
    for line in text.split("\n")[:-1]:
        pos += len(line) + 1
        offs.append(pos)
    return offs


def lines_of(text: str) -> list[str]:
    """Lines as Python counts them — `\n` only."""
    return text.split("\n")


def surface_offset(node, text: str, token: str) -> tuple[int, bool]:
    """Character offset of the construct's own surface token, and whether it was found.

    The AST node's `col_offset` is *not* the surface token for every construct. `for` and
    `def` start their node, so the node offset is right; but an `Assign` node starts at its
    target and a `Call` node at its function, so anchoring on the node offset reads the
    identifier before the `=` or the `(`. The handout asks for the "keyword/symbol offset"
    and this computes it.

    The rule is general rather than per-construct: a construct's own surface token lies
    inside its span but **outside every child node's span**, because the children own their
    own text. So the token is searched for in the node's span minus the union of its
    children's spans, and the node start is the fallback.
    """
    import ast

    starts = line_start_offsets(text)
    lo, hi = node_span(node, starts)
    holes = []
    for child in ast.iter_child_nodes(node):
        if hasattr(child, "lineno"):
            holes.append(node_span(child, starts))
    holes.sort()
    cursor = lo
    for c_lo, c_hi in holes:
        if c_lo > cursor:
            found = text.find(token, cursor, min(c_lo, hi))
            if found >= 0:
                return found, True
        cursor = max(cursor, c_hi)
    if cursor < hi:
        found = text.find(token, cursor, hi)
        if found >= 0:
            return found, True
    return lo, False


#: The surface token of an operator construct is whichever operator the core used, not the
#: construct's primary token. `Compare`'s primary token is `==` because that is what its B
#: set is built around, but a core may carry `!=`, `<`, `in` or `is`; anchoring on `==` then
#: fails to find anything and silently falls back to the node start — which for `Compare` is
#: its left operand. The operator is in the node, so it is read from there.
OPERATOR_SYMBOL = {
    "Add": "+", "Sub": "-", "Mult": "*", "MatMult": "@", "Div": "/", "Mod": "%",
    "Pow": "**", "LShift": "<<", "RShift": ">>", "BitOr": "|", "BitXor": "^",
    "BitAnd": "&", "FloorDiv": "//",
    "And": "and", "Or": "or",
    "Not": "not", "Invert": "~", "UAdd": "+", "USub": "-",
    "Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
    "Is": "is", "IsNot": "is", "In": "in", "NotIn": "not in",
}


def surface_tokens(node, construct: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """Candidate surface tokens for this node, most specific first."""
    import ast

    op = None
    if isinstance(node, (ast.BinOp, ast.BoolOp, ast.UnaryOp)):
        op = type(node.op).__name__
    elif isinstance(node, ast.AugAssign):
        sym = OPERATOR_SYMBOL.get(type(node.op).__name__)
        return ((sym + "=",) if sym else ()) + fallback
    elif isinstance(node, ast.Compare) and node.ops:
        op = type(node.ops[0]).__name__
    if op and op in OPERATOR_SYMBOL:
        return (OPERATOR_SYMBOL[op],) + fallback
    return fallback
