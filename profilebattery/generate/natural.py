"""The natural sample — real occurrences from held-out corpus files.

`codeparrot-clean-valid` is the one corpus file reserved for this: the calibration tables
were built from train shards precisely so that the sample the generator is *compared against*
has not shaped it.

The window, not the file, is the prompt. A real file is far longer than a battery prompt, so
each occurrence contributes a window sized to the construct's own token-length band from §2.
The window need not parse on its own — a method body does not — because the anchor comes from
a parse of the **whole file**, and the anchor's position inside the window is then character
arithmetic. That is the only reason the natural sample can use real fragments at all.

Recorded per occurrence: repo, path, character offset in the file, the window bounds, and the
three read sites as token indices — the same schema the synthetic sets carry, so the
comparison in check 4 is like for like.
"""

from __future__ import annotations

import ast
import hashlib
import warnings
from dataclasses import dataclass

from .align import (char_offset, line_start_offsets, node_span, resolve, surface_offset,
                    surface_tokens)


@dataclass
class Occurrence:
    construct: str
    repo: str
    path: str
    file_offset: int
    window_start: int
    window_end: int
    text: str
    anchor_lineno: int
    anchor_col: int
    span_end_lineno: int | None
    surface_token: str | None


def parse_file(source: str):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None


def window_for(source: str, offset: int, target_tokens: int,
               chars_per_token: float) -> tuple[int, int]:
    """Line-aligned window around a character offset, sized to a token budget.

    Sized in characters via the measured chars-per-token ratio and then snapped to line
    boundaries, so a window never begins or ends mid-line — a fragment starting mid-statement
    would put the model in a state no real prefix produces.
    """
    budget = int(target_tokens * chars_per_token)
    before = int(budget * 0.6)
    lo = max(0, offset - before)
    lo = source.rfind("\n", 0, lo) + 1 if lo else 0
    hi = min(len(source), lo + budget)
    nl = source.find("\n", hi)
    hi = (nl + 1) if nl != -1 else len(source)
    return lo, hi


def occurrences(source: str, repo: str, path: str, constructs: dict,
                bands: dict, chars_per_token: float,
                tokens_of) -> list[Occurrence]:
    """Every inventory occurrence in one file, as windowed prompts."""
    tree = parse_file(source)
    if tree is None:
        return []
    starts = line_start_offsets(source)
    out: list[Occurrence] = []
    for node in ast.walk(tree):
        name = type(node).__name__
        if name not in constructs:
            continue
        cands = surface_tokens(node, name, tokens_of(name))
        off, found, token = node_span(node, starts)[0], False, None
        for cand in cands:
            o, f = surface_offset(node, source, cand)
            if f:
                off, found, token = o, True, cand
                break
        band = bands.get(name, {}).get("tokens", {})
        target = int(band.get("median") or 60) + 40
        lo, hi = window_for(source, off, target, chars_per_token)
        if not (lo <= off < hi):
            continue
        text = source[lo:hi]
        rel = off - lo
        w_starts = line_start_offsets(text)
        ln = max(i for i, o in enumerate(w_starts, start=1) if o <= rel)
        end_lineno = getattr(node, "end_lineno", None)
        span_end = None
        if end_lineno is not None:
            end_off = node_span(node, starts)[1]
            if lo <= end_off < hi:
                span_end = max(i for i, o in enumerate(w_starts, start=1)
                               if o <= end_off - lo)
        out.append(Occurrence(construct=name, repo=repo, path=path, file_offset=off,
                              window_start=lo, window_end=hi, text=text,
                              anchor_lineno=ln, anchor_col=rel - w_starts[ln - 1],
                              span_end_lineno=span_end, surface_token=token))
    return out


def sites_of(occ: Occurrence, tok) -> dict:
    ids = tok(occ.text, add_special_tokens=False)["input_ids"]
    pieces = [tok.convert_ids_to_tokens(i) for i in ids]
    s = resolve(occ.text, pieces, occ.anchor_lineno, occ.anchor_col, occ.span_end_lineno)
    return {"n_tokens": s.n_tokens, "site_construct_token": s.construct_token,
            "site_span_end": s.span_end, "site_last_token": s.last_token,
            "anchor_exact": s.exact, "anchor_piece": s.anchor_piece,
            "offset_from_start": s.construct_token,
            "offset_from_end": s.n_tokens - 1 - s.construct_token}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf8", "replace")).hexdigest()[:16]
