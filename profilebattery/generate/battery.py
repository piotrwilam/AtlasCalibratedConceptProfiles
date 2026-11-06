"""Assemble, match and persist the battery — handout §3's remaining deliverables.

Three things live here.

**Per-prompt metadata.** Everything §3 requires, so that any later analysis can group by
core, cell, twin, contrast class or validity without re-deriving it — and so the anchor's
*token* indices travel with the prompt rather than being recomputed at record time by
whatever code happens to be running.

**Anchor-offset matching.** The handout matches A and B on the distribution of the read
site's distance from the start and the end, not on total token length. That is the right
invariant: the profile is read *at* the site, so what must agree is where the site sits, and
padding B to equal A's length would move B's site without moving A's. Matching is done by
selecting from a surplus against A's quantiles.

**Pair IDs, and an honest limit.** The handout asks for A/B pairs from the same core where
the role allows. With a shared, token-parameterised B pool the roles do not allow it: B's
carriers are pool shapes, not this construct's cores, which is exactly what makes the pool
economical. Same-core pairs therefore exist only for the *sibling* other-role subtype, where
B is an A core of a construct sharing the token. The paired variant of the concept profile
is available there and nowhere else, and the count is reported rather than assumed.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .align import resolve
from .assemble import assemble, assemble_b
from .menu import Cell, Menu


@dataclass
class Record:
    """One persisted prompt. Field names are the schema."""

    construct: str
    set_kind: str                  # "A" | "B" | "C"
    subtype: str | None
    text: str
    family: str
    core_id: str
    axes: dict
    domain: str
    scope: str
    padding: int
    draw: int
    seed: int
    twin: int
    contrast_class: str | None
    frame_class: str | None
    components: list
    n_tokens: int
    site_construct_token: int | None
    site_span_end: int | None
    site_last_token: int | None
    anchor_exact: bool | None
    anchor_piece: str | None
    surface_token: str | None
    surface_token_found: bool | None
    offset_from_start: int | None
    offset_from_end: int | None
    pair_id: str | None = None
    valid: bool = True
    reason: str = ""
    flags: dict = field(default_factory=dict)
    #: C2's diagnostic positions, one entry per position: class, role, token index, distance
    #: from the construct token, bound name, and the counterpart in the matched control set
    positions: list = field(default_factory=list)
    #: classes this construct's grammar cannot have at all -- *not applicable*, which is not
    #: the same as a count of zero (C2, C0.14)
    positions_na: list = field(default_factory=list)
    #: per-class counts, so a reading need not re-derive them from `positions`
    position_counts: dict = field(default_factory=dict)
    #: a stable identity for this prompt, from its keys and not its text, so it survives
    #: reformatting: the control record points back at it
    prompt_id: str = ""
    #: control records only: the `prompt_id` of the A prompt this controls, and which variant
    #: of C2's control it is -- (i) same name construct removed, (ii) different name construct
    #: kept. `control_fell_back` is true where the table asked for (i) and the body forbade it
    control_of: str | None = None
    control_variant: str | None = None
    control_fell_back: bool = False


def _sites(prompt, tok) -> tuple[dict, list[str]]:
    ids = tok(prompt.text, add_special_tokens=False)["input_ids"]
    pieces = [tok.convert_ids_to_tokens(i) for i in ids]
    if prompt.anchor_lineno is None:
        return ({"n_tokens": len(pieces)}, pieces)
    s = resolve(prompt.text, pieces, prompt.anchor_lineno, prompt.anchor_col,
                prompt.span_end_lineno)
    return ({"n_tokens": s.n_tokens, "site_construct_token": s.construct_token,
             "site_span_end": s.span_end, "site_last_token": s.last_token,
             "anchor_exact": s.exact, "anchor_piece": s.anchor_piece,
             "offset_from_start": s.construct_token,
             "offset_from_end": s.n_tokens - 1 - s.construct_token}, pieces)


def prompt_id(construct: str, kind: str, twin: int, core_id: str, seed: int,
              draw: int) -> str:
    """A stable short identity from a prompt's keys, not its text.

    Keys rather than text because the assembler runs `black` and a control differs from its A
    prompt by design; an identity over text would change with either. Twelve hex characters is
    about 10^-8 collision probability over the battery's 135,000 prompts (arithmetic).
    """
    import hashlib
    key = f"{construct}|{kind}|{twin}|{core_id}|{seed}|{draw}"
    return hashlib.blake2b(key.encode(), digest_size=6).hexdigest()


def control_from(a: Record, tok) -> Record | None:
    """The matched control record for one A record, filling A's `control_index` in place.

    Loss-only: nothing reads its activations, so it carries no yardstick and enters no fit. Its
    positions are A's, carried through the edit by character offset rather than re-found by the
    position finder -- which would give a different and correct answer on the control, since
    variant (ii) renames the binding and variant (i) removes the construct (see `Control`).
    """
    from .controls import build_control, needs_control
    from .align import token_at, token_spans
    from .positions import positions_for

    if a.set_kind != "A" or not needs_control(a.construct):
        return None
    if a.site_construct_token is None:
        return None
    pieces = [tok.convert_ids_to_tokens(i)
              for i in tok(a.text, add_special_tokens=False)["input_ids"]]
    spans = token_spans(pieces)
    if a.site_construct_token >= len(spans):
        return None
    lo, hi = spans[a.site_construct_token]
    ctl = build_control(a.text, a.construct, lo, hi)
    if ctl is None:
        return None
    found = positions_for(a.text, a.construct, pieces, a.site_construct_token) or []
    c_pieces = [tok.convert_ids_to_tokens(i)
                for i in tok(ctl.text, add_special_tokens=False)["input_ids"]]
    c_spans = token_spans(c_pieces)
    marks = []
    for i, p in enumerate(found):
        mapped = ctl.map_offset(p.char)
        idx = None if mapped is None else token_at(mapped, c_spans)
        if i < len(a.positions):
            a.positions[i]["control_index"] = idx
        if idx is not None:
            d = p.as_dict()
            d["token_index"] = idx
            d["control_index"] = a.positions[i]["token_index"] if i < len(a.positions) else None
            marks.append(d)
    counts = {c: sum(1 for m in marks if m["class"] == c) for c in {m["class"] for m in marks}}
    return Record(
        construct=a.construct, set_kind="CTL", subtype=a.subtype, text=ctl.text,
        family=a.family, core_id=a.core_id, axes=dict(a.axes), domain=a.domain,
        scope=a.scope, padding=a.padding, draw=a.draw, seed=a.seed, twin=a.twin,
        contrast_class=a.contrast_class, frame_class=a.frame_class,
        components=list(a.components), n_tokens=len(c_pieces),
        site_construct_token=None, site_span_end=None, site_last_token=len(c_pieces) - 1,
        anchor_exact=None, anchor_piece=None, surface_token=None,
        surface_token_found=None, offset_from_start=None, offset_from_end=None,
        valid=a.valid, reason="",
        positions=marks, positions_na=list(a.positions_na), position_counts=counts,
        prompt_id=prompt_id(a.construct, "CTL", a.twin, a.core_id, a.seed, a.draw),
        control_of=a.prompt_id, control_variant=ctl.variant,
        control_fell_back=ctl.fell_back,
        flags={"control_reason": ctl.reason, "control_substitute": ctl.substitute,
               "control_bound_name": ctl.bound_name})


def _positions_of(prompt, site: dict, pieces: list[str]):
    """C2's diagnostic positions for one prompt, with the classes its grammar cannot have.

    The positions are recovered from the black-formatted text by AST, as the construct's own
    anchor already is: no template can declare a line number that survives the formatter. A B
    or C prompt has no construct node, so it carries no marks -- which is correct, and is why
    the role gap needs the matched control set rather than the B set (C2).
    """
    from .positions import CLASSES, declared_classes, positions_for

    na = sorted(set(CLASSES) - declared_classes(prompt.construct))
    idx = site.get("site_construct_token")
    if idx is None or not pieces or idx >= len(pieces):
        return [], na, {}
    found = positions_for(prompt.text, prompt.construct, pieces, idx)
    if found is None:
        return [], na, {}
    marks = [p.as_dict() for p in found]
    counts = {c: sum(1 for m in marks if m["class"] == c)
              for c in CLASSES if c not in na}
    return marks, na, counts


def record_from(prompt, core, cell: Cell, seed: int, twin: int, kind: str,
                tok, inventory: dict | None = None, subtype: str | None = None) -> Record:
    site, pieces = _sites(prompt, tok)
    inv = (inventory or {}).get(prompt.construct, {})
    marks, na, counts = _positions_of(prompt, site, pieces)
    return Record(
        construct=prompt.construct, set_kind=kind, subtype=subtype, text=prompt.text,
        family=getattr(core, "family", prompt.family), core_id=prompt.core_id,
        axes=dict(prompt.axes), domain=cell.domain, scope=cell.scope,
        padding=cell.padding, draw=cell.draw, seed=seed, twin=twin,
        contrast_class=inv.get("contrast_class"), frame_class=prompt.frame_class,
        components=list(prompt.companions),
        surface_token=prompt.flags.get("surface_token"),
        surface_token_found=prompt.flags.get("surface_token_found"),
        valid=prompt.valid, reason=prompt.reason,
        positions=marks, positions_na=na, position_counts=counts,
        prompt_id=prompt_id(prompt.construct, kind, twin, prompt.core_id, seed, cell.draw),
        flags={k: v for k, v in prompt.flags.items()
               if k not in ("surface_token", "surface_token_found")},
        **{k: site.get(k) for k in
           ("n_tokens", "site_construct_token", "site_span_end", "site_last_token",
            "anchor_exact", "anchor_piece", "offset_from_start", "offset_from_end")})


#: quantiles A's read-site offsets are matched on
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def offset_target(a_records: list[Record]) -> dict:
    starts = [r.offset_from_start for r in a_records if r.offset_from_start is not None]
    ends = [r.offset_from_end for r in a_records if r.offset_from_end is not None]
    if not starts:
        return {}
    return {"start": [float(np.quantile(starts, q)) for q in QUANTILES],
            "end": [float(np.quantile(ends, q)) for q in QUANTILES]}


def match_offsets(surplus: list[Record], target: dict, n: int,
                  n_bins: int = 10, rng=None) -> list[Record]:
    """Select n records whose read-site offsets match A's distribution.

    Stratified on A's own deciles of the from-start offset: bin the surplus by those
    boundaries and draw proportionally, topping up from the nearest non-empty bin when a
    bin is short. A greedy quantile-discrepancy search was the first attempt and is
    O(n^3) — at 720 prompts against an 864 surplus it does not finish.

    The requirement is that the *distributions* agree, not that any prompt is individually
    matched, and the realised discrepancy is reported rather than assumed.
    """
    rng = rng or np.random.default_rng(0)
    pool = [r for r in surplus if r.offset_from_start is not None]
    if not target or not pool:
        return pool[:n]
    edges = np.quantile(target["start"], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    bins: list[list[Record]] = [[] for _ in range(max(len(edges) - 1, 1))]
    for r in pool:
        i = int(np.searchsorted(edges, r.offset_from_start, side="right") - 1)
        bins[min(max(i, 0), len(bins) - 1)].append(r)
    for b in bins:
        rng.shuffle(b)
    want = n // len(bins)
    chosen: list[Record] = []
    for b in bins:
        chosen.extend(b[:want])
    leftovers = [r for b in bins for r in b[want:]]
    rng.shuffle(leftovers)
    chosen.extend(leftovers[: max(0, n - len(chosen))])
    return chosen[:n]


def offset_discrepancy(selected: list[Record], target: dict) -> float:
    """Summed absolute quantile difference between a selection and A — the residual."""
    if not target or not selected:
        return float("nan")
    s = [r.offset_from_start for r in selected if r.offset_from_start is not None]
    e = [r.offset_from_end for r in selected if r.offset_from_end is not None]
    if not s:
        return float("nan")
    return float(sum(abs(np.quantile(s, q) - t)
                     for q, t in zip(QUANTILES, target["start"]))
                 + sum(abs(np.quantile(e, q) - t)
                       for q, t in zip(QUANTILES, target["end"])))


def write(records: list[Record], path) -> dict:
    """One gzipped JSONL per set. Text travels with the metadata; nothing is recomputed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), default=str) + "\n")
    return {"path": str(path), "n": len(records),
            "valid": sum(1 for r in records if r.valid)}
