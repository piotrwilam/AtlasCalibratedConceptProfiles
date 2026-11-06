r"""The floor at each diagnostic position: the battery's own conditional entropy (C2/D2, E10).

C2 reads whether a construct's loss has reached its floor. The floor was specified as a
unigram or bigram loss on the proxy corpus, and that is not a floor at all: **an n-gram loss
sits far above a 1.5B model's loss at an obliged position**, so the remaining drop C2 measures
against goes negative and C2's own meaningfulness test would file the obliged class as
*pre-record* for every construct in the battery. Measured at `For`'s obliged `in`: the model's
loss is 0.691 nats at 84B, while a corpus n-gram at that slot is several nats -- the raw entropy
of the battery's own offset-+2 continuations is 3.79, and a corpus estimate is no better.

The floor is the **conditional entropy of the battery's own distribution at that position**, and
it is the right quantity for one reason: the prompts are not sampled from the world, they are
drawn from a distribution this code declares. So the entropy of the continuation at a marked
position is *known*, not estimated. At `For`'s `in` the battery emits `in` 535 times and `,` 171
times -- a floor of **0.554 nats**, which the model's 0.691 sits above, as it must.

**What is implemented and what is not, stated here because the gap is easy to mistake for a
bug.** The field, the `forced` mark, the per-set computation and the write-back are in place.
What is not is the **slot table**: a slot is the position where the grammar *branches*, and that
is role-specific. For `For`'s `in` the branch is after the target's **first component** -- after
`x` in `for x, y in …`, where `,` and `in` are both legal -- and not after the whole target,
whose span already contains the comma. Defined after the whole target, as it is now, the slot
always holds `in` and its entropy is zero: the measured 0.0 is that, not a failure of the idea.
The manual check that gave 0.554 nats looked at the first of {`in`, `,`} after the target's first
token, which is the right slot. One small per-role table, like `OBLIGED_LITERALS`, closes it.

A second fact, and the first draft of this module got it backwards. It observed that a
header's `:` has a piece-level entropy of 0.42 to 1.47 nats where the grammar's was declared
zero, and called the difference a tokenisation artefact of `:` merging into `:\n`, `):\n` and
`":`. The merging is real but it is not the explanation: **a header's `:` is not forced at all**
(Cowork, 1 Oct). After `if x` the condition may continue, after `range(n)` the iterable may --
the model does not know the expression is complete -- so the position admits alternatives and
the entropy there *is* its floor. `If`'s measured 0.912 nats at that position agrees. Forced
positions are a short list: the `(` after a `def` name, and the `:` after `else`, `try`,
`finally` and a bare `except`, where no expression precedes. Everything else is expected and
carries a floor above zero, which makes the floor table more important rather than less.

That same quantity supplies the distinction C2 needed and did not have. A **forced** position has
one legal continuation and an entropy of zero: a header's closing `:`, a call's closing `)`. An
**expected** position has an alternative and a floor above zero: `in` after `for x`, because
`for x, y in …` is legal. Pooling them makes "the loss reached its floor" mean two things.
"""

from __future__ import annotations

import collections
import math

from .positions import SLOT_ALTERNATIVES

#: the share of slot pieces that may carry the preceding expression before a floor is refused
UNCLEAN_TOLERANCE = 0.01


def conditional_entropy(continuations: collections.Counter) -> float:
    """Shannon entropy in nats of the observed continuations at one position."""
    n = sum(continuations.values())
    if n == 0:
        return 0.0
    return -sum((v / n) * math.log(v / n) for v in continuations.values() if v)


def _usable(mark) -> bool:
    """Whether this mark's token index can be trusted.

    `token_spans` reconstructs character spans from piece lengths, and on a few prompts that
    accounting does not reach the end of the text -- 8 of the pilot battery's 21,600, every one
    of them carrying non-ASCII padding. On those the mark's token index points somewhere else
    entirely: one Korean-padded `Assign` prompt put the piece `_id` at the `=` slot, and that
    single prompt in 720 was enough to make the whole role read as not estimable. `positions_for`
    now sets `matched = False` for every mark in such a prompt, so the flag is the filter.

    Not `anchor_exact`, which was tried first and is a different fact: it records whether the
    anchor *piece* equals the surface token, so it is False wherever the tokeniser merges -- 720
    of 720 prompts for twelve constructs. Filtering on it would have discarded most of the
    battery.
    """
    return mark.get("matched", True) is not False


def _is_clean(piece: str, alternatives: tuple[str, ...]) -> bool:
    """Whether the piece at a slot holds the branch and nothing of the expression before it.

    The tokeniser merges a literal with its neighbours, and the two directions are not alike.
    Merging *forward* is harmless: `:\n` and ` in` still begin at the slot, so the piece is the
    branch plus what deterministically follows it. Merging *backward* is fatal: `):\n` and
    `"]:\n` carry the end of the preceding expression into the piece, so the slot is a different
    syntactic position in different prompts and counting across them measures the variety of the
    expressions in front rather than the branch. A piece is clean when one of the role's declared
    alternatives is a *prefix* of it, after the word-boundary marker is normalised to a space.
    """
    s = piece.replace("\u2581", " ")
    return any(s.startswith(a) for a in alternatives)


def floors_for_set(records, pieces_of) -> dict[str, float]:
    """The floor at each role across one A set, keyed by role; None where it is not estimable.

    Computed over the set rather than per prompt, because the floor is a property of the
    *distribution* the generator draws from and a single prompt has no entropy. `pieces_of`
    takes a record's text and returns its token pieces, so this stays free of the tokeniser.

    **A role's floor is only estimable where every observed piece is clean** (above). Where the
    pieces carry the preceding expression -- a header's `:`, which arrives as `):\n`, `]:\n`,
    `"]:\n` as often as `:\n` -- the role's floor is **None**, not a number: the pooled entropy
    there was 1.558 nats for `For` against 0.377 for `If`, and that spread is the variety of the
    iterables in front of the colon, not anything about the branch. C2 then reads that position
    on the loss alone and says so, which is the fallback the contract already provides for.
    """
    seen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in records:
        marks = getattr(r, "positions", None) or []
        if not marks:
            continue
        pieces = pieces_of(r.text)
        for m in marks:
            if m.get("class") != "obliged" or not _usable(m):
                continue
            # the SLOT, not the mark: at the mark the piece is the literal by construction for
            # every role but `in`, and its entropy is zero -- what the first version measured
            idx = m.get("slot")
            if idx is None or idx >= len(pieces):
                continue
            seen[m["role"]][pieces[idx]] += 1
    out: dict[str, float | None] = {}
    for role, counter in seen.items():
        alts = SLOT_ALTERNATIVES.get(role)
        if not alts:
            out[role] = None
            continue
        n = sum(counter.values())
        unclean = sum(v for pc, v in counter.items() if not _is_clean(pc, alts))
        # A share, not an all-or-nothing test. A rare unclean piece is residue -- `Assign` has
        # one in 720 -- while a quarter of them unclean means the slot is several syntactic
        # positions pooled, which is the case the floor cannot survive. The line is drawn at 1%:
        # the observed shares are either below 0.2% or above 10%, with nothing in between, so no
        # role's verdict turns on where in that gap the threshold sits.
        if n and unclean / n <= UNCLEAN_TOLERANCE:
            clean = collections.Counter({pc: v for pc, v in counter.items()
                                         if _is_clean(pc, alts)})
            out[role] = conditional_entropy(clean)
        else:
            out[role] = None
    return out


def slot_report(records, pieces_of) -> dict[str, dict]:
    """Per role: the pieces at its slot, and the unclean share that decides its floor.

    This is what makes a refused floor answerable rather than mysterious -- it names the pieces
    that carry the preceding expression and what fraction of the set they are.
    """
    seen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in records:
        pieces = pieces_of(r.text)
        for m in getattr(r, "positions", None) or []:
            idx = m.get("slot")
            if m.get("class") != "obliged" or idx is None or idx >= len(pieces):
                continue
            if not _usable(m):
                continue
            seen[m["role"]][pieces[idx]] += 1
    out = {}
    for role, counter in seen.items():
        alts = SLOT_ALTERNATIVES.get(role) or ()
        n = sum(counter.values())
        bad = {pc: v for pc, v in counter.items() if not _is_clean(pc, alts)}
        out[role] = {"n": n, "pieces": dict(counter.most_common()),
                     "unclean_share": round(sum(bad.values()) / n, 4) if n else None,
                     "unclean_pieces": dict(sorted(bad.items(), key=lambda kv: -kv[1])[:6])}
    return out


def apply_floors(records, floors: dict[str, float]) -> int:
    """Write each role's floor onto every obliged mark that does not already have one.

    A forced role keeps its zero: the grammar leaves one continuation legal, so its entropy is
    zero by construction and an observed entropy above zero would mean the mark is misplaced --
    which is worth seeing rather than smoothing over, so it is reported and not overwritten.
    """
    n = 0
    for r in records:
        for m in getattr(r, "positions", None) or []:
            if m.get("class") != "obliged":
                continue
            if m.get("forced"):
                m["floor_nats"] = 0.0
                m["observed_entropy"] = floors.get(m["role"])
            else:
                f = floors.get(m["role"])
                m["floor_nats"] = f
                # a None floor is a fact to carry, not a gap: C2 reads this position on the
                # loss alone, and the reason belongs with the mark rather than in a log
                if f is None:
                    m["floor_unavailable"] = "slot pieces carry the preceding expression"
            n += 1
    return n
