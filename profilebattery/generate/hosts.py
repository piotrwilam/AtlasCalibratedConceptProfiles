"""Where a non-compound construct sits: the shared `host` axis for families 2-5.

A compound statement is its own home. An expression is not — it must sit somewhere,
and that somewhere is always another construct. So for families 3-5 the host is
*obligatory in kind but free in identity*, which is what makes it an axis rather
than a confound: rotate the host across cores and no single host companion can
dominate the A set. With eight hosts over eight cores each appears at share 0.125,
inside the 0.15 cap the companion audit set for compound statements.

Hosts are ordered so the first six carry no compound-statement companion at all.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .menu import Cell, Menu

IND = "    "


def _assign_rhs(expr, menu, cell, rng):
    return [f"{menu.name(cell.domain, 'total', rng)} = {expr}"]


def _call_arg(expr, menu, cell, rng):
    return [f"{menu.verb(cell.domain, rng)}_one({expr})"]


def _kwarg(expr, menu, cell, rng):
    return [f"{menu.verb(cell.domain, rng)}_one({menu.name(cell.domain, 'plural', rng)}, "
            f"limit={expr})"]


def _list_element(expr, menu, cell, rng):
    return [f"{menu.name(cell.domain, 'plural', rng)}_batch = [{expr}, "
            f"{menu.int_literal(rng)}]"]


def _dict_value(expr, menu, cell, rng):
    return [f"{menu.name(cell.domain, 'singular', rng)}_spec = "
            f"{{{menu.field(cell.domain, rng)!r}: {expr}}}"]


def _augassign_rhs(expr, menu, cell, rng):
    return [f"{menu.name(cell.domain, 'total', rng)} += {expr}"]


def _condition(expr, menu, cell, rng):
    return [f"if {expr}:", IND + f"{menu.verb(cell.domain, rng)}_ok = True"]


def _return_value(expr, menu, cell, rng):
    return [f"def {menu.verb(cell.domain, rng)}_value("
            f"{menu.name(cell.domain, 'plural', rng)}):",
            IND + f"return {expr}"]


#: name -> (renderer, the node types this host's own scaffolding creates)
#:
#: The `creates` set is what makes host selection data-driven: a host may never be
#: used for a construct its own scaffolding produces, or the anchor would land on the
#: scaffolding rather than on the core. `call_arg` for a Call target is the case that
#: made this necessary — the host's call is *outside* the core's, so it wins the
#: outermost-node rule.
HOSTS: dict[str, tuple[Callable, tuple[str, ...]]] = {
    "assign_rhs": (_assign_rhs, ("Assign",)),
    "call_arg": (_call_arg, ("Call",)),
    "kwarg": (_kwarg, ("Call",)),
    "list_element": (_list_element, ("Assign", "List")),
    "dict_value": (_dict_value, ("Assign", "Dict")),
    "augassign_rhs": (_augassign_rhs, ("AugAssign",)),
    "condition": (_condition, ("If",)),
    "return_value": (_return_value, ("FunctionDef", "Return")),
}

#: hosts that drag no compound statement — preferred where a compound companion hurts
CLEAN_HOSTS = tuple(k for k, (_, cr) in HOSTS.items()
                    if not {"If", "FunctionDef"} & set(cr))
ALL_HOSTS = tuple(HOSTS)


#: The character each host places **immediately before** the expression it hosts. This is the
#: second half of the generator change `STORY-Battery` §40 required and the phase battery was
#: built without: *"placing the statement forms at indented positions only, and equalising the
#: leading space between A and B"*.
#:
#: It matters because the read site is a token, not a character. `▁len` (id 2930) and `len`
#: (id 3411) are two vocabulary entries whose input embeddings have cosine **+0.034** -- about
#: what two tokens drawn at random give -- so a profile that pools them is not a profile of
#: anything. Measured on the phase battery before the fix, `len`'s A set read `▁len` 150 times,
#: `len` 60 and `=len` 30 *within every scope*: exactly the five space-preceding hosts against
#: `call_arg`'s `(`, `list_element`'s `[` and `kwarg`'s `=`.
#:
#: The remedy is the one `HOST_CREATES` already establishes -- **exclude the host**, rather than
#: insert a space the corpus would not contain. `f( len(xs))` and `limit= len(xs)` are not code
#: anyone writes, and the generator's governing rule is that surroundings are calibrated to the
#: corpus and never chosen for the reading. Five hosts remain, which is enough for the axis.
HOST_PRECEDES: dict[str, str] = {
    "assign_rhs": " ", "call_arg": "(", "kwarg": "=", "list_element": "[",
    "dict_value": " ", "augassign_rhs": " ", "condition": " ", "return_value": " ",
}


def host_keeps_token(host: str, token: str) -> bool:
    """Whether this host leaves the construct's first token piece intact.

    Only identifier-shaped tokens are affected: a leading space is what distinguishes `▁len`
    from `len`, and a symbol token merges with its neighbours whatever precedes it -- that is
    ledger L5 and not something host selection can repair.
    """
    if not token or not token.strip().isidentifier():
        return True
    return HOST_PRECEDES.get(host, " ") == " "


def hosts_for(construct: str, token: str | None = None) -> tuple[str, ...]:
    """Hosts usable for a construct: never one whose scaffolding creates it, and -- when the
    construct's surface token is given -- never one that would change the token the read site
    sees (`HOST_PRECEDES`)."""
    out = tuple(name for name, (_, creates) in HOSTS.items()
                if construct not in creates)
    if token:
        kept = tuple(h for h in out if host_keeps_token(h, token))
        # never strip the axis to nothing: a construct every host would break is a finding for
        # P1 to report, not a reason to fail the build
        if len(kept) >= 2:
            return kept
    return out


def place(host: str, expr: str, menu: Menu, cell: Cell,
          rng: np.random.Generator) -> list[str]:
    return HOSTS[host][0](expr, menu, cell, rng)
