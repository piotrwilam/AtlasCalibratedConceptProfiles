"""The recording sweep — §4.

Two paths, because the sets differ in what is read.

**Anchored sets** (A, B, natural, alternative) are read at the three token indices already
resolved and stored with each prompt. Nothing here recomputes an anchor: all three aligner
defects lived in that recomputation, so the indices travel with the prompt.

**C** is read at *every* position, binned on the fly into the twenty (band x token class)
yardstick cells. Binning in the loop rather than storing is what makes it affordable — C's
1,000 prompts carry ~97,000 read positions per twin, which as per-prompt vectors would be
28 GB per reference against 47 MB as accumulators.

The **untrained reference** is not a checkpoint but a null: the architecture instantiated from
the public config with no training, seed recorded. No released checkpoint is early enough to
serve as an initialisation baseline — the earliest is 4.19B tokens in — and every date
criterion needs calibrating against an untrained model.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..profiles.cells import band, cell_key, token_class
from .hooks import Recorder
from .store import Welford, save_accumulators

REPO = "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints"
UNTRAINED_SEED = 20260920
#: The three read sites the hooks gather, in the column order `set_sites` is given them in.
#: Only the first is accumulated on A and B sets (A21, D1a §286); the tuple stays because the
#: gather is cheap beside the forward pass and `sweep_c` and the tests use the other two.
ALL_SITES = ("token", "span_end", "last")
#: The four checkpoints of D1 at which the residual arm is recorded. **Everywhere else the
#: `resid` stream is not recorded at all** — D1a §286 drops it on A and B sets, and
#: `build_kernel`'s docstring says why it would be unusable even if kept: it is keyed by `set`
#: with no per-core breakdown, so it cannot support the between-core SE the method rests on,
#: and the C pass has no residual yardstick to standardise it against.
RESID_REFS = ("untrained", "pretrain_iter_0001000", "pretrain_iter_0020000",
              "pretrain_iter_0484865")


@dataclass
class Reference:
    name: str
    revision: str | None          # None for the untrained null
    seed: int | None = None


def load_reference(ref: Reference, device: str, dtype):
    from transformers import AutoConfig, AutoModel

    if ref.revision is None:
        cfg = AutoConfig.from_pretrained(REPO, revision="pretrain_iter_0008000")
        torch.manual_seed(ref.seed or UNTRAINED_SEED)
        model = AutoModel.from_config(cfg)
        model = model.to(device=device, dtype=dtype).eval()
        return model, cfg
    model = AutoModel.from_pretrained(REPO, revision=ref.revision, dtype=dtype)
    return model.to(device).eval(), model.config


def _pieces(tok, text: str) -> list[str]:
    return [tok.convert_ids_to_tokens(i)
            for i in tok(text, add_special_tokens=False)["input_ids"]]


@torch.no_grad()
def sweep_anchored(model, tok, rows: list[dict], rec: Recorder, device: str,
                   batch: int = 4, by_scope: bool = False,
                   sites: tuple[str, ...] = ("token",), spread: bool = False,
                   per_prompt: tuple[str, ...] = ("h",)
                   ) -> tuple[dict[str, np.ndarray], dict, dict]:
    """Returns ({stream: [n, layers, width] float32}, accumulators, stats).

    **`per_prompt` is A21 decision 3, which the first runner did not implement at all.** The
    amendment keeps two per-prompt streams at D1's four checkpoints and they do different jobs,
    which CC had conflated and Cowork separated: the **hidden** vectors serve instrument checks
    (convergence with prompt count, length sensitivity, the split-half null) and 32 constructs
    answer those as well as 188, so they are pruned to `PER_PROMPT_SUBSET.json`; the **residual**
    vectors are *a reading* -- C9's readability probe's only possible input, since a linear probe
    trains on raw vectors and an accumulator cannot be split by prompt -- so they are kept for
    all 169 of P1. Pruning them would prune a result on 137 constructs.

    The first return value is a dict and not one array because of that asymmetry: a set can be
    in the residual arm and out of the hidden one, and a tuple position would have had to carry
    an empty array to say so.

    **`sites` and `spread` are amendment A21's option C, which the first runner did not
    implement.** D1a §286 drops three things on A and B sets, and the record was being written
    with all three present: the `span_end` and `last` read sites (nothing reads them —
    `read_site_overlap.py`, the check that justified recording three, works from battery token
    indices and the inventory and never from activations), and the `m2` and `active` fields
    (`from_accumulators` reads `|n` and `|mean` and nothing else; the two spread fields are the
    C yardstick's, and `sweep_c` still passes `spread=True`). Writing them cost 3 sites x 6
    bytes per unit against 1 x 2 — **18x** the bytes A21's 363 GB is arithmetic on, which is
    ~3.6 TB over 59 references against a 1 TB tier.

    The defaults are the amended record, so a caller that says nothing records what the
    contract licenses; the old behaviour is `sites=("token", "span_end", "last"), spread=True`
    and is what the tests use to check the two forms agree on the fields they share.

    `by_scope` adds a second, **additive** family of accumulators keyed by core *and* scope --
    `hs|{core}@{scope}|token|L{i}|{cell}` -- for the sets whose readings need it: the two-use
    sets and their controls `GeneratorExp` and `While` (D1). The role-or-context question
    cannot be asked of the test record at all, because it has no scope key and a key not
    written at recording time cannot be recovered by any later compute.

    Additive rather than a change to the existing key, and that is the point: the `h` family
    stays keyed by core alone, so the core bootstrap of C0.5 still resamples cores and every
    current reading is untouched. `hs` rides the existing `act` axis -- the kernel already
    distinguishes `h` from `mk` -- so a scope-restricted profile is
    `groups(ref, twin, construct, kind, act="hs", keep="@method")` with no new machinery. Only
    the construct-token site is accumulated, since that is the only site any reading uses.
    """
    n_layers = rec.n_layers
    if "resid" in per_prompt and not rec.want_resid:
        raise ValueError("per-prompt residual vectors were asked for from a Recorder that "
                         "does not capture the residual stream; this is C9's only input and "
                         "cannot be recovered later, so it fails loudly rather than silently")
    primary: dict[str, list[np.ndarray]] = {k: [] for k in per_prompt}
    acc: dict[str, Welford] = {}
    overflow: dict[int, int] = {}
    order = [r for r in rows if r.get("site_construct_token") is not None]
    # **Item 89's first lesson: bucket by length.** Padding is per batch, so a batch holding a
    # 40-token prompt and a 400-token one pays the 400 for both. The loss probe measured a 6x
    # throughput difference between sorted and unsorted order on the same prompts. Sorting is
    # safe here because every accumulator is keyed by core and cell and updated by index, so
    # the order prompts arrive in cannot reach any recorded quantity -- the original index is
    # carried so `n_tokens` and the site columns stay attached to their own prompt.
    order = sorted(order, key=lambda r: r.get("n_tokens") or len(r.get("text", "")))
    for s in range(0, len(order), batch):
        chunk = order[s:s + batch]
        enc = tok([r["text"] for r in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        # the yardstick cell of *each* site of each prompt. Keying accumulators by cell is
        # what lets a pooled set meet its own yardstick: standardising a whole set against one
        # cell is wrong wherever its prompts span cells, and at the last-token site they
        # certainly do — a prompt's final token is sometimes a dedent, sometimes a symbol.
        site_cells = []
        for r in chunk:
            pieces = [tok.convert_ids_to_tokens(i) for i in
                      tok(r["text"], add_special_tokens=False)["input_ids"]]
            n = len(pieces)
            site_cells.append([
                cell_key(band(min(r[k], n - 1), n),
                         token_class(pieces[min(r[k], n - 1)]))
                for k in ("site_construct_token", "site_span_end", "site_last_token")])
        # `site_cols`, not `sites`: the parameter of that name says which sites are
        # *accumulated*, and all three are always *gathered* -- the gather is three columns
        # of one tensor and costs nothing beside the forward pass.
        site_cols = torch.tensor([[r["site_construct_token"], r["site_span_end"],
                                   r["site_last_token"]] for r in chunk], dtype=torch.long)
        rec.set_sites(site_cols, device=device)
        model(ids)
        got = rec.collect()
        h = got["h"]
        # `mk` is absent unless the Recorder was asked for it: D1a drops the stream everywhere
        mk = got.get("mk")
        resid = got.get("resid")
        for stream, arr in (("h", h), ("resid", resid)):       # construct-token site
            if stream in primary and arr is not None:
                primary[stream].append(arr[:, :, 0, :])
        # Per (core, site, **layer**). Accumulating across layers would treat a layer as an
        # observation of the same quantity, which it is not: layer 3 and layer 20 run on
        # different scales, and pooling them is what standardisation exists to avoid.
        #
        # Grouped by core within the batch so each accumulator takes one vectorised update
        # rather than one per prompt.
        by_core: dict[str, list[int]] = {}
        for i, r in enumerate(chunk):
            by_core.setdefault(r.get("core_id", "?"), []).append(i)
        if by_scope:
            by_core_scope: dict[str, list[int]] = {}
            for i, r in enumerate(chunk):
                key = f"{r.get('core_id', '?')}@{r.get('scope', '?')}"
                by_core_scope.setdefault(key, []).append(i)
            for key, idxs in by_core_scope.items():
                by_cell_s: dict[str, list[int]] = {}
                for i in idxs:
                    by_cell_s.setdefault(site_cells[i][0], []).append(i)
                for ck, cidx in by_cell_s.items():
                    x = h[cidx, :, 0, :]
                    for layer_i in range(n_layers):
                        acc.setdefault(f"hs|{key}|token|L{layer_i}|{ck}",
                                       Welford(spread=spread)).update(x[:, layer_i, :])
        for core, idxs in by_core.items():
            # the site's own column in `site_cells` and in the gathered tensor, looked up by
            # name rather than by position, so a caller asking for one site cannot silently
            # get another's column
            for site in sites:
                site_idx = ALL_SITES.index(site)
                by_cell: dict[str, list[int]] = {}
                for i in idxs:
                    by_cell.setdefault(site_cells[i][site_idx], []).append(i)
                for ck, cidx in by_cell.items():
                    # **One fancy-indexed copy per (core, site, cell), not 24.**
                    # `h[cidx, layer_i, site_idx, :]` copies its rows out of a 226 MB host
                    # array once per layer; hoisting the row selection makes it once per
                    # cell and slices a contiguous view per layer. Measured together with
                    # the two dropped sites and `spread=False`: 836 ms -> 137 ms per batch
                    # of 128, the same accumulators to the last bit.
                    x = h[cidx, :, site_idx, :]
                    xm = mk[cidx, :, site_idx, :] if mk is not None else None
                    for layer_i in range(n_layers):
                        acc.setdefault(f"h|{core}|{site}|L{layer_i}|{ck}",
                                       Welford(spread=spread)).update(x[:, layer_i, :])
                        if xm is not None:
                            acc.setdefault(f"mk|{core}|{site}|L{layer_i}|{ck}",
                                           Welford(spread=spread)).update(xm[:, layer_i, :])
        if resid is not None:
            by_cell_all: dict[str, list[int]] = {}
            for i in range(len(chunk)):
                by_cell_all.setdefault(site_cells[i][0], []).append(i)
            for ck, cidx in by_cell_all.items():
                x = resid[cidx, :, 0, :]
                for layer_i in range(n_layers):
                    acc.setdefault(f"resid|set|token|L{layer_i}|{ck}",
                                   Welford(spread=spread)).update(x[:, layer_i, :])
        for k, v in rec.cap.overflow.items():
            overflow[k] = overflow.get(k, 0) + v
    out = {k: (np.concatenate(v, axis=0) if v else np.zeros((0, n_layers, 1), dtype=np.float32))
           for k, v in primary.items()}
    # **`per_prompt_shapes`, not `per_prompt`.** `run_sweep` merges this dict over `info`,
    # where `info["per_prompt"]` records which arms were *asked for*; the same key silently
    # replaced the declaration with the shapes and made a verification read "none" on a
    # reference whose arms were all present.
    return out, acc, {"n": len(order), "fp16_overflow_by_layer": overflow,
                      "per_prompt_shapes": {k: list(v.shape) for k, v in out.items()}}


@torch.no_grad()
def sweep_c(model, tok, rows: list[dict], n_layers: int, device: str,
            batch: int = 2, mask_tokens: tuple[str, ...] = ()) -> tuple[dict, dict]:
    """C at every position: the shared yardstick, the m_k yardstick, and the mask subsets.

    Three things are accumulated in one pass, and two of them were missing.

    **m_k needs its own yardstick.** Decision 1 records that m_k's spread is not derivable
    from h's accumulators — the variance of a linear combination needs h's covariance, not its
    diagonal — and that applies to C exactly as it does to A and B. Without m_k over C there
    is no denominator to standardise an m_k profile against, so the v1 bridge has no units.

    **The masked yardstick is accumulated by subtraction.** Decision 2 asks for a per-construct
    yardstick with that construct's own token positions excluded, for the deep tier. Keeping a
    full masked accumulator per construct would be 30 x 24 x 20 Welfords; instead the
    *excluded* subset is accumulated — positions whose piece is that construct's surface token,
    which is a small fraction — and the masked statistics are recovered by removing it from the
    shared one. The excluded subsets are what the z-deflation report needs anyway.
    """
    layers = model.layers if hasattr(model, "layers") else model.model.layers
    store: dict[str, Welford] = {}
    buf_h: dict[int, torch.Tensor] = {}
    buf_mk: dict[int, torch.Tensor] = {}
    handles = []
    mask_set = {t for t in mask_tokens}

    def pre_h(i):
        def fn(_m, args):
            buf_h[i] = args[0].detach().float().cpu()
            return None
        return fn

    def post_mk(i):
        def fn(_m, _args, out):
            t = out[0] if isinstance(out, tuple) else out
            buf_mk[i] = t.detach().float().cpu()
            return None
        return fn

    for i, layer in enumerate(layers):
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre_h(i)))
        handles.append(layer.mlp.register_forward_hook(post_mk(i)))
    try:
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            enc = tok([r["text"] for r in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False)
            buf_h.clear()
            buf_mk.clear()
            model(enc["input_ids"].to(device))
            mask = enc["attention_mask"].numpy()
            for bi, r in enumerate(chunk):
                pieces = _pieces(tok, r["text"])
                n = min(len(pieces), int(mask[bi].sum()))
                plain = [p.replace("\u2581", " ").strip() for p in pieces[:n]]
                by_cell: dict[str, list[int]] = {}
                by_cell_token: dict[tuple[str, str], list[int]] = {}
                for pos in range(n):
                    ck = cell_key(band(pos, n), token_class(pieces[pos]))
                    by_cell.setdefault(ck, []).append(pos)
                    if plain[pos] in mask_set:
                        by_cell_token.setdefault((ck, plain[pos]), []).append(pos)
                for layer_i in range(n_layers):
                    vh = buf_h[layer_i][bi].numpy()
                    vm = buf_mk[layer_i][bi].numpy()
                    for ck, idxs in by_cell.items():
                        store.setdefault(f"C|L{layer_i}|{ck}", Welford()).update(vh[idxs])
                        store.setdefault(f"M|L{layer_i}|{ck}", Welford()).update(vm[idxs])
                    for (ck, tokstr), idxs in by_cell_token.items():
                        store.setdefault(f"X|L{layer_i}|{ck}|{tokstr}",
                                         Welford()).update(vh[idxs])
    finally:
        for h in handles:
            h.remove()
    return store, {"n": len(rows), "cells": len({k.split("|")[2] for k in store}),
                   "masked_tokens": sorted({k.split("|")[3] for k in store
                                            if k.startswith("X|")})}
