r"""B9 (the readability probes) and B10's per-prompt half, run on the Mac.

Piotr's amendments of 18 Oct, approved: the pass runs here, not on Colab; the A-against-B probe
is reported **as measured, ceiling included**; a **two-role probe** is added -- `For` against each
of the four comprehensions and `If` against `IfExp`, which share the read-site token and so do not
separate on the gross differences that make A-against-B trivial; the hidden units are kept; and
B10 records how many of alternative (iv)'s sixteen within-line cells each construct's read sites
actually occupy.

**The hidden-unit array covers 31 constructs, not 164** (measured from the npz headers at all four
references, 18 Oct): `PER_PROMPT_SUBSET.json`'s planned 32 minus `tuple__builtin` carry
`primary` (720, 24, 6144); the other 133 carry an empty `primary` of shape (0, 24, 1) and the full
`primary_resid` (720, 24, 2240). So every hidden-unit reading here is a 31-construct statement and
every residual reading a 164-construct one, and each row says which.

    python experiments/b9_b10.py pass1     # the long read: A-against-B, and B10 (iv)'s cells
    python experiments/b9_b10.py tworole   # the two-role probe, the seven constructs only

`pass1` is resumable per (reference, construct): it writes one file per construct and skips what
is already there, because the whole pass is 304 GB and a dropped connection must not cost it.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import pathlib
import time

import numpy as np

from experiments.atlas import D, OUT, REPO, provenance, token
from profilebattery.profiles.cells import CLASSES, token_class

REFS = ("untrained", "pretrain_iter_0001000", "pretrain_iter_0020000", "pretrain_iter_0484865")
MODEL = "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints"
MODEL_REV = "pretrain_iter_0008000"
#: alternative (iv): within-line position in tokens since the last newline, times the token
#: class. The bands are the design's; **the classes are the recording's own**, imported rather
#: than restated -- `keyword / identifier / symbol / newline-indent`. CC's first implementation
#: invented a four-class rule (keyword / word / number / symbol) and it was wrong in a way that
#: would have invalidated the comparison: (iv)'s carrier sets are measured against the stored
#: 20-cell reference, so the class axis has to be the same object in both, not merely four of
#: something. Caught after 45 of 676 construct-references and the pass was restarted.
BANDS = ((1, 1), (2, 3), (4, 8), (9, 10 ** 9))
#: the two-role pairs: one token at the read site, two grammatical roles
TWO_ROLE = (("For", "ListComp"), ("For", "GeneratorExp"), ("For", "SetComp"),
            ("For", "DictComp"), ("If", "IfExp"))
WORK = D / "validation/paper1/b9b10"
_FS = None


def band_of(k: int) -> str:
    for lo, hi in BANDS:
        if lo <= k <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
    raise ValueError(k)


def all_cells() -> list[str]:
    return [f"{band_of(lo)}:{c}" for lo, _ in BANDS for c in CLASSES]


def stream(ref: str, name: str):
    """Read one record file over the network, bypassing both the HF cache and the stale local
    tree (A50). The cache is what made the first trial read 1.16 GB at 1,745 MB/s."""
    global _FS
    if _FS is None:
        from huggingface_hub import HfFileSystem
        _FS = HfFileSystem(token=token())
    with _FS.open(f"datasets/{REPO}/record/{ref}/{name}", "rb") as f:
        raw = f.read()
    return len(raw), io.BytesIO(raw)


def recorder_order(rows: list[dict]) -> list[dict]:
    """The prompt rows in the order the **per-prompt arrays** hold them.

    `sweep_anchored` sorts each set by length before batching, to cut padding
    (`profilebattery/record/sweep.py:137`: `sorted(order, key=lambda r: r.get("n_tokens") or
    len(r.get("text", "")))`), and writes the per-prompt vectors in that order. The jsonl is in
    its own order. Pairing row *i* of the jsonl with row *i* of the array is therefore wrong, and
    it is wrong invisibly: the set's mean profile is unaffected, because a mean does not care
    about order, so every check that compares profiles passes while every per-prompt label --
    the core a row belongs to, the cell it lands in -- is attached to the wrong vector.

    The first run of B9 paired them in jsonl order, which scrambled the leave-one-core-out folds.
    Recovered exactly: regrouping under this order reproduces the stored `h|` accumulators at
    cosine 1.000000 for every (core, cell) group at every reference, twin, construct and set
    tested, against a median of 0.96 and a minimum of 0.05 in jsonl order.
    """
    return sorted(rows, key=lambda r: r.get("n_tokens") or len(r.get("text", "")))


def battery(twin: str, c: str, kind: str):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, f"phase_battery/{twin}/{c}_{kind}.jsonl.gz", repo_type="dataset",
                        token=token())
    return recorder_order([json.loads(l) for l in gzip.open(p, "rt")])


def folds_of(core_ids, kind_of_row) -> np.ndarray:
    """Eight held-out folds, keyed by core.

    A has eight cores and B twenty-one, so "leave one core out over the eight cores" needs a
    convention for B: an A row's fold is its own core's index, a B row's fold is its core's index
    modulo eight. Holding out fold k therefore removes one whole A core and about a quarter of
    B's cores, which is the point -- the probe must not be read off a core it was fitted on.
    """
    out = np.empty(len(core_ids), np.int64)
    for kind in ("A", "B"):
        m = kind_of_row == kind
        if not m.any():
            continue
        uniq = {c: i for i, c in enumerate(sorted(set(np.asarray(core_ids)[m].tolist())))}
        out[m] = [uniq[c] % 8 for c in np.asarray(core_ids)[m]]
    return out


def probe(X: np.ndarray, y: np.ndarray, fold: np.ndarray, rng) -> dict:
    """One probe, held out by core, with a size-matched random-label control."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def scaled(tr, te):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        return (X[tr] - mu) / sd, (X[te] - mu) / sd

    acc, auc, ns, iters = [], [], [], []
    margins, margins_correct = [], []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        Xtr, Xte = scaled(tr, te)
        m = LogisticRegression(max_iter=1000, n_jobs=1).fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]
        acc.append(float(((p >= 0.5).astype(int) == y[te]).mean()))
        auc.append(float(roc_auc_score(y[te], p)))
        # A1.4: the signed distance to the boundary, in the probe's own standardised units.
        # Accuracy saturates at 1.0 and stops reporting; the margin keeps moving after it.
        f = m.decision_function(Xte)
        sgn = np.where(y[te] == 1, 1.0, -1.0)
        margins.append(float(np.mean(sgn * f)))
        margins_correct.append(float(np.mean((sgn * f) > 0)))
        ns.append(int(te.sum()))
        iters.append(int(np.max(m.n_iter_)))
    yc = rng.permutation(y)
    k0 = np.unique(fold)[0]
    tr, te = fold != k0, fold == k0
    Xtr, Xte = scaled(tr, te)
    mc = LogisticRegression(max_iter=1000, n_jobs=1).fit(Xtr, yc[tr])
    pc = mc.predict_proba(Xte)[:, 1]
    fc = mc.decision_function(Xte)
    sc = np.where(yc[te] == 1, 1.0, -1.0)
    return {"n_fit": int(X.shape[0]), "d": int(X.shape[1]), "folds": len(acc),
            "n_held_out_per_fold": ns,
            "accuracy": float(np.mean(acc)) if acc else None,
            "accuracy_per_fold": acc,
            "auc": float(np.mean(auc)) if auc else None,
            "auc_per_fold": auc,
            "mean_margin": float(np.mean(margins)) if margins else None,
            "mean_margin_per_fold": margins,
            "share_on_the_right_side": (float(np.mean(margins_correct))
                                        if margins_correct else None),
            "control_accuracy": float(((pc >= 0.5).astype(int) == yc[te]).mean()),
            "control_auc": float(roc_auc_score(yc[te], pc)) if len(np.unique(yc[te])) > 1 else None,
            "control_mean_margin": float(np.mean(sc * fc)),
            "max_lbfgs_iterations": max(iters) if iters else None,
            "hit_the_iteration_cap": bool(iters and max(iters) >= 1000)}


def cells_for(rows, tk) -> tuple[np.ndarray, list]:
    """Alternative (iv)'s cell for every prompt: within-line band x token class at the read site."""
    out, detail = [], []
    for r in rows:
        site = r.get("site_construct_token")
        if site is None:
            site = r.get("site_last_token")
        pieces = tk.convert_ids_to_tokens(tk(r["text"])["input_ids"])
        i0 = max(0, min(int(site), len(pieces) - 1))
        k = i0 + 1
        for i in range(i0, -1, -1):
            if "\n" in (pieces[i] or ""):
                k = i0 - i
                break
        k = max(k, 1)
        cell = f"{band_of(k)}:{token_class(pieces[i0])}"
        out.append(cell)
        detail.append({"within_line_tokens": int(k), "piece": pieces[i0], "cell": cell})
    return np.asarray(out), detail


def one(ref: str, c: str, tk, verdict: int, rng) -> dict:
    t0 = time.time()
    bufs, nbytes = {}, 0
    for twin in ("twin1", "twin2"):
        for kind in ("A", "B"):
            n, b = stream(ref, f"{twin}__{c}_{kind}.npz")
            bufs[(twin, kind)] = b
            nbytes += n
    t_read = time.time() - t0
    arrs = {}
    for k, b in bufs.items():
        with np.load(b) as z:
            arrs[k] = (np.asarray(z["primary"]), np.asarray(z["primary_resid"]))
    bufs.clear()
    has_hidden = all(arrs[k][0].shape[0] > 0 for k in arrs)
    rows = {(t, k): battery(t, c, k) for t in ("twin1", "twin2") for k in ("A", "B")}
    order = [(t, k) for t in ("twin1", "twin2") for k in ("A", "B")]
    y = np.concatenate([np.full(len(rows[o]), 1 if o[1] == "A" else 0) for o in order])
    core_ids = [r["core_id"] for o in order for r in rows[o]]
    kind_of = np.concatenate([np.full(len(rows[o]), o[1]) for o in order])
    fold = folds_of(core_ids, kind_of)
    peak = int(np.argmax(np.linalg.norm(
        arrs[("twin1", "A")][0 if has_hidden else 1].mean(0), axis=1)))

    t0 = time.time()
    b9 = {}
    for space, j in ([("resid_2240", 1)] + ([("hidden_6144", 0)] if has_hidden else [])):
        for lab, L in (("verdict", verdict), ("peak_norm", peak)):
            X = np.concatenate([arrs[o][j][:, L, :].astype(np.float32) for o in order])
            b9[f"{space}|{lab}|L{L}"] = probe(X, y, fold, rng)
    t_b9 = time.time() - t0

    # --- B10 (iv): which of the sixteen cells the read sites occupy -------------------
    t0 = time.time()
    cells, detail = [], []
    for o in order:
        cc, dd = cells_for(rows[o], tk)
        cells.append(cc)
        detail += dd
    cells = np.concatenate(cells)
    occupied = {str(x): int((cells == x).sum()) for x in sorted(set(cells.tolist()))}
    acc_path = None
    if has_hidden:
        # the per-cell hidden accumulation re-standardising under the new yardstick needs,
        # kept so the 304 GB read is paid once
        sums, counts = {}, {}
        off = 0
        for o in order:
            n = arrs[o][0].shape[0]
            sel = cells[off:off + n]
            for x in np.unique(sel):
                m = sel == x
                key = f"{o[0]}|{o[1]}|{x}"
                sums[key] = arrs[o][0][m].astype(np.float32).sum(0)
                counts[key] = int(m.sum())
            off += n
        acc_path = WORK / ref / f"{c}_cells.npz"
        acc_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(acc_path, **{f"sum|{k}": v for k, v in sums.items()},
                            **{f"n|{k}": np.array([v]) for k, v in counts.items()})
    t_b10 = time.time() - t0
    return {"construct": c, "reference": ref, "has_hidden_units": bool(has_hidden),
            "bytes": nbytes, "n_prompts": int(len(y)), "verdict_layer": verdict,
            "peak_norm_layer": peak, "probes": b9,
            "cells_occupied_of_16": len(occupied), "cells": occupied,
            "within_line_tokens_median": float(np.median([d["within_line_tokens"]
                                                          for d in detail])),
            "hidden_cell_accumulation": str(acc_path) if acc_path else None,
            "seconds": {"read": t_read, "b9": t_b9, "b10": t_b10}}


def pass1(args) -> None:
    from profilebattery.profiles.layers import home_layers, verdict_layer
    from profilebattery.profiles.names import tree_name
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV, trust_remote_code=True)
    homes = home_layers()
    rng = np.random.default_rng(20261018)
    names = sorted({tree_name(x) for x in
                    json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]})
    refs = [args.reference] if args.reference else list(REFS)
    WORK.mkdir(parents=True, exist_ok=True)
    t_start, done, gb = time.time(), 0, 0.0
    for ref in refs:
        (WORK / ref).mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(names, 1):
            out = WORK / ref / f"{c}.json"
            if out.exists() and not args.force:
                continue
            try:
                r = one(ref, c, tk, verdict_layer(c, homes), rng)
            except Exception as exc:                                   # noqa: BLE001
                print(f"   {ref} {c:24s} SKIP {type(exc).__name__}: {str(exc)[:60]}", flush=True)
                continue
            out.write_text(json.dumps(r, indent=1, default=float))
            done += 1
            gb += r["bytes"] / 1e9
            el = time.time() - t_start
            p = r["probes"]
            k0 = next(iter(p))
            print(f"   {ref[-7:]} {i:3d}/{len(names)} {c:22s} "
                  f"{'both' if r['has_hidden_units'] else 'res '} "
                  f"{r['bytes']/1e9:.2f} GB  acc {p[k0]['accuracy']:.3f} "
                  f"ctl {p[k0]['control_accuracy']:.3f}  cells {r['cells_occupied_of_16']:2d}/16 "
                  f"| {gb/max(el,1)*1000:5.1f} MB/s  {el/3600:.2f} h", flush=True)
    print(f"\n{done} construct-references, {gb:.1f} GB, {(time.time()-t_start)/3600:.2f} h")
    collect(args)


def collect(args) -> None:
    """Gather the per-construct files into PROBE.json and the B10 cell table."""
    rows = {}
    for ref in REFS:
        d = WORK / ref
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            r = json.loads(p.read_text())
            rows[f"{ref}|{r['construct']}"] = r
    if not rows:
        print("nothing collected")
        return
    hid = sorted({r["construct"] for r in rows.values() if r["has_hidden_units"]})
    occ_ = [r["cells_occupied_of_16"] for r in rows.values()]
    probe_out = {
        "question": "can set A be told from set B by a linear probe on one layer's activations",
        "spaces": {"resid_2240": f"the residual stream, {len(rows)} construct-references",
                   "hidden_6144": f"the MLP hidden units, the {len(hid)} constructs that have "
                                  "per-prompt hidden vectors"},
        "hidden_unit_population": {"n": len(hid), "constructs": hid,
                                   "note": "PER_PROMPT_SUBSET.json's 32 minus tuple__builtin; "
                                           "the other 133 carry an empty primary (0, 24, 1)"},
        "held_out_by": "core, eight folds; an A row's fold is its core index, a B row's its core "
                       "index modulo eight",
        "reported_as_measured": "the A-against-B accuracy is reported ceiling included, per "
                                "Piotr's amendment of 18 Oct; see the two-role probe for the "
                                "reading that does not separate on gross differences",
        "rows": rows, "provenance": provenance(["record/<ref>/twin{1,2}__<c>_{A,B}.npz",
                                                "phase_battery/*"])}
    (OUT / "PROBE.json").write_text(json.dumps(probe_out, indent=1, default=float))
    # --- the yardstick table: which of the sixteen cells each construct actually occupies ---
    # Piotr's reading of 19 Oct is that with this split "each construct uses only four of the
    # sixteen, one per band". It is checked here rather than assumed: a construct occupies more
    # than one class exactly when the piece at its read site is not constant, which is the same
    # condition as the read-site gate of section 7.3, and it occupies fewer than four bands when
    # some within-line position never occurs for it.
    table, exceptions = {}, {}
    for k, r in rows.items():
        occ = r["cells"]
        bands = sorted({c.split(":")[0] for c in occ})
        classes = sorted({c.split(":")[1] for c in occ})
        e = {"n_cells": len(occ), "bands": bands, "classes": classes,
             "counts": occ, "n_prompts": r["n_prompts"],
             "within_line_tokens_median": r["within_line_tokens_median"],
             "one_class_all_four_bands": len(classes) == 1 and len(bands) == 4}
        table[k] = e
        if not e["one_class_all_four_bands"]:
            e["why"] = ("more than one token class at the read site"
                        if len(classes) > 1 else
                        f"only {len(bands)} of the four bands occur")
            exceptions[k] = e
    holds = sum(1 for e in table.values() if e["one_class_all_four_bands"])
    by_construct = {}
    for k, e in table.items():
        by_construct.setdefault(k.split("|")[1], set()).add(len(e["classes"]))
    multiclass = sorted(c for c, v in by_construct.items() if max(v) > 1)
    cells_out = {"alternative": "(iv) within-line band (1 / 2-3 / 4-8 / 9+) x token class "
                                "(keyword / word / number / symbol) = 16 cells",
                 "all_cells": all_cells(),
                 "the_claim_checked": {
                     "claim": "with this split each construct uses only four of the sixteen, "
                              "one per band (Piotr, 19 Oct)",
                     "holds_for": holds, "of": len(table),
                     "share": holds / len(table) if table else None,
                     "constructs_using_more_than_one_token_class": multiclass},
                 "per_construct_reference": table,
                 "exceptions": exceptions,
                 "summary": {"min": int(min(occ_)), "median": float(np.median(occ_)),
                             "max": int(max(occ_)),
                             "distribution": {str(n): int((np.asarray(occ_) == n).sum())
                                              for n in sorted(set(occ_))}},
                 "provenance": provenance(["phase_battery/*"])}
    (OUT / "YARDSTICK_iv_cells.json").write_text(json.dumps(cells_out, indent=1, default=float))
    print(f"-> {OUT / 'PROBE.json'}  ({len(rows)} construct-references, {len(hid)} with hidden)")
    print(f"-> {OUT / 'YARDSTICK_iv_cells.json'}  (cells of 16 occupied: "
          f"min {min(occ_)}, median {np.median(occ_):.0f}, max {max(occ_)})")
    print(f"   'four cells, one per band' holds for {holds} of {len(table)} "
          f"construct-references ({holds/len(table):.1%})")
    if multiclass:
        print(f"   constructs with more than one token class at the read site: "
              f"{', '.join(multiclass)}")


# ---------------------------------------------------------------- the two-role probe
#: Piotr's amendment of 18 Oct. Each pair shares the token at the read site, so the probe cannot
#: separate them on the gross differences -- length, token identity, context -- that make
#: A-against-B trivial. The two controls say what the scale means: same role and shared token
#: should sit at the floor, no shared token at the ceiling.
TWO_ROLE_CONTROLS = (("ListComp", "GeneratorExp", "control: same role, shared token"),
                     ("For", "While", "control: different role, no shared token"))


def _a_sets(ref: str, c: str):
    """Both twins' A arrays and rows for one construct."""
    arrs, rows, nbytes = {}, {}, 0
    for twin in ("twin1", "twin2"):
        n, b = stream(ref, f"{twin}__{c}_A.npz")
        nbytes += n
        with np.load(b) as z:
            arrs[twin] = (np.asarray(z["primary"]), np.asarray(z["primary_resid"]))
        rows[twin] = battery(twin, c, "A")
    return arrs, rows, nbytes


def _pair(ref: str, a: str, b: str, kind: str, cache, layers, rng) -> dict:
    for c in (a, b):
        if c not in cache:
            cache[c] = _a_sets(ref, c)
    (aa, ar, _), (ba, br, _) = cache[a], cache[b]
    has_hidden = all(x[0].shape[0] > 0 for x in list(aa.values()) + list(ba.values()))
    y = np.concatenate([np.ones(sum(len(ar[t]) for t in ("twin1", "twin2"))),
                        np.zeros(sum(len(br[t]) for t in ("twin1", "twin2")))]).astype(int)
    cores = [r["core_id"] for t in ("twin1", "twin2") for r in ar[t]] + \
            [r["core_id"] for t in ("twin1", "twin2") for r in br[t]]
    kinds = np.array(["A"] * int(y.sum()) + ["B"] * int((1 - y).sum()))
    fold = folds_of(cores, kinds)
    out = {}
    for space, j in ([("resid_2240", 1)] + ([("hidden_6144", 0)] if has_hidden else [])):
        for lab, L in layers.items():
            X = np.concatenate([aa[t][j][:, L, :].astype(np.float32) for t in ("twin1", "twin2")]
                               + [ba[t][j][:, L, :].astype(np.float32)
                                  for t in ("twin1", "twin2")])
            out[f"{space}|{lab}|L{L}"] = probe(X, y, fold, rng)
    return {"pair": f"{a} vs {b}", "kind": kind, "has_hidden_units": bool(has_hidden),
            "n": int(len(y)), "probes": out}


def tworole(args) -> None:
    """`For` against each comprehension and `If` against `IfExp`, plus the two controls."""
    from profilebattery.profiles.layers import home_layers, verdict_layer
    homes = home_layers()
    rng = np.random.default_rng(20261018)
    refs = [args.reference] if args.reference else list(REFS)
    res, t0 = {}, time.time()
    for ref in refs:
        cache = {}
        for a, b, kind in ([(x, y_, "two roles, one token at the read site") for x, y_ in TWO_ROLE]
                           + list(TWO_ROLE_CONTROLS)):
            layers = {f"{a}_verdict": verdict_layer(a, homes),
                      f"{b}_verdict": verdict_layer(b, homes)}
            try:
                r = _pair(ref, a, b, kind, cache, layers, rng)
            except Exception as exc:                                   # noqa: BLE001
                print(f"   {ref} {a}/{b}: SKIP {type(exc).__name__}: {str(exc)[:60]}", flush=True)
                continue
            res[f"{ref}|{a}|{b}"] = r
            k0 = next(iter(r["probes"]))
            hid = [v for k, v in r["probes"].items() if k.startswith("hidden")]
            h = f"{hid[0]['accuracy']:.3f}" if hid else "  -  "
            print(f"   {ref[-8:]:>8s} {a:12s} vs {b:14s} resid {r['probes'][k0]['accuracy']:.3f} "
                  f" hidden {h}  ctl {r['probes'][k0]['control_accuracy']:.3f}", flush=True)
        cache.clear()
    out = {"question": "two roles sharing one token at the read site: can a linear probe tell "
                       "them apart, where A-against-B is trivial",
           "pairs": "For against the four comprehensions; If against IfExp",
           "controls": "ListComp/GeneratorExp is the floor (same role, shared token); For/While "
                       "the ceiling (different role, no shared token)",
           "held_out_by": "core, eight folds",
           "hidden_units": "available for the seven two-role constructs, all of which are inside "
                           "the 31 that carry per-prompt hidden vectors; `While` is not, so its "
                           "control row is residual only",
           "rows": res, "hours": (time.time() - t0) / 3600,
           "provenance": provenance(["record/<ref>/twin{1,2}__<c>_A.npz", "phase_battery/*"])}
    (OUT / "PROBE_TWO_ROLE.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {OUT / 'PROBE_TWO_ROLE.json'}  ({len(res)} pair-references, "
          f"{(time.time()-t0)/3600:.2f} h)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="item", required=True)
    for name, fn in (("pass1", pass1), ("tworole", tworole), ("collect", collect)):
        s = sub.add_parser(name)
        s.add_argument("--reference", default=None)
        s.add_argument("--force", action="store_true")
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
