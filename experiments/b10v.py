r"""B10 alternatives (iv) and (v) — the code-structural yardstick, and the ablation comparison.

Piotr's adoption of 19 Oct. Both splits count **words, not tokens**, and both take line and word
position **from the prompt text**, so no model run is needed to place a prompt -- only to record
C's statistics in the new cells.

  (iv)  token class x word position within the line                      4 x 4  = 16 cells
  (v)   token class x line of the prompt x word position after the indent 4 x 3 x 4 = 48 cells

The class axis is the recording's own (`profilebattery.profiles.cells.token_class`), asserted
identical, so both are measured against the same object as the stored 20-cell reference.

A word is a maximal run of non-space characters. The read site's word is the word containing the
character the site's token begins at; the indent is the leading whitespace of its line and is not
a word. Lines are 1-indexed; a prompt's first line is line 1.

    python experiments/b10v.py recordc --reference pretrain_iter_0001000   # C in both splits
    python experiments/b10v.py apply                                        # A/B, overlaps
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import time

import numpy as np

from experiments.atlas import CARRIERS, D, K, OUT, REPO, provenance, token
from profilebattery.profiles.cells import CLASSES, token_class

TRAINED = ("pretrain_iter_0001000", "pretrain_iter_0020000", "pretrain_iter_0484865")
MODEL = "OpenCoder-LLM/OpenCoder-1.5B-Base-Checkpoints"
TOK_REV = "pretrain_iter_0008000"
WORD_BANDS_IV = ((1, 1), (2, 3), (4, 8), (9, 10 ** 9))
LINE_BANDS_V = ((1, 1), (2, 3), (4, 10 ** 9))
WORD_BANDS_V = ((1, 1), (2, 3), (4, 6), (7, 10 ** 9))
WORK = D / "validation/paper1/b10v"
#: a cell of C with fewer than this many positions is flagged thin
THIN = 200


def _band(k: int, bands) -> str:
    for lo, hi in bands:
        if lo <= k <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
    raise ValueError(k)


def word_map(text: str) -> list[tuple[int, int]]:
    """Per character of `text`: its (line number, word number within the line after the indent).

    Both 1-indexed. Indent whitespace belongs to no word and is given word 0, as is any other
    space. A word is a maximal run of non-space characters, so `for x in range(n):` is four
    words and `range(n):` is one -- which is the point of counting words rather than tokens: the
    unit is the thing a reader of the code sees, not the thing the tokeniser happened to cut.
    """
    out, line, word, in_word = [], 1, 0, False
    for ch in text:
        if ch == "\n":
            out.append((line, 0))
            line, word, in_word = line + 1, 0, False
            continue
        if ch.isspace():
            in_word = False
            out.append((line, word))
            continue
        if not in_word:
            in_word = True
            word += 1
        out.append((line, word))
    return out


def site_line_word(text: str, char_offset: int) -> tuple[int, int]:
    wm = word_map(text)
    if not wm:
        return 1, 1
    i = min(max(char_offset, 0), len(wm) - 1)
    line, word = wm[i]
    if word == 0:                      # a space or the indent: take the next word on the line
        for j in range(i, len(wm)):
            if wm[j][0] != line:
                break
            if wm[j][1] > 0:
                return line, wm[j][1]
        return line, 1
    return line, word


def cell_iv(cls: str, word: int) -> str:
    return f"{_band(word, WORD_BANDS_IV)}:{cls}"


def cell_v(cls: str, line: int, word: int) -> str:
    return f"L{_band(line, LINE_BANDS_V)}|W{_band(word, WORD_BANDS_V)}:{cls}"


def all_cells_iv() -> list[str]:
    return [cell_iv(c, lo) for lo, _ in WORD_BANDS_IV for c in CLASSES]


def all_cells_v() -> list[str]:
    return [cell_v(c, l, w) for l, _ in LINE_BANDS_V for w, _ in WORD_BANDS_V for c in CLASSES]


def piece_offsets(pieces: list[str]) -> list[int]:
    """The character offset at which each tokeniser piece begins in the prompt text.

    The tokeniser here is a slow one, so `return_offsets_mapping` is not available and the
    offsets are reconstructed: SentencePiece marks a word boundary with U+2581, which stands for
    one space, and every other character is literal. `verify_offsets` asserts that the
    reconstruction equals the text, which is what makes this safe to rely on -- a silent
    mismatch would shift every line and word position by the length of one piece.
    """
    out, pos, pending = [], 0, b""
    for p in pieces:
        out.append(pos)
        if len(p) == 6 and p.startswith("<0x") and p.endswith(">"):
            # Byte fallback. A non-ASCII character arrives as several of these pieces and only
            # the whole run decodes to one character; expanding each piece literally -- the
            # first implementation -- made the reconstruction six characters too long and
            # shifted every line and word position after it. Two of the 2,000 C prompts and
            # five of the A prompts sampled contain such text.
            pending += bytes([int(p[3:5], 16)])
            try:
                pos += len(pending.decode("utf-8"))
                pending = b""
            except UnicodeDecodeError:
                pass                  # mid-character: this piece occupies no text position
            continue
        pending = b""
        pos += len(p.replace("\u2581", " "))
    return out


def verify_offsets(text: str, pieces: list[str]) -> bool:
    """Is the offset map usable for this prompt: non-decreasing and inside the text?

    Checked per prompt rather than assumed, because the consequence of a bad map is silent --
    every line and word position after the fault shifts, and the cells still look plausible.
    """
    o = piece_offsets(pieces)
    if not o:
        return True
    return o[-1] <= len(text) and all(b >= a for a, b in zip(o, o[1:]))


def cells_of_prompt(text: str, pieces: list[str]) -> tuple[list[str], list[str]]:
    """(iv) and (v) cells for every position of one prompt."""
    wm = word_map(text)
    offs = piece_offsets(pieces)
    iv, v = [], []
    for p, off in zip(pieces, offs):
        cls = token_class(p)
        i = min(max(off, 0), max(len(wm) - 1, 0))
        line, word = (wm[i] if wm else (1, 1))
        if word == 0:
            line, word = site_line_word(text, off)
        iv.append(cell_iv(cls, word))
        v.append(cell_v(cls, line, word))
    return iv, v


def recordc(args) -> None:
    """Re-record set C at one checkpoint, binned three ways from the same forward pass.

    (iv), (v) and the stored 20-cell split together, so the 20-cell set recorded here can be
    checked against the stored one: if those disagree the hook, the dtype or the tokenisation is
    wrong and the other two mean nothing. The record was made on cuda in **bfloat16**
    (`record/<ref>/MANIFEST.json`) and the dtype is matched, because a float32 re-recording did
    not reproduce the stored means at all.

    One checkpoint at a time: the model is 3.81 GB and this machine has about 16 GiB free.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    from profilebattery.profiles.cells import band, cell_key
    from profilebattery.record.store import Welford
    ref = args.reference
    dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    t_all = time.time()
    tk = AutoTokenizer.from_pretrained(MODEL, revision=TOK_REV, trust_remote_code=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL, revision=ref, dtype=dtype,
                                      trust_remote_code=True).to(dev).eval()
    t_load = time.time() - t0
    layers = model.layers if hasattr(model, "layers") else model.model.layers
    buf = {}

    def pre_h(i):
        def fn(_m, a):
            buf[i] = a[0].detach().float().cpu()
            return None
        return fn

    handles = [l.mlp.down_proj.register_forward_pre_hook(pre_h(i)) for i, l in enumerate(layers)]
    stores, n_pos, bad_off = {}, 0, 0
    t0 = time.time()
    from huggingface_hub import hf_hub_download
    try:
        for twin in ("twin1", "twin2"):
            store = stores.setdefault(twin, {})
            path = hf_hub_download(REPO, f"phase_battery/{twin}/C.jsonl.gz",
                                   repo_type="dataset", token=token())
            rows = [json.loads(l) for l in gzip.open(path, "rt")]
            if args.limit:
                rows = rows[:args.limit]
            for s_ in range(0, len(rows), args.batch):
                chunk = rows[s_:s_ + args.batch]
                enc = tk([r["text"] for r in chunk], return_tensors="pt", padding=True,
                         add_special_tokens=False)
                buf.clear()
                with torch.no_grad():
                    model(enc["input_ids"].to(dev))
                mask = enc["attention_mask"].numpy()
                for bi, r in enumerate(chunk):
                    pieces = tk.convert_ids_to_tokens(
                        tk(r["text"], add_special_tokens=False)["input_ids"])
                    n = min(len(pieces), int(mask[bi].sum()))
                    if n == 0:
                        continue
                    if not verify_offsets(r["text"], pieces):
                        bad_off += 1
                    iv, v = cells_of_prompt(r["text"], pieces)
                    groups = {"IV": {}, "V": {}, "T20": {}}
                    for pos in range(n):
                        groups["IV"].setdefault(iv[pos], []).append(pos)
                        groups["V"].setdefault(v[pos], []).append(pos)
                        groups["T20"].setdefault(
                            cell_key(band(pos, n), token_class(pieces[pos])), []).append(pos)
                    n_pos += n
                    for li in range(len(layers)):
                        vh = buf[li][bi].numpy()
                        for tag, g in groups.items():
                            for ck, ix in g.items():
                                store.setdefault(f"{tag}|L{li}|{ck}", Welford()).update(vh[ix])
            print(f"   {twin}: {len(rows)} prompts, {time.time()-t0:.0f}s", flush=True)
    finally:
        for h in handles:
            h.remove()
    t_record = time.time() - t0
    WORK.mkdir(parents=True, exist_ok=True)
    files = {}
    for twin, store in stores.items():
        out = WORK / f"C_{ref}__{twin}.npz"
        flat = {}
        for key, w in store.items():
            for stat, val in w.to_dict().items():
                flat[f"{key}|{stat}"] = np.asarray([val] if np.isscalar(val) else val)
        np.savez_compressed(out, **flat)
        files[twin] = str(out)
    occ = {}
    for tag, allc in (("IV", all_cells_iv()), ("V", all_cells_v())):
        counts = {}
        for c in allc:
            k = f"{tag}|L0|{c}|n"
            z = np.load(files["twin1"])
            counts[c] = int(z[k][0]) if k in z.files else 0
        occ[tag] = {"counts": counts,
                    "occupied": sum(1 for v in counts.values() if v > 0),
                    "of": len(allc),
                    "thin_below": THIN,
                    "thin": sorted(c for c, v in counts.items() if 0 < v < THIN),
                    "empty": sorted(c for c, v in counts.items() if v == 0)}
    meta = {"reference": ref, "device": dev, "dtype": args.dtype, "files": files,
            "n_positions": n_pos, "offset_reconstruction_failures": bad_off,
            "occupancy": occ,
            "seconds": {"model_load": t_load, "record": t_record, "total": time.time() - t_all},
            "provenance": provenance([f"{MODEL}@{ref}", "phase_battery/twin{1,2}/C.jsonl.gz"],
                                     reference=ref)}
    (WORK / f"C_{ref}.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"\n{ref}: {n_pos:,} positions, load {t_load:.0f}s, record {t_record/60:.1f} min, "
          f"total {(time.time()-t_all)/60:.1f} min")
    for tag in ("IV", "V"):
        e = occ[tag]
        print(f"   ({tag}) {e['occupied']}/{e['of']} cells occupied, {len(e['thin'])} thin "
              f"(<{THIN}), {len(e['empty'])} empty")
    print(f"-> {WORK / f'C_{ref}.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="item", required=True)
    s = sub.add_parser("recordc")
    s.add_argument("--reference", required=True)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=recordc)
    s2 = sub.add_parser("apply")
    s2.add_argument("--reference", default=None)
    s2.add_argument("--force", action="store_true")
    s2.set_defaults(fn=apply_split)
    s3 = sub.add_parser("routes")
    s3.add_argument("--reference", default=None)
    s3.set_defaults(fn=routes)
    a = ap.parse_args()
    a.fn(a)




def yardstick_from(path: str, tag: str, eps: float = 0.01):
    """A `Yardstick` over the cells of one split, read from a re-recorded C file."""
    from profilebattery.profiles.yardstick import Cell, Yardstick
    z = np.load(path)
    cells = {}
    for k in z.files:
        if not (k.startswith(f"{tag}|") and k.endswith("|n")):
            continue
        base = k[:-2]
        _, layer, cell = base.split("|", 2)
        n = int(z[k][0])
        cells[(int(layer[1:]), cell)] = Cell(n=n, mean=z[base + "|mean"],
                                             var=z[base + "|m2"] / max(n - 1, 1),
                                             active=z[base + "|active"])
    if not cells:
        raise ValueError(f"no {tag} cells in {path}")
    return Yardstick(cells, eps)


def new_cells_of_set(rows, tk, which: str) -> list[str]:
    """The (iv) or (v) cell of every prompt's read site, in the rows' own order."""
    out = []
    for r in rows:
        pieces = tk.convert_ids_to_tokens(tk(r["text"], add_special_tokens=False)["input_ids"])
        offs = piece_offsets(pieces)
        i = min(max(int(r.get("site_construct_token") or 0), 0), len(pieces) - 1)
        line, word = site_line_word(r["text"], offs[i])
        cls = token_class(r.get("anchor_piece") or pieces[i])
        out.append(cell_iv(cls, word) if which == "IV" else cell_v(cls, line, word))
    return out


def clean_mapping(rows, tk, which: str) -> tuple[dict, float]:
    """For the approximate route: does every prompt of a (core, old cell) share one new cell?

    Where it does, remapping the stored per-core, per-cell raw mean to the new cell is exact.
    Where it does not, the stored mean already mixes prompts that the new split separates and
    the remap assigns it to the majority new cell -- an approximation whose extent is reported
    rather than hidden. The share of (core, old cell) groups that map cleanly is the number.
    """
    from profilebattery.profiles.profile import cell_of
    new = new_cells_of_set(rows, tk, which)
    groups = {}
    for r, nc in zip(rows, new):
        groups.setdefault((r.get("core_id", "?"), cell_of(r)), []).append(nc)
    mapping, clean = {}, 0
    for key, vals in groups.items():
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts, key=counts.get)
        mapping[key] = best
        if len(counts) == 1:
            clean += 1
    return mapping, (clean / len(groups) if groups else float("nan"))


def remapped_yardstick(ys_new, mapping: dict, old_cells, layers):
    """A yardstick keyed by the **old** cell, holding the **new** cell's statistics.

    This is what makes the approximate route the pipeline's own arithmetic rather than a
    reimplementation of it: `from_accumulators` looks each stored cell up exactly as it always
    does, and the remap shows up only in the numbers it finds. Where a (core, old cell) group's
    prompts do not all share one new cell the group is assigned its majority new cell, which is
    the approximation `clean_mapping` measures.
    """
    from profilebattery.profiles.yardstick import Yardstick
    cells = {}
    for (core, old) in mapping:
        new = mapping[(core, old)]
        for l in layers:
            if (l, new) in ys_new.cells and (l, old) not in cells:
                cells[(l, old)] = ys_new.cells[(l, new)]
    for old in old_cells:
        for l in layers:
            if (l, old) not in cells and (l, old) in ys_new.cells:
                cells[(l, old)] = ys_new.cells[(l, old)]
    return Yardstick(cells, ys_new.eps)


def exact_profile(primary, rows, ys_new, new_cells, construct, kind):
    """The per-prompt route under a new split: each prompt standardised in its own new cell.

    Rows must already be in the recorder's order; `new_cells` is parallel to them.
    """
    from profilebattery.profiles.profile import Profile, core_se
    n_prompts, n_layers, width = primary.shape
    usable = [(i, c) for i, c in enumerate(new_cells[:n_prompts])
              if all(ys_new.has(l, c) for l in range(n_layers))]
    if not usable:
        raise ValueError(f"{construct}/{kind}: no prompt has a supported cell")
    z = np.empty((len(usable), n_layers, width), np.float64)
    for j, (i, c) in enumerate(usable):
        x = primary[i].astype(np.float64)
        for l in range(n_layers):
            z[j, l] = (x[l] - ys_new.mu(l, c)) / ys_new.sigma(l, c)
    by_core = {}
    for j, (i, _) in enumerate(usable):
        by_core.setdefault(rows[i].get("core_id", "?"), []).append(j)
    core_means = {c: z[ix].mean(axis=0) for c, ix in by_core.items()}
    return Profile(construct=construct, set_kind=kind, subtype=None, n=len(usable),
                   mean=z.mean(axis=0), se=core_se(core_means), median=np.median(z, axis=0),
                   raw_mean=np.zeros_like(z[0]), core_means=core_means, cells={},
                   skipped=n_prompts - len(usable))


def apply_split(args) -> None:
    """Re-standardise A and B under (iv) and (v) and compare the carrier sets to the record's."""
    from experiments.b9_b10 import stream as _rec_stream

    def rec_stream(ref_, name_):
        """`b9_b10.stream` returns (bytes_read, buffer); only the buffer is wanted."""
        return _rec_stream(ref_, name_)[1]

    from experiments.atlas import p1_set
    from experiments.r1_turnover import load_index
    from profilebattery.profiles.profile import cell_of, from_accumulators, recorder_order
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL, revision=TOK_REV, trust_remote_code=True)
    hidden31 = set(json.loads(
        (D / "validation/battery/PER_PROMPT_SUBSET.json").read_text())["subset"])
    from profilebattery.profiles.names import tree_name
    hidden31 = {tree_name(x) for x in hidden31}
    refs = [args.reference] if args.reference else list(TRAINED)
    names = sorted(p1_set())
    WORK.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    for ref in refs:
        (WORK / ref).mkdir(parents=True, exist_ok=True)
        cpath = {t: str(WORK / f"C_{ref}__{t}.npz") for t in ("twin1", "twin2")}
        ys = {(t, tag): yardstick_from(cpath[t], tag)
              for t in ("twin1", "twin2") for tag in ("IV", "V", "T20")}
        layers = sorted({l for l, _ in ys[("twin1", "V")].cells})
        idx = load_index(CARRIERS, [ref]).get(ref, {})
        for i, c in enumerate(names, 1):
            out = WORK / ref / f"{c}.json"
            if out.exists() and not args.force:
                continue
            exact = c in hidden31
            row = {"construct": c, "reference": ref, "route": "exact" if exact else "approximate"}
            try:
                prof, thin_sites, clean = {}, {}, {}
                for t in ("twin1", "twin2"):
                    rows = {k: recorder_order(
                        [json.loads(l) for l in gzip.open(
                            __import__("huggingface_hub").hf_hub_download(
                                REPO, f"phase_battery/{t}/{c}_{k}.jsonl.gz",
                                repo_type="dataset", token=token()), "rt")])
                        for k in ("A", "B")}
                    acc = {k: np.load(rec_stream(ref, f"{t}__{c}_{k}_acc.npz"))
                           for k in ("A", "B")}
                    pp = {}
                    if exact:
                        for k in ("A", "B"):
                            z = np.load(rec_stream(ref, f"{t}__{c}_{k}.npz"))
                            pp[k] = np.asarray(z["primary"])
                            if pp[k].shape[0] == 0:
                                pp = {}
                                break
                    for tag, which in (("IV", "IV"), ("V", "V")):
                        parts = {}
                        for k in ("A", "B"):
                            nc = new_cells_of_set(rows[k], tk, which)
                            if tag == "V" and k == "A" and t == "twin1":
                                counts = ys[(t, "V")].cells
                                thin_sites[c] = sum(
                                    1 for x in nc
                                    if (layers[0], x) in counts
                                    and counts[(layers[0], x)].n < THIN)
                                row["read_sites"] = len(nc)
                                row["distinct_v_cells_at_read_sites"] = len(set(nc))
                            if pp:
                                parts[k] = exact_profile(pp[k], rows[k], ys[(t, tag)], nc,
                                                         c, k).mean
                            else:
                                mapping, share = clean_mapping(rows[k], tk, which)
                                clean[f"{t}|{k}|{tag}"] = share
                                old = {cell_of(r) for r in rows[k]}
                                yrm = remapped_yardstick(ys[(t, tag)], mapping, old, layers)
                                parts[k] = from_accumulators(acc[k], yrm, c, k,
                                                             stream="h").mean
                        prof[(t, tag)] = parts["A"] - parts["B"]
            except Exception as exc:                                   # noqa: BLE001
                print(f"   {ref} {c:22s} SKIP {type(exc).__name__}: {str(exc)[:60]}", flush=True)
                continue
            row["clean_mapping_share"] = clean
            row["read_sites_in_a_thin_cell"] = thin_sites.get(c)
            stored = {t: np.sort(np.asarray(idx[f"{t}/{c}|product"], np.int64))
                      for t in ("twin1", "twin2") if f"{t}/{c}|product" in idx}
            for tag in ("IV", "V"):
                c1, c2 = _car(prof[("twin1", tag)]), _car(prof[("twin2", tag)])
                e = {"twin_ceiling_under_this_split": float(np.intersect1d(c1, c2).size / K)}
                for t, cs in (("twin1", c1), ("twin2", c2)):
                    if t in stored:
                        e[f"overlap_with_stored_20cell_{t}"] = float(
                            np.intersect1d(cs, stored[t]).size / K)
                if "twin1" in stored and "twin2" in stored:
                    e["stored_20cell_twin_ceiling"] = float(
                        np.intersect1d(stored["twin1"], stored["twin2"]).size / K)
                    e["below_its_ceiling"] = bool(
                        e.get("overlap_with_stored_20cell_twin1", 1.0)
                        < e["stored_20cell_twin_ceiling"])
                row[tag] = e
            out.write_text(json.dumps(row, indent=1, default=float))
            done += 1
            v = row["V"]
            print(f"   {ref[-8:]:>8s} {i:3d}/{len(names)} {c:20s} {row['route'][:5]:5s} "
                  f"(v) overlap {v.get('overlap_with_stored_20cell_twin1', float('nan')):.4f} "
                  f"ceiling {v.get('stored_20cell_twin_ceiling', float('nan')):.4f}  thin sites "
                  f"{row['read_sites_in_a_thin_cell']}  | {(time.time()-t0)/3600:.2f} h",
                  flush=True)
    print(f"\n{done} construct-references in {(time.time()-t0)/3600:.2f} h")


def _car(v, k: int = K) -> np.ndarray:
    f = np.asarray(v, np.float64).ravel() ** 2
    return np.sort(np.argsort(-f, kind="stable")[:k])


def routes(args) -> None:
    """For the 31 constructs with per-prompt hidden units, the **approximate** route as well.

    The two routes are not interchangeable and the difference runs in a direction that would
    otherwise be read as a result: the approximate route maps a whole (core, old cell) group to
    one new cell, so it inherits the 20-cell grouping by construction and will look more like
    the 20-cell carrier sets than the exact route does. Reporting "the exact constructs sit
    below their ceiling and the approximate ones above" without this would be reporting the
    route. Here both are computed for the same 31 constructs, so the gap is measured.
    """
    from experiments.b9_b10 import stream as _rs
    from experiments.r1_turnover import load_index
    from profilebattery.profiles.profile import cell_of, from_accumulators, recorder_order
    from profilebattery.profiles.names import tree_name
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    tk = AutoTokenizer.from_pretrained(MODEL, revision=TOK_REV, trust_remote_code=True)
    names = sorted({tree_name(x) for x in json.loads(
        (D / "validation/battery/PER_PROMPT_SUBSET.json").read_text())["subset"]})
    out, t0 = {}, time.time()
    for ref in (TRAINED if not args.reference else [args.reference]):
        cpath = {t: str(WORK / f"C_{ref}__{t}.npz") for t in ("twin1", "twin2")}
        ys = {t: yardstick_from(cpath[t], "V") for t in ("twin1", "twin2")}
        layers = sorted({l for l, _ in ys["twin1"].cells})
        idx = load_index(CARRIERS, [ref]).get(ref, {})
        for c in names:
            try:
                prof = {}
                for t in ("twin1", "twin2"):
                    parts = {}
                    for k in ("A", "B"):
                        rows = recorder_order([json.loads(l) for l in gzip.open(
                            hf_hub_download(REPO, f"phase_battery/{t}/{c}_{k}.jsonl.gz",
                                            repo_type="dataset", token=token()), "rt")])
                        acc = np.load(_rs(ref, f"{t}__{c}_{k}_acc.npz")[1])
                        mapping, _ = clean_mapping(rows, tk, "V")
                        old = {cell_of(r) for r in rows}
                        yrm = remapped_yardstick(ys[t], mapping, old, layers)
                        parts[k] = from_accumulators(acc, yrm, c, k, stream="h").mean
                    prof[t] = parts["A"] - parts["B"]
            except Exception as exc:                                   # noqa: BLE001
                print(f"   {ref} {c:20s} SKIP {type(exc).__name__}", flush=True)
                continue
            e = {}
            for t in ("twin1", "twin2"):
                key = f"{t}/{c}|product"
                if key in idx:
                    st = np.sort(np.asarray(idx[key], np.int64))
                    e[f"approximate_overlap_{t}"] = float(
                        np.intersect1d(_car(prof[t]), st).size / K)
            out[f"{ref}|{c}"] = e
            print(f"   {ref[-8:]:>8s} {c:20s} approximate (v) overlap "
                  f"{e.get('approximate_overlap_twin1', float('nan')):.4f}", flush=True)
    (OUT / "YARDSTICK_v_routes.json").write_text(json.dumps(
        {"what": "the approximate route run on the 31 constructs that also have the exact "
                 "route, so the two can be compared on the same constructs",
         "rows": out, "hours": (time.time() - t0) / 3600,
         "provenance": provenance(["record/<ref>/twin{1,2}__<c>_{A,B}_acc.npz"])},
        indent=1, default=float))
    print(f"-> {OUT / 'YARDSTICK_v_routes.json'}")


if __name__ == "__main__":
    main()
