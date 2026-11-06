"""R1 / curve 1 -- turnover, and clause C1's threshold on the first half of the split.

**The reading** (`PLAN-Analysis` part 3, curve 1). At checkpoint `t`, the overlap between **twin
1's** carrier set at `t` and **twin 2's** carrier set at the checkpoint nearest **1.5 t** in
tokens, divided by the **geometric mean of the two checkpoints' twin ceilings**. The ceiling at a
checkpoint is the twin-against-twin carrier overlap *there* -- the most two batteries of the same
construct agree with each other -- so the ratio asks how much of the agreement available at those
two checkpoints the construct actually keeps across the step. The product space is primary; the
verdict layer is reported beside it.

Overlap is `|A n B| / |A|`, both sets being the same size by construction, which is the form
`TWINS.json`'s own `carrier_overlap` takes (checked: `For` at `untrained` is 4,962 / 7,373).

**The date is not the common rule.** It is the first checkpoint at which the ratio crosses theta --
the *stability date*. Clause C1 fixes theta as the value for which the share of first-half
constructs with a date inside the record is at least 50%, with 0.70 the expectation and 0.75 the
sensitivity.

**Everything is read from the carrier index**, never from the tree: 61 references x 2 twins x 188
constructs of top-5% `uint32` indices, built once on Colab because the tree costs 7 s per array
by ranged read from here. An overlap is then an `intersect1d` on two sorted index arrays.

**The second half of the split is not read.** C1 is calibrated on `first_half` only.

**The six exploratory constructs are in neither half, and that is why they can be shown.** The
split partitions the **pool of 160** -- `P0 minus the pilot's 30` (`pool_rule`, DESIGN §9) -- and
all six of `For`, `If`, `While`, `ListComp`, `GeneratorExp`, `IfExp` are among those 30 held-out
pilot constructs (checked: P0 190 minus the pool 160 is exactly the 30, and the six are in it).
So illustrating on them reads neither the first half nor the second, and "illustrate on the six"
and "the second half is not read" are not in conflict. `curves_for` builds them separately from
the C1 set so the two cannot be pooled by accident.
"""
from __future__ import annotations
from profilebattery.paths import data_root

import argparse
import json
import os
import pathlib

import numpy as np

from experiments.construct_token_loss import FACTOR_BAND, branches, partner_index
from profilebattery.profiles.names import tree_name

D = data_root()
INDEX = D / "validation/carriers"
SPLIT = D / "validation/battery/SPLIT.json"
SCHEDULE = D / "validation/schedule/CHECKPOINTS.json"
EXPLORATORY = ("For", "If", "While", "ListComp", "GeneratorExp", "IfExp")
THETA_GRID = np.round(np.arange(0.50, 0.96, 0.01), 2)
C1_SHARE = 0.50
THETA = 0.70                    # A31: the contract's fallback, the pilot's rule


def load_index(root: pathlib.Path, refs) -> dict:
    """`{reference: {key: indices}}` for the references asked for."""
    out = {}
    for r in refs:
        p = root / f"{r}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            out[r] = {k: z[k] for k in z.files}
    return out


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """`|A n B| / |A|` on two sorted index arrays; `nan` if either is empty."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.intersect1d(a, b, assume_unique=True).size / a.size)


def ceiling(idx: dict, ref: str, c: str, space: str) -> float:
    """The twin ceiling at one checkpoint: twin 1 against twin 2, there."""
    try:
        return overlap(idx[ref][f"twin1/{c}|{space}"], idx[ref][f"twin2/{c}|{space}"])
    except KeyError:
        return float("nan")


def curve(idx: dict, seq, c: str, space: str = "product") -> dict:
    """The turnover curve for one construct on one branch.

    `seq` is `[(reference, tokens)]` in token order. A checkpoint with no 1.5x partner has no
    point -- the tail is not extrapolated, which is the same rule A30 ruling 2 settled for the
    loss curves.
    """
    refs = [r for r, _ in seq]
    toks = [t for _, t in seq]
    pts = []
    for i, (r, t) in enumerate(seq):
        j = partner_index(toks, i, band=FACTOR_BAND)
        if j is None:
            continue
        a = idx.get(r, {}).get(f"twin1/{c}|{space}")
        b = idx.get(refs[j], {}).get(f"twin2/{c}|{space}")
        if a is None or b is None:
            continue
        raw = overlap(a, b)
        c1, c2 = ceiling(idx, r, c, space), ceiling(idx, refs[j], c, space)
        cap = float(np.sqrt(c1 * c2)) if np.isfinite(c1) and np.isfinite(c2) else float("nan")
        pts.append({"reference": r, "tokens": t, "partner": refs[j],
                    "partner_tokens": toks[j], "raw": raw, "ceiling": cap,
                    "ratio": raw / cap if cap and np.isfinite(cap) and cap > 0
                    else float("nan")})
    return {"construct": c, "space": space, "points": pts}


def stability_date(cur: dict, theta: float, rule: str = "stays") -> dict:
    """The stability date under A31: **crosses theta and stays above it to the end of the record**.

    A31 (13 Oct) amended C1 after this module measured that the literal *crosses* rule dates 41 of
    80 first-half constructs at theta = 0.65 and that **only 7 of those 41 stay above it** -- the
    ratio oscillates across theta rather than settling, so a single crossing is not evidence that
    turnover stopped. A crossing that does not hold is reported as `transient`, with its
    checkpoint, and is **not** a date.

    `rule="crosses"` keeps the pre-A31 reading, used only to reproduce the figures A31 rests on.
    """
    pts = [p for p in cur["points"] if np.isfinite(p["ratio"])]
    if not pts:
        return {"outcome": "unreadable", "why": "no finite point on this curve"}
    first = next((i for i, p in enumerate(pts) if p["ratio"] >= theta), None)
    if first is None:
        return {"outcome": "censored", "why": "the ratio never crosses theta inside the record",
                "max_ratio": max(p["ratio"] for p in pts)}
    held = next((i for i, p in enumerate(pts)
                 if all(q["ratio"] >= theta for q in pts[i:])), None)
    if rule == "crosses":
        p = pts[first]
        return {"outcome": "dated", "index": first, "reference": p["reference"],
                "tokens": p["tokens"], "ratio": p["ratio"], "stayed": held == first}
    if held is None:
        return {"outcome": "transient", "why": "crosses theta but does not stay above it",
                "first_crossing": pts[first]["reference"],
                "first_crossing_tokens": pts[first]["tokens"],
                "max_ratio": max(p["ratio"] for p in pts)}
    p = pts[held]
    return {"outcome": "dated", "index": held, "reference": p["reference"],
            "tokens": p["tokens"], "ratio": p["ratio"], "stayed": True,
            "first_crossing": pts[first]["reference"]}


def ratio_shape(cur: dict, draws: int = 10_000, seed: int = 20261013) -> dict:
    """A31's four additions, so that *stationary* is a measurement and not an inference.

    The range, the level (median), the trend against **log** tokens as a Spearman with a
    **permutation interval** -- the null being the ratio order shuffled against the token order,
    which is the right null for a rank statistic on an ordered series -- and **A30's flatness
    label applied to the ratio itself**, R being the range in noise units.

    The noise unit here is the ratio's own point-to-point spread, `median |diff|`: curve 1 has no
    core bootstrap in this artefact (that comes from the per-core arrays), and a unit taken from
    the curve's own increments is the one quantity available from the carrier index alone. It is
    named in the output so no reading mistakes it for the core-bootstrap SE.
    """
    pts = [p for p in cur["points"] if np.isfinite(p["ratio"])]
    if len(pts) < 3:
        return {"n": len(pts), "why": "fewer than three finite points"}
    r = np.array([p["ratio"] for p in pts], float)
    t = np.log(np.array([p["tokens"] for p in pts], float))
    rho = spearman(r, t)
    rng = np.random.default_rng(seed)
    null = np.array([spearman(rng.permutation(r), t) for _ in range(draws)])
    unit = float(np.median(np.abs(np.diff(r))))
    out = {"n": len(pts), "min": float(r.min()), "max": float(r.max()),
           "range": float(r.max() - r.min()), "median": float(np.median(r)),
           "spearman_vs_log_tokens": rho,
           "permutation_interval": [float(np.quantile(null, 0.025)),
                                    float(np.quantile(null, 0.975))],
           "permutation_p": float((np.abs(null) >= abs(rho)).mean()),
           "noise_unit": unit, "noise_unit_is": "median |point-to-point change| of the ratio, "
                                                "not the core-bootstrap SE"}
    if unit > 0:
        R = out["range"] / unit
        out["R"] = R
        out["flatness"] = ("flat" if R <= 1 else "unreadable" if R < 5 else "moving")
    else:
        out["R"] = None
        out["flatness"] = "unreadable"
        out["why_flatness"] = "the ratio never changes between checkpoints"
    return out


def share_dated(curves: dict, theta: float, rule: str = "stays") -> tuple[float, int, int]:
    """`(share, dated, readable)` -- the share of readable curves with a date inside the record.

    `rule` is passed through, because the share is rule-dependent and a share computed under one
    rule and labelled with the other is how C1's own figures would be misreported: the pre-A31
    theta is 0.65 under `crosses` and 0.64 under `stays`, and the two are not interchangeable.
    """
    n = d = 0
    for cur in curves.values():
        r = stability_date(cur, theta, rule)
        if r["outcome"] == "unreadable":
            continue
        n += 1
        d += r["outcome"] == "dated"
    return (d / n if n else float("nan")), d, n


def choose_theta(curves: dict, grid=THETA_GRID, need: float = C1_SHARE,
                 rule: str = "stays") -> dict:
    """C1: the largest theta whose dated share is at least `need`.

    The largest, because theta is a demand on the construct -- a lower theta dates more curves
    trivially -- so the clause is read as the strictest threshold that still reaches half the
    first half. Every grid point's share is returned, so the choice is inspectable and a
    different reading of "such that" can be taken from the same table.
    """
    table = {}
    for th in grid:
        s, d, n = share_dated(curves, float(th), rule)
        table[float(th)] = {"share": s, "dated": d, "readable": n}
    ok = [t for t, v in table.items() if np.isfinite(v["share"]) and v["share"] >= need]
    return {"theta": max(ok) if ok else None, "need": need, "rule": rule, "table": table,
            "why_none": None if ok else f"no theta on the grid reaches a {need:.0%} share"}


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    sd = rx.std() * ry.std()
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / sd) if sd else float("nan")


def rank_agreement(curves: dict, a: float, b: float) -> dict:
    """Spearman between the stability-date orders under two thetas, on curves dated under both."""
    pairs = []
    for c, cur in curves.items():
        da, db = stability_date(cur, a), stability_date(cur, b)
        if da["outcome"] == "dated" and db["outcome"] == "dated":
            pairs.append((c, da["tokens"], db["tokens"]))
    if len(pairs) < 3:
        return {"n": len(pairs), "spearman": None,
                "why": "fewer than three constructs are dated under both thetas"}
    return {"n": len(pairs), "spearman": spearman([p[1] for p in pairs], [p[2] for p in pairs]),
            "theta_a": a, "theta_b": b,
            "dated_only_under_a": sorted(c for c, cur in curves.items()
                                         if stability_date(cur, a)["outcome"] == "dated"
                                         and stability_date(cur, b)["outcome"] != "dated")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--space", default="product", choices=("product", "layer"))
    ap.add_argument("--branch", default="pretrain", choices=("pretrain", "anneal"))
    ap.add_argument("--out", default=str(D / "validation/r1"))
    a = ap.parse_args()

    split = json.loads(SPLIT.read_text())
    first = [tree_name(c) for c in split["first_half"]]
    schedule = json.loads(SCHEDULE.read_text())
    seq = branches(schedule)[a.branch]

    root = pathlib.Path(a.index)
    idx = load_index(root, [r for r, _ in seq])
    seq = [(r, t) for r, t in seq if r in idx]
    print(f"{len(first)} first-half constructs, {len(seq)} {a.branch} checkpoints in the index, "
          f"space={a.space}")

    curves = {c: curve(idx, seq, c, a.space) for c in first}
    # built separately, and never merged into `curves`: C1's share and theta are defined on the
    # first half alone, and the six are held-out pilot constructs outside both halves
    pool = set(split["first_half"]) | set(split["second_half"])
    illus = {}
    for c in EXPLORATORY:
        assert c not in pool, f"{c} is in the split's pool; it cannot be an illustration"
        illus[tree_name(c)] = curve(idx, seq, tree_name(c), a.space)
    npts = {len([p for p in cur["points"] if np.isfinite(p["ratio"])]) for cur in curves.values()}
    print(f"points per curve: {sorted(npts)}")

    ch = choose_theta(curves, rule="crosses")       # the pre-A31 figures A31 rests on
    ch_a31 = choose_theta(curves, rule="stays")     # and the same clause under A31's rule
    print(f"\nA31: theta = {THETA} (the contract's fallback), rule = crosses and stays")
    import collections
    out_counts = collections.Counter(stability_date(c, THETA)["outcome"] for c in curves.values())
    print(f"  outcomes on the first half: {dict(out_counts)}")
    print(f"  no theta satisfies C1's 50% clause under A31's rule: "
          f"theta = {ch_a31['theta']}")
    print(f"\npre-A31, for the record: theta = {ch['theta']} "
          f"(the largest with a dated share >= {C1_SHARE:.0%} under 'crosses')")
    for th in (0.70, 0.75):
        v, w = ch["table"][th], ch_a31["table"][th]
        print(f"  share at {th:.2f}: crosses {v['share']:.3f} "
              f"({v['dated']} of {v['readable']})   stays {w['share']:.3f} ({w['dated']})")
    ra = rank_agreement(curves, 0.70, 0.75)        # under A31's rule
    print(f"  spearman between the date orders at 0.70 and 0.75: "
          f"{ra['spearman'] if ra['spearman'] is None else round(ra['spearman'], 4)} "
          f"on {ra['n']} constructs dated under both")

    shapes = {c: ratio_shape(cur) for c, cur in curves.items()}
    fl = collections.Counter(v.get("flatness", "n/a") for v in shapes.values())
    med = [v["median"] for v in shapes.values() if "median" in v]
    rho = [v["spearman_vs_log_tokens"] for v in shapes.values() if "spearman_vs_log_tokens" in v]
    sig = [v for v in shapes.values() if "permutation_p" in v and v["permutation_p"] < 0.05]
    print(f"\nthe ratio's shape on the first half (A31's four additions):")
    print(f"  flatness labels: {dict(fl)}")
    print(f"  level (median ratio): median {np.median(med):.3f}, "
          f"range {min(med):.3f}-{max(med):.3f}")
    print(f"  trend vs log tokens: median rho {np.median(rho):+.3f}; "
          f"{len(sig)} of {len(shapes)} outside their permutation interval at p<0.05")
    print(f"  of those, rising {sum(1 for v in sig if v['spearman_vs_log_tokens'] > 0)}, "
          f"falling {sum(1 for v in sig if v['spearman_vs_log_tokens'] < 0)}")

    print("\nthe six exploratory constructs:")
    ill = {}
    for c in EXPLORATORY:
        cur = illus[tree_name(c)]
        ill[c] = {"curve": cur, "shape": ratio_shape(cur),
                  "date_070": stability_date(cur, 0.70), "date_075": stability_date(cur, 0.75),
                  "date_070_crosses": stability_date(cur, 0.70, rule="crosses")}
        d7, d75 = ill[c]["date_070"], ill[c]["date_075"]
        pts = [p["ratio"] for p in cur["points"] if np.isfinite(p["ratio"])]
        print(f"  {c:14s} n={len(pts):2d} ratio {min(pts):.3f}-{max(pts):.3f}  "
              f"0.70 -> {d7['outcome']:9s}"
              f"{' @' + str(round(d7['tokens'] / 1e9)) + 'G' if d7['outcome'] == 'dated' else ''}"
              f"   0.75 -> {d75['outcome']:9s}"
              f"{' @' + str(round(d75['tokens'] / 1e9)) + 'G' if d75['outcome'] == 'dated' else ''}")

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"R1_{a.branch}_{a.space}.json").write_text(json.dumps(
        {"branch": a.branch, "space": a.space, "n_first_half": len(first),
         "checkpoints": [r for r, _ in seq], "c1": ch, "rank_agreement": ra,
         "c1_under_a31_rule": ch_a31,
         "theta_a31": THETA, "rule_a31": "crosses and stays above theta to the end",
         "outcomes_a31": dict(out_counts), "shape": shapes,
         "exploratory": ill, "curves": curves}, indent=1, default=float))
    print(f"\n-> {out / f'R1_{a.branch}_{a.space}.json'}")


if __name__ == "__main__":
    main()
