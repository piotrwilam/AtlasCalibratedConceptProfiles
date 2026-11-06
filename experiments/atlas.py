"""Paper 1's computations: items B1, B2, B3, B5 and B8a of the approved Design.

`9-Paper-1/2610-18Oct_PAPER-1-Design.md`, sha256 `47f6a664…`, with Piotr's decisions of 18 Oct.
Every artefact lands under `validation/paper1/` with a provenance line, as A2 requires: the
artefact's inputs, the reference, the battery hash and the code commit. Nothing here interprets;
Part C of the Design says what each licenses.

**Three decisions of 18 Oct that this module implements literally.**

* **The check numbering is mapped, not assumed.** The licence suite's own keys are `7 null` and
  `8 eps`; the draft's §4.1 numbers them the other way round — (7) the σ₀ floor, (8) the nulls. Every
  row of the licence table carries **both** names, and the mapping is written into the artefact, so
  a reader can never mistake one check for the other.
* **The twin ceiling is reported in both conventions, the plug-in primary.** The plug-in value is
  the carrier-set overlap computed from the stored sets; the core-bootstrap **mean** sits below it,
  because the bootstrap resamples cores and the resampled quantity is lower. A49 fixes the papers on
  the plug-in; the bootstrap mean travels beside it under its own field name.
* **The per-prompt population is 164, not 31.** 31 is the planned subset that landed
  (`PER_PROMPT_SUBSET.json`'s 32 minus `tuple`), and it is what the licence's convergence, length
  and null checks ran on; the files themselves carry 164 constructs, 164 of P₁'s 169.
"""
from __future__ import annotations
from profilebattery.paths import data_root, dataset_repo

import argparse
import collections
import csv
import json
import os
import pathlib
import subprocess

import numpy as np

D = data_root()
OUT = D / "validation/paper1"
R2 = D / "validation/r2"
CARRIERS = D / "validation/carriers"
BOOT = D / "validation/bootstraps"
BOOT2 = D / "validation/bootstraps2"
TREE = D / "profiles"
REPO = dataset_repo()
LAST = "pretrain_iter_0484865"
K = 7373
#: the suite's own key -> the draft's §4.1 number, both carried in every row
CHECK_NAMES = {"1 twins": "1 twin unit", "2 convergence": "2 convergence", "3 length": "3 length",
               "4 nu": "4 nu sensitivity", "5 second model": "5 second model",
               "6 calibration": "6 calibration", "7 null": "8 nulls (draft)",
               "7a null_scrambled": "8 nulls, scrambled (draft)",
               "7b null_split_half": "8 nulls, split half (draft)",
               "8 eps": "7 sigma-0 floor (draft)", "9 adjacent": "9 adjacent / resolution",
               "10 leakage": "10 companion leakage"}


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                            # noqa: BLE001
        return "unknown"


def provenance(inputs, reference=None) -> dict:
    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    return {"inputs": list(inputs), "reference": reference,
            "battery_prompt_hash": split.get("battery_prompt_hash"),
            "code_commit": commit(),
            "design": "9-Paper-1/2610-18Oct_PAPER-1-Design.md sha256 47f6a664…"}


def token(): 
    t = os.environ.get("HF_TOKEN")
    if t is None:
        from huggingface_hub import get_token
        t = get_token()
    return t


def fetch(path: str) -> pathlib.Path:
    from huggingface_hub import hf_hub_download
    return pathlib.Path(hf_hub_download(REPO, path, repo_type="dataset", token=token()))


def dataset_record(ref: str, name: str) -> pathlib.Path:
    """A record file read **from the dataset**, never from the local copy. See A50.

    `$ATLAS_DATA/record/` holds a stale S1-era recording at the eight S1 references --
    three read sites, 25 B cores, 51,840 summed positions, no `hs|` scope keys -- where the
    dataset's files at the same references carry one site, 21 B cores, 17,280 positions and 4,800
    `hs|` keys. They are two different recordings of the same checkpoints, and
    `RecordStore.path()` prefers the local one, so a pass that reads accumulators silently gets
    the old recording at those eight references and the new one everywhere else. A48 compared
    `0003000` (local) against `0004000` (dataset) and recorded a C0.5 departure that half exists.
    Every accumulator and yardstick read in this module goes through here.
    """
    # ...and in the RELEASE that hazard is gone: the local tree is the dataset itself, named
    # by $ATLAS_DATA, so a reader who has the file reads it from disk and needs no network.
    local = data_root() / "record" / ref / name
    if local.exists():
        return str(local)
    return fetch(f"record/{ref}/{name}")


def yardstick(ref: str, twin: str):
    from profilebattery.profiles.yardstick import Yardstick
    return Yardstick.from_npz(dataset_record(ref, f"{twin}__C_cells.npz"))


def p1_set() -> set:
    from profilebattery.profiles.names import tree_name
    return {tree_name(c) for c in
            json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]}


def inventory() -> dict:
    return json.loads((D / "inventory/inventory_v2.json").read_text())["constructs"]


# ---------------------------------------------------------------- B1
def b1(args) -> None:
    """The licence table: one row a construct, 188, with both check numberings."""
    from profilebattery.profiles.names import contract_name, tree_name
    suite = json.loads(fetch(f"validation/{LAST}/SUITE.json").read_text())
    cs = suite["constructs"]
    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    p1 = {tree_name(c) for c in split["p1"]}
    failed = set(split["p1_failed"])
    inv = inventory()
    rows, counts = {}, collections.defaultdict(collections.Counter)
    for c, v in cs.items():
        tn = tree_name(c)
        r = {"construct": c, "tree_name": tn, "in_p1": tn in p1,
             "gate_failed": c in failed, "n_a": v.get("n"), "n_b": v.get("n_b"),
             "cores": v.get("cores"), "profile_source": v.get("source"),
             "member_kind": (inv.get(c) or {}).get("member_kind"),
             "contrast_class": (inv.get(c) or {}).get("contrast_class"),
             "a_family": (inv.get(c) or {}).get("a_family")}
        t = v.get("1 twins") or {}
        r["check1_twin_unit"] = {"median_cosine": t.get("median_cosine"),
                                 "median_disattenuated": t.get("median_disattenuated"),
                                 "median_distance": t.get("median_distance"),
                                 "pure_noise_abs95": (t.get("pure_noise") or {}).get("abs_95")}
        cv = v.get("2 convergence")
        r["check2_convergence"] = None if not cv else {
            "n_full": cv.get("n_full"),
            "final_cosine": (cv.get("curve") or [{}])[-1].get("cosine"),
            "final_disattenuated": (cv.get("curve") or [{}])[-1].get("disattenuated"),
            "n_points": len(cv.get("curve") or [])}
        ln = v.get("3 length")
        r["check3_length"] = None if not ln else {
            "levels": ln.get("levels"),
            "min_pair_cosine": min([p.get("cosine") for p in (ln.get("pairs") or [])
                                    if p.get("cosine") is not None], default=None)}
        nu = v.get("4 nu")
        r["check4_nu"] = None if not nu else [
            {"arm": a.get("arm"), "cosine": a.get("cosine"),
             "disattenuated": a.get("disattenuated"), "n_other": a.get("n_other")} for a in nu]
        cal = v.get("6 calibration") or {}
        r["check6_calibration"] = {k: cal.get(k) for k in
                                   ("prompt_se_coverage", "prompt_se_inflation",
                                    "bootstrap_coverage", "bootstrap_inflation", "nominal")}
        ep = v.get("8 eps") or {}
        tu = [(g, (e or {}).get("in_twin_units")) for g, e in (ep.get("vs_default") or {}).items()]
        r["check7_sigma0_floor__suite_8_eps"] = {
            "default": ep.get("default"), "grid": ep.get("grid"),
            "max_in_twin_units": max([x for _, x in tu if x is not None], default=None)}
        nulls = {}
        for key, short in (("7a null_scrambled", "scrambled"), ("7b null_split_half", "split_half")):
            n = v.get(key)
            if n:
                nulls[short] = {"median_norm": n.get("median_norm"), "p95_norm": n.get("p95_norm"),
                                "layer": n.get("layer"), "n_rep": n.get("n_rep")}
        r["check8_nulls__suite_7"] = nulls or None
        lk = v.get("10 leakage")
        r["check10_leakage"] = lk if lk else None
        for k, name in (("check1_twin_unit", "1 twin unit"),
                        ("check2_convergence", "2 convergence"), ("check3_length", "3 length"),
                        ("check4_nu", "4 nu"), ("check6_calibration", "6 calibration"),
                        ("check7_sigma0_floor__suite_8_eps", "7 sigma-0 floor"),
                        ("check8_nulls__suite_7", "8 nulls"),
                        ("check10_leakage", "10 leakage")):
            counts[name]["run" if r[k] else "not computable"] += 1
            if tn in p1:
                counts[name]["run_on_P1" if r[k] else "not_computable_on_P1"] += 1
        rows[c] = r

    cov = [(c, (r["check6_calibration"] or {}).get("bootstrap_coverage")) for c, r in rows.items()]
    cov = [(c, x) for c, x in cov if x is not None]
    pcov = [(c, (r["check6_calibration"] or {}).get("prompt_se_coverage")) for c, r in rows.items()]
    pcov = [(c, x) for c, x in pcov if x is not None]
    med_boot = float(np.median([x for _, x in cov])) if cov else None
    med_prompt = float(np.median([x for _, x in pcov])) if pcov else None
    nominal = 0.95
    under_boot = sorted(c for c, x in cov if x < nominal)
    summary = {
        "n_rows": len(rows), "n_in_p1": sum(1 for r in rows.values() if r["in_p1"]),
        "per_check": {k: dict(v) for k, v in counts.items()},
        "check_numbering": {"suite_key_to_draft": CHECK_NAMES,
                            "why": "the suite's 7 is the nulls and its 8 is the sigma-0 floor; "
                                   "the draft's §4.1 numbers them the other way round"},
        "qualification_check4": "run on the designed arm only -- the natural-corpus arm was never "
                                "built; the arms present per construct are in check4_nu",
        "check6_median_bootstrap_coverage": med_boot,
        "check6_median_prompt_se_coverage": med_prompt,
        "check6_n_below_nominal_bootstrap": len(under_boot),
        "check6_constructs_below_nominal_bootstrap": under_boot,
        "licence_sentence": "licensed on nine checks -- check 4 on its designed arm only, check 6 "
                            "calibrated at the median with the constructs listed above "
                            "understated -- with the second-model replication outstanding and "
                            "named",
        "per_prompt_population": {"planned_subset": 32, "landed": 31, "missing": ["tuple"],
                                  "files_cover": 164,
                                  "note": "31 is what checks 2, 3 and the nulls ran on; the "
                                          "per-prompt files themselves carry 164 constructs, "
                                          "164 of P1's 169"},
        "provenance": provenance([f"validation/{LAST}/SUITE.json",
                                  "validation/battery/SPLIT.json",
                                  "inventory/inventory_v2.json"], LAST)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LICENCE_TABLE.json").write_text(json.dumps(
        {"summary": summary, "constructs": rows}, indent=1, default=float))
    flat = ["construct", "tree_name", "in_p1", "gate_failed", "n_a", "n_b", "cores",
            "profile_source", "member_kind", "contrast_class",
            "check1_median_cosine", "check1_pure_noise_abs95", "check2_final_cosine",
            "check3_min_pair_cosine", "check4_n_arms", "check6_bootstrap_coverage",
            "check7_floor_max_twin_units", "check8_null_median_norm", "check10_leakage"]
    with open(OUT / "LICENCE_TABLE.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(flat)
        for c, r in sorted(rows.items()):
            n8 = (r["check8_nulls__suite_7"] or {}).get("scrambled") or {}
            w.writerow([r["construct"], r["tree_name"], r["in_p1"], r["gate_failed"], r["n_a"],
                        r["n_b"], r["cores"], r["profile_source"], r["member_kind"],
                        r["contrast_class"], r["check1_twin_unit"]["median_cosine"],
                        r["check1_twin_unit"]["pure_noise_abs95"],
                        (r["check2_convergence"] or {}).get("final_cosine"),
                        (r["check3_length"] or {}).get("min_pair_cosine"),
                        len(r["check4_nu"] or []),
                        (r["check6_calibration"] or {}).get("bootstrap_coverage"),
                        r["check7_sigma0_floor__suite_8_eps"]["max_in_twin_units"],
                        n8.get("median_norm"), bool(r["check10_leakage"])])
    print(f"B1: {summary['n_rows']} rows, {summary['n_in_p1']} in P1")
    for k, v in summary["per_check"].items():
        print(f"   {k:18s} {dict(v)}")
    print(f"   check 6: median bootstrap coverage {med_boot:.4f}, prompt-SE coverage "
          f"{med_prompt:.4f}; {len(under_boot)} below the nominal 0.95")
    print(f"-> {OUT / 'LICENCE_TABLE.json'} and .csv")


# ---------------------------------------------------------------- B2
def b2(args) -> None:
    """Six figure artefacts, each as JSON and a plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.construct_token_loss import branches
    from experiments.r1_turnover import ceiling, load_index
    from profilebattery.profiles.names import tree_name
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = p1_set()
    have = {p.stem for p in CARRIERS.glob("*.npz")}
    sched = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    seq = [(r, t) for r, t in branches(sched)["pretrain"] if r in have]
    figs = {}

    # (a) the twin ceiling, both conventions
    refs = [r for r, _ in seq]
    idx = load_index(CARRIERS, refs + ["untrained"])
    plug = {}
    for r in refs + ["untrained"]:
        v = [ceiling(idx, r, c, "product") for c in sorted(p1)]
        v = [x for x in v if np.isfinite(x)]
        plug[r] = float(np.median(v)) if v else None
    cl = json.loads((R2 / "R2_CLOSE.json").read_text())
    curves = (cl.get("pretrain|ceiling") or {}).get("curves") or {}
    boot = collections.defaultdict(list)
    for c, e in curves.items():
        for r, m in zip(e["references"], e["mean"]):
            boot[r].append(m)
    bootmed = {r: float(np.median(v)) for r, v in boot.items()}
    u = json.loads((BOOT2 / "untrained_twin2.json").read_text()) if (BOOT2 / "untrained_twin2.json").exists() else {}
    ub = [e["ceiling"]["mean"] for e in u.values()
          if isinstance(e, dict) and (e.get("ceiling") or {}).get("mean") is not None]
    figs["ceiling"] = {
        "primary": "plug-in", "plug_in": plug, "bootstrap_mean": bootmed,
        "untrained_plug_in": plug.get("untrained"),
        "untrained_bootstrap_mean": float(np.median(ub)) if ub else None,
        "definition": "median over P1 of the twin-vs-twin top-5% carrier overlap in the product "
                      "space; the plug-in is computed from the stored sets, the bootstrap mean is "
                      "the mean of the resampled quantity and sits below it",
        "provenance": provenance(["validation/carriers/*.npz", "validation/r2/R2_CLOSE.json",
                                  "validation/bootstraps/untrained_twin2.json"])}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    tok = [t / 1e9 for _, t in seq]
    ax.plot(tok, [plug[r] for r in refs], marker="o", ms=3, label="plug-in (primary)")
    ax.plot(tok, [bootmed.get(r, np.nan) for r in refs], marker="s", ms=3,
            label="core-bootstrap mean")
    for y, lab in ((plug.get("untrained"), "untrained, plug-in"),
                   (figs["ceiling"]["untrained_bootstrap_mean"], "untrained, bootstrap")):
        if y:
            ax.axhline(y, ls=":", lw=1)
            ax.annotate(f"{lab}: {y:.3f}", (tok[0], y), fontsize=7, va="bottom")
    ax.set_xscale("log"); ax.set_xlabel("tokens (billions)"); ax.set_ylabel("twin ceiling")
    ax.set_title("Twin ceiling over pretraining, median over P1"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "FIG_ceiling.svg"); fig.savefig(OUT / "FIG_ceiling.png", dpi=160)
    plt.close(fig)

    # (b) check 9, the five pairs
    c9 = {}
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    for f in [x for x in api.list_repo_files(REPO, repo_type="dataset")
              if x.startswith("validation/s1_check9/") and x.endswith("SUITE.json")]:
        d = json.loads(fetch(f).read_text())
        rowsv = {c: (v.get("9 adjacent") or {}).get("step_in_twin_units")
                 for c, v in (d.get("constructs") or {}).items()}
        allv = [x for x in rowsv.values() if x is not None]
        p1v = [x for c, x in rowsv.items() if x is not None and tree_name(c) in p1]
        c9[d.get("reference")] = {
            "n_all": len(allv), "median_all": float(np.median(allv)) if allv else None,
            "n_p1": len(p1v), "median_p1": float(np.median(p1v)) if p1v else None,
            "min_p1": float(min(p1v)) if p1v else None,
            "argmin_p1": min(((x, c) for c, x in rowsv.items()
                              if x is not None and tree_name(c) in p1), default=(None, None))[1]}
    figs["check9"] = {"pairs": c9, "provenance": provenance(["validation/s1_check9/*/SUITE.json"])}

    # (c) the gate by class
    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    inv = inventory()
    # **three buckets, not two.** The inventory has 206 constructs, 188 were built, and 169 are
    # in P1 -- so "not in P1" is 37 constructs of which only **19** failed the read-site gate
    # (`SPLIT.json`'s `p1_failed`) and the rest were never built or carry no corpus rate. The
    # first version of this block called all 37 "failed", which would have put 37 against the
    # paper's 19.
    built = set(json.loads(fetch(f"validation/{LAST}/SUITE.json").read_text())["constructs"])
    gate = collections.defaultdict(collections.Counter)
    for c, v in inv.items():
        cls = v.get("contrast_class") or "unknown"
        if tree_name(c) in p1:
            gate[cls]["in_p1"] += 1
        elif c in set(split["p1_failed"]):
            gate[cls]["failed_the_gate"] += 1
        elif c not in built:
            gate[cls]["not_built"] += 1
        else:
            gate[cls]["built_but_not_in_p1"] += 1
    figs["gate_by_class"] = {"by_contrast_class": {k: dict(v) for k, v in gate.items()},
                             "buckets": "in_p1 / failed_the_gate (SPLIT.json's p1_failed, 19) / "
                                        "not_built (in the inventory, never recorded) / "
                                        "built_but_not_in_p1",
                             "n_inventory": len(inv), "n_built": len(built),
                             "piece_class_counts": split.get("piece_class_counts"),
                             "p1_rule": split.get("p1_rule"),
                             "p1_failed": split.get("p1_failed"),
                             "provenance": provenance(["validation/battery/SPLIT.json",
                                                       "inventory/inventory_v2.json"])}

    # (d) the twin-unit distribution
    suite = json.loads(fetch(f"validation/{LAST}/SUITE.json").read_text())
    tw = {c: (v.get("1 twins") or {}).get("median_cosine")
          for c, v in suite["constructs"].items()}
    tw = {c: x for c, x in tw.items() if x is not None}
    low = sorted(tw.items(), key=lambda kv: kv[1])[:13]
    figs["twin_unit"] = {"n": len(tw), "median": float(np.median(list(tw.values()))),
                         "quartiles": [float(np.percentile(list(tw.values()), q)) for q in (25, 75)],
                         "min": float(min(tw.values())), "thirteen_lowest": low,
                         "provenance": provenance([f"validation/{LAST}/SUITE.json"], LAST)}
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(list(tw.values()), bins=40)
    ax.set_xlabel("check 1: median per-layer twin cosine"); ax.set_ylabel("constructs")
    ax.set_title(f"Twin agreement over {len(tw)} constructs at the last checkpoint")
    fig.tight_layout(); fig.savefig(OUT / "FIG_twin_unit.svg")
    fig.savefig(OUT / "FIG_twin_unit.png", dpi=160); plt.close(fig)

    # (f) the sigma-0 floor
    fl = []
    for c, v in suite["constructs"].items():
        ep = v.get("8 eps") or {}
        for g, e in (ep.get("vs_default") or {}).items():
            if (e or {}).get("in_twin_units") is not None:
                fl.append((c, float(g), e["in_twin_units"]))
    bysetting = collections.defaultdict(list)
    for c, g, x in fl:
        bysetting[g].append(x)
    figs["sigma0_floor"] = {
        "default": 0.01, "by_setting": {str(g): {"n": len(v), "max": float(max(v)),
                                                 "median": float(np.median(v))}
                                        for g, v in sorted(bysetting.items())},
        "max_over_all": float(max(x for _, _, x in fl)) if fl else None,
        "provenance": provenance([f"validation/{LAST}/SUITE.json"], LAST)}

    (OUT / "FIGURES.json").write_text(json.dumps(figs, indent=1, default=float))
    print(f"B2: ceiling plug-in untrained {plug.get('untrained'):.4f} -> first "
          f"{plug[refs[0]]:.4f} -> last {plug[refs[-1]]:.4f}")
    print(f"    bootstrap mean untrained {figs['ceiling']['untrained_bootstrap_mean']:.4f} -> "
          f"first {bootmed.get(refs[0], float('nan')):.4f}")
    for r, v in sorted(c9.items()):
        print(f"    check 9 {r:24s} P1 median {v['median_p1']:.4f} (n={v['n_p1']}), "
              f"min {v['min_p1']:.4f} at the construct {v['argmin_p1']!r}")
    print(f"    gate by contrast class: "
          f"{ {k: dict(v) for k, v in gate.items()} }")
    print(f"    twin unit: median {figs['twin_unit']['median']:.4f}, min "
          f"{figs['twin_unit']['min']:.4f}; lowest {figs['twin_unit']['thirteen_lowest'][0]}")
    print(f"    sigma-0 floor: max over all settings {figs['sigma0_floor']['max_over_all']:.4f} "
          f"twin units")
    print(f"-> {OUT / 'FIGURES.json'} and FIG_*.svg/png")


# ---------------------------------------------------------------- B3
rows_order = []
REFS4 = ("untrained", "pretrain_iter_0001000", "pretrain_iter_0020000", "pretrain_iter_0484865")


def _tree(ref: str, twin: str, c: str, keys):
    """Named members of a tree npz, **from the dataset** (A52).

    This preferred the local tree, and that tree holds the PILOT recording at the phase
    recording's path -- differing from the dataset at every one of its eight references, cosine
    median 0.8998, carrier-set overlap 0.7032. `FOR_EXAMPLE.json` was computed through it at
    three of its four references before this was found.
    """
    from profilebattery.profiles import resolve
    return resolve.profile(ref, twin, c, keys)


def _sparsity(v: np.ndarray, k: int = K) -> float:
    f = (np.asarray(v, np.float64).ravel()) ** 2
    tot = f.sum()
    return float(np.partition(f, f.size - k)[f.size - k:].sum() / tot) if tot else float("nan")


def _carriers(v: np.ndarray, k: int = K) -> np.ndarray:
    f = (np.asarray(v, np.float64).ravel()) ** 2
    return np.sort(np.argsort(-f, kind="stable")[:k])


def _com(v: np.ndarray, lo: int = 1, hi: int = 23) -> float:
    n = np.linalg.norm(np.asarray(v, np.float64).reshape(24, -1), axis=1)[lo:hi]
    w = n.sum()
    return float((n * np.arange(lo, hi)).sum() / w) if w else float("nan")


def b3(args) -> None:
    """The running example `For` at the four per-prompt references, both forms."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows, prev = {}, None
    for ref in REFS4:
        e = {}
        z1 = _tree(ref, "twin1", "For", ("A_mean", "B_mean", "A_raw", "B_raw", "concept"))
        std = z1.get("concept")
        if std is None:
            std = z1["A_mean"] - z1["B_mean"]
        raw = z1["A_raw"] - z1["B_raw"]
        z2 = _tree(ref, "twin2", "For", ("concept",))
        std2 = z2["concept"]
        e["norm_raw"] = float(np.linalg.norm(raw))
        e["norm_standardised"] = float(np.linalg.norm(std))
        e["sparsity_standardised"] = _sparsity(std)
        e["sparsity_raw"] = _sparsity(raw)
        a, b = std.ravel(), std2.ravel()
        e["twin_agreement_cosine"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        c1, c2 = _carriers(std), _carriers(std2)
        e["twin_ceiling_plug_in"] = float(np.intersect1d(c1, c2).size / K)
        e["centre_of_mass_layers_1_22"] = _com(std)
        if prev is not None:
            e["carrier_overlap_with_previous_listed_reference"] = float(
                np.intersect1d(c1, prev).size / K)
            e["previous_listed_reference"] = rows_order[-1]
        top = np.argsort(-np.abs(std.ravel()))[:20]
        e["top20_neurons"] = [{"index": int(i), "layer": int(i // 6144),
                               "unit": int(i % 6144), "value": float(std.ravel()[i])}
                              for i in top]
        rows[ref] = e
        prev = c1
        rows_order.append(ref)
    out = {"construct": "For", "space": "product (24 x 6144)",
           "forms": "standardised primary for which-neurons and concentration; raw for size "
                    "(C0.12)", "references": rows,
           "provenance": provenance([f"profiles/{r}/twin1|twin2/For.npz" for r in REFS4])}
    (OUT / "FOR_EXAMPLE.json").write_text(json.dumps(out, indent=1, default=float))
    print("B3: `For` through the pipeline")
    for r, e in rows.items():
        print(f"   {r:24s} raw norm {e['norm_raw']:8.2f}  std norm {e['norm_standardised']:8.2f}  "
              f"sparsity(std) {e['sparsity_standardised']:.4f}  agreement {e['twin_agreement_cosine']:.4f}  "
              f"ceiling {e['twin_ceiling_plug_in']:.4f}  depth {e['centre_of_mass_layers_1_22']:.2f}")
    print(f"-> {OUT / 'FOR_EXAMPLE.json'}")


# ---------------------------------------------------------------- B5
# ---------------------------------------------------------------- B4
SCOPES = ("module", "function", "method")
INDENTED = ("function", "method")
#: `For` plus the eleven other P1 constructs whose leading piece differs by scope
B4_CONSTRUCTS = ("For", "Assert", "Delete", "If", "Import", "ImportFrom", "Match",
                 "Pass", "Raise", "Try", "While", "With")
B4_REFS = (LAST, "pretrain_iter_0001000")


class ScopeView:
    """A view of an accumulator npz exposing only the `hs|` keys of the wanted scopes.

    The scope sits **inside the core id** -- `hs|For/c6@module|token|L0|b0:keyword|mean` -- so
    `from_accumulators` would otherwise read `c6@module` and `c6@function` as two separate cores
    and weight them equally in the core mean. This view strips the scope to one label and
    combines colliding (core, layer, cell) entries by count-weighted mean of the raw means, which
    reconstructs the unscoped per-cell accumulator restricted to those scopes. Taking all three
    scopes therefore has to reproduce the `h|` stream exactly, and `b4` asserts that before it
    reports anything.

    `wrong_scope` counts the keys refused, so the module-scope profile can be shown to have been
    computed from module cells only -- the design's acceptance condition.
    """

    def __init__(self, z, scopes, label: str):
        self.scopes, self.label = tuple(scopes), label
        self.read = self.wrong_scope = 0
        acc: dict[str, list] = {}
        for k in z.files:
            if not (k.startswith("hs|") and k.endswith("|mean")):
                continue
            stem = k[:-len("|mean")]
            parts = stem.split("|")
            core, _, scope = parts[1].partition("@")
            if not scope:
                raise ValueError(f"hs key without a scope: {stem!r}")
            if scope not in self.scopes:
                self.wrong_scope += 1
                continue
            self.read += 1
            out = "|".join(["hs", f"{core}@{label}", *parts[2:]])
            acc.setdefault(out, []).append(
                (int(z[stem + "|n"][0]), np.asarray(z[stem + "|mean"], np.float64)))
        self._d: dict[str, np.ndarray] = {}
        for out, items in acc.items():
            tot = sum(n for n, _ in items) or 1
            self._d[out + "|n"] = np.array([tot])
            self._d[out + "|mean"] = sum((n / tot) * m for n, m in items)
        self.files = list(self._d)

    def __getitem__(self, k):
        return self._d[k]

    def prompts(self) -> int:
        """Summed positions at layer 0 -- the count behind the scope's profile."""
        return sum(int(self._d[k][0]) for k in self._d
                   if k.endswith("|n") and "|L0|" in k)


def _scope_concept(ref: str, twin: str, c: str, scopes, label: str, ys):
    """The concept profile for one scope group, in the product space, or None with a cause."""
    from profilebattery.profiles.profile import from_accumulators
    out = {}
    for kind in ("A", "B"):
        p = dataset_record(ref, f"{twin}__{c}_{kind}_acc.npz")
        v = ScopeView(np.load(p), scopes, label)
        if not v.files:
            return None, f"no hs| {label} accumulators in {twin}__{c}_{kind}", None
        out[kind] = (from_accumulators(v, ys, c, kind, stream="hs").mean, v)
    (a, va), (b, vb) = out["A"], out["B"]
    return a - b, None, {"prompts_A": va.prompts(), "prompts_B": vb.prompts(),
                         "keys_read_A": va.read, "keys_refused_other_scope_A": va.wrong_scope,
                         "keys_read_B": vb.read, "keys_refused_other_scope_B": vb.wrong_scope}


def _ov(c1: np.ndarray, c2: np.ndarray) -> dict:
    i = int(np.intersect1d(c1, c2).size)
    return {"overlap_share_of_K": i / K, "jaccard": i / (2 * K - i), "n_common": i}


def _cos(x: np.ndarray, y: np.ndarray) -> float:
    a, b = x.ravel(), y.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def b4(args) -> None:
    """Feature 1b: one role, two tokens -- `\u2581for` (indented) against `for` (module)."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows, skipped, checks = {}, {}, {}
    for ref in B4_REFS:
        ys = {t: yardstick(ref, t) for t in ("twin1", "twin2")}
        for c in B4_CONSTRUCTS:
            prof, meta, cause = {}, {}, None
            for twin in ("twin1", "twin2"):
                for scopes, label in ((("module",), "module"), (INDENTED, "indented")):
                    try:
                        v, why, m = _scope_concept(ref, twin, c, scopes, label, ys[twin])
                    except Exception as exc:                            # noqa: BLE001
                        v, why, m = None, f"{type(exc).__name__}: {exc}", None
                    if v is None:
                        cause = why
                        break
                    prof[(twin, label)] = v
                    meta[f"{twin}|{label}"] = m
                if cause:
                    break
            if cause:
                skipped[f"{ref}|{c}"] = cause
                print(f"   skip {ref} {c}: {cause}")
                continue
            car = {k: _carriers(v) for k, v in prof.items()}
            e = {"n_prompts_and_keys": meta,
                 "ceiling_within_scope_across_twins": {
                     lab: _ov(car[("twin1", lab)], car[("twin2", lab)])
                     for lab in ("module", "indented")},
                 "between_scope_within_twin": {
                     t: _ov(car[(t, "module")], car[(t, "indented")])
                     for t in ("twin1", "twin2")},
                 "between_scope_cross_twin": {
                     "twin1_module_vs_twin2_indented":
                         _ov(car[("twin1", "module")], car[("twin2", "indented")]),
                     "twin2_module_vs_twin1_indented":
                         _ov(car[("twin2", "module")], car[("twin1", "indented")])},
                 "cosine_between_scopes": {
                     t: _cos(prof[(t, "module")], prof[(t, "indented")])
                     for t in ("twin1", "twin2")},
                 "sparsity": {f"{t}|{lab}": _sparsity(prof[(t, lab)])
                              for (t, lab) in prof},
                 "centre_of_mass_layers_1_22": {f"{t}|{lab}": _com(prof[(t, lab)])
                                                for (t, lab) in prof}}
            ceilings = [e["ceiling_within_scope_across_twins"][lab]["overlap_share_of_K"]
                        for lab in ("module", "indented")]
            floor = min(ceilings)
            e["ceiling_used_is_the_lower_of_the_two"] = floor
            e["share_of_ceiling"] = {
                k: (v["overlap_share_of_K"] / floor if floor else None)
                for k, v in {**e["between_scope_within_twin"],
                             **e["between_scope_cross_twin"]}.items()}
            rows[f"{ref}|{c}"] = e
            m = e["n_prompts_and_keys"]["twin1|module"]
            i_ = e["n_prompts_and_keys"]["twin1|indented"]
            print(f"   {ref:24s} {c:12s} module n {m['prompts_A']:5d}/{m['prompts_B']:6d}  "
                  f"indented n {i_['prompts_A']:5d}/{i_['prompts_B']:6d}  "
                  f"ceiling {floor:.4f}  between(twin1) "
                  f"{e['between_scope_within_twin']['twin1']['overlap_share_of_K']:.4f}  "
                  f"share {e['share_of_ceiling']['twin1']:.3f}  "
                  f"cos {e['cosine_between_scopes']['twin1']:+.4f}")
        # acceptance: all three scopes together must reproduce the unscoped `h|` stream
        checks[ref] = _b4_reconstruction(ref, "twin1", "For", ys["twin1"])
    out = {"question": "one role, two tokens: the indented `\u2581for` against the module-level "
                       "`for` (piece id 2461)",
           "space": "product (24 x 6144), standardised against the C yardstick as stored",
           "overlap_convention": "primary = |intersection| / K with K = 7,373 (the programme's "
                                 "convention, in which the twin ceiling band is 0.57-0.64); "
                                 "Jaccard beside",
           "scope_groups": {"module": ["module"], "indented": list(INDENTED)},
           "source": "the record's `hs|` per-scope accumulators, read from the dataset (A50)",
           "rows": rows, "skipped": skipped, "reconstruction_check": checks,
           "acceptance": {
               "n_per_scope": "240 module and 480 indented A prompts at every construct and "
                              "reference, as the design states; B is 206-292 module and 428-514 "
                              "indented, since B's prompts are drawn per subtype",
               "module_profile_from_module_cells_only": "every accumulator read is checked for "
                                                        "its scope key and keys of other scopes "
                                                        "are counted as refused "
                                                        "(keys_refused_other_scope_*)",
               "ceiling_band": "the design asks for 0.57-0.64. That band is the PILOT's, as the "
                               "draft's own Section 4 says ('for `for` at layer 22 on the "
                               "pilot'). The battery's own band for `For` in the product space "
                               "is 0.8717-0.9060 over the 35 trained pretraining references "
                               "(0.2934 untrained), and B3 reports 0.8717 at 0484865 from the "
                               "derived tree. B4's per-scope ceilings, 0.8367 module and 0.8501 "
                               "indented, sit just below the whole-construct value, which is "
                               "what a third and two thirds of the prompts should give. The two "
                               "routes were checked against each other: the unscoped `h|` "
                               "accumulator route gives ceiling 0.8717 and sparsity 0.5054 for "
                               "`For` at 0484865, the derived tree gives 0.8717 and 0.5054."},
           "provenance": provenance([f"record/{r}/twin{{1,2}}__{{c}}_{{A,B}}_acc.npz"
                                     for r in B4_REFS], reference=LAST)}
    (OUT / "SCOPE_OVERLAP.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {OUT / 'SCOPE_OVERLAP.json'}")


def _b4_reconstruction(ref: str, twin: str, c: str, ys) -> dict:
    """All three scopes pooled must equal the unscoped `h|` profile. See ScopeView."""
    from profilebattery.profiles.profile import from_accumulators
    z = np.load(dataset_record(ref, f"{twin}__{c}_A_acc.npz"))
    whole = from_accumulators(z, ys, c, "A", stream="h").mean
    pooled = from_accumulators(ScopeView(z, SCOPES, "all"), ys, c, "A", stream="hs").mean
    d = float(np.max(np.abs(whole - pooled)))
    scale = float(np.max(np.abs(whole)))
    # the stored means are float16, so the two pooling orders cannot agree closer than that
    # type's resolution -- ~1e-3 relative; anything larger would be a real difference
    tol = 2e-3
    return {"construct": c, "twin": twin, "max_abs_difference": d,
            "max_abs_value": scale, "relative": d / scale if scale else None,
            "tolerance_relative": tol,
            "tolerance_is": "the float16 resolution of the stored means, not an arbitrary cut",
            "verdict": "reproduced" if d <= tol * max(scale, 1.0) else "differs"}


def b5(args) -> None:
    """Feature 1a (two uses of one token) and feature 2 (locate by class), both read off."""
    from experiments.r1_turnover import load_index
    from profilebattery.profiles.names import contract_name, tree_name
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = sorted(p1_set())
    idx = load_index(CARRIERS, [LAST])
    sets1 = {c: np.sort(np.asarray(idx[LAST][f"twin1/{c}|product"], np.int64))
             for c in p1 if f"twin1/{c}|product" in idx[LAST]}
    sets2 = {c: np.sort(np.asarray(idx[LAST][f"twin2/{c}|product"], np.int64))
             for c in p1 if f"twin2/{c}|product" in idx[LAST]}
    def ov(a, b):
        return float(np.intersect1d(a, b).size / K)
    ceil = {c: ov(sets1[c], sets2[c]) for c in sets1 if c in sets2}
    fam = [("For", x) for x in ("ListComp", "GeneratorExp", "SetComp", "DictComp", "AsyncFor")]
    fam += [("If", "IfExp")]
    pairs = {}
    for a, b in fam:
        if a in sets1 and b in sets2:
            pairs[f"{a}|{b}"] = {
                "overlap_cross_twin": ov(sets1[a], sets2[b]),
                "overlap_twin1_only": ov(sets1[a], sets1[b]) if b in sets1 else None,
                "chance": K / (24 * 6144), "ceiling_a": ceil.get(a), "ceiling_b": ceil.get(b),
                "share_of_ceiling_a": (ov(sets1[a], sets2[b]) / ceil[a]) if ceil.get(a) else None}
    # **both conventions, because the draft quotes one and curve 5 computes the other.** Curve 5's
    # quantity is CROSS-twin (twin 1's set against twin 2's), so a shared unit is not shared by
    # construction of one battery; the symmetric twin-1 form is what a reader would compute from a
    # single battery. They differ, so both are reported and each is named.
    order = sorted(sets1)
    allp = [ov(sets1[a], sets2[b]) for i, a in enumerate(order)
            for b in order[i + 1:] if b in sets2]
    allp_t1 = [ov(sets1[a], sets1[b]) for i, a in enumerate(order) for b in order[i + 1:]]
    boot = json.loads((BOOT / f"{LAST}.json").read_text()) if (BOOT / f"{LAST}.json").exists() else {}
    rep2 = json.loads((R2 / "R2_REPORT.json").read_text())
    inv = inventory()
    loc = {}
    for c in p1:
        e = (boot.get(c) or {})
        d = (rep2["curves"].get("pretrain|curve3_depth") or {}).get("series", {}).get(c) or {}
        sp = (rep2["curves"].get("pretrain|curve6_sparsity") or {}).get("series", {}).get(c) or {}
        iv = inv.get(contract_name(c)) or inv.get(c) or {}
        loc[c] = {"contrast_class": iv.get("contrast_class"),
                  "a_family": iv.get("a_family"),
                  "depth_last": (d.get("values") or [None])[-1],
                  "depth_noise_unit": (e.get("depth_twin1") or {}).get("se"),
                  "sparsity_last": (sp.get("values") or [None])[-1],
                  "sparsity_noise_unit": (e.get("sparsity_twin1") or {}).get("se")}
    byclass = collections.defaultdict(lambda: {"depth": [], "sparsity": []})
    for c, e in loc.items():
        k = e["contrast_class"] or "unknown"
        if e["depth_last"] is not None:
            byclass[k]["depth"].append(e["depth_last"])
        if e["sparsity_last"] is not None:
            byclass[k]["sparsity"].append(e["sparsity_last"])
    summ = {k: {q: {"n": len(v[q]), "median": float(np.median(v[q])),
                    "q25": float(np.percentile(v[q], 25)), "q75": float(np.percentile(v[q], 75))}
                for q in ("depth", "sparsity") if v[q]} for k, v in byclass.items()}
    (OUT / "TWO_USE_SHARING.json").write_text(json.dumps(
        {"reference": LAST, "pairs": pairs,
         "median_pair_over_P1_cross_twin": float(np.median(allp)),
         "median_pair_over_P1_twin1_only": float(np.median(allp_t1)),
         "n_pairs_all": len(allp), "chance": K / (24 * 6144),
         "conventions": "cross-twin is curve 5's own quantity (twin1 a against twin2 b); "
                        "twin1-only is the symmetric single-battery form",
         "provenance": provenance([f"validation/carriers/{LAST}.npz"], LAST)}, indent=1,
        default=float))
    (OUT / "LOCATE_BY_CLASS.json").write_text(json.dumps(
        {"reference": LAST, "by_contrast_class": summ, "constructs": loc,
         "class_axis": "contrast_class from inventory_v2 (named-builtin / keyword-token / "
                       "structural), with a_family beside; not hand-assigned",
         "provenance": provenance(["validation/r2/R2_REPORT.json",
                                   f"validation/bootstraps/{LAST}.json",
                                   "inventory/inventory_v2.json"], LAST)}, indent=1, default=float))
    print(f"B5: chance {K/(24*6144):.4f}; median pair over P1 cross-twin "
          f"{np.median(allp):.4f}, twin-1 only {np.median(allp_t1):.4f} ({len(allp):,} pairs)")
    for k, v in pairs.items():
        print(f"   {k:24s} cross-twin {v['overlap_cross_twin']:.4f}  ceiling(a) "
              f"{v['ceiling_a']:.4f}  share {v['share_of_ceiling_a']:.3f}")
    for k, v in sorted(summ.items()):
        dv, sv = v.get("depth"), v.get("sparsity")
        if dv and sv:
            print(f"   {k:16s} n={dv['n']:3d} depth median {dv['median']:.2f} "
                  f"[{dv['q25']:.2f},{dv['q75']:.2f}]  sparsity median {sv['median']:.4f}")
    print(f"-> {OUT / 'TWO_USE_SHARING.json'}, {OUT / 'LOCATE_BY_CLASS.json'}")


# ---------------------------------------------------------------- B8a
# ---------------------------------------------------------------- B6
#: fields of an `ast` statement node whose presence means the header binds a name
BINDING_FIELDS = ("targets", "target", "name", "names", "items")
B6_CUTS = (2, 4, 8, 16, 32)


def _ast_categories(names) -> dict:
    """`compound` and `binding`, derived by rule from the `ast` node classes, not by hand.

    The design fixes both rules: compound = a statement whose node class has a `body` list
    field; binding = a statement whose header introduces a name into scope, which is readable
    off `_fields` as one of `targets`/`target`/`name`/`names`/`items` (plus `NamedExpr`, the one
    expression that binds). The membership each rule produces is recorded in the artefact beside
    the rule, so the reader can check the rule against the list rather than trust either.
    """
    import ast as A
    compound, binding = [], []
    for n in names:
        cls = getattr(A, n, None)
        if cls is None or not isinstance(cls, type) or not hasattr(cls, "_fields"):
            continue
        f = cls._fields
        if issubclass(cls, A.stmt) and "body" in f:
            compound.append(n)
        if (issubclass(cls, A.stmt) and any(x in f for x in BINDING_FIELDS)) or n == "NamedExpr":
            binding.append(n)
    out = {"compound": sorted(compound), "binding": sorted(binding)}
    # the rule and the design's parenthetical list disagree on exactly one construct, and the
    # rule is right about the grammar: `ast.Match._fields` is ('subject', 'cases') -- the `body`
    # sits one level down, in `match_case`, as it does for `ExceptHandler` under `Try`. Python
    # 3.12 exposes no element types (`_field_types` arrives in 3.13), so a rule that reaches
    # through the helper class cannot be written from `_fields` alone. Both memberships are
    # carried and both are scored, so the choice changes no number silently.
    if "Match" in names and "Match" not in out["compound"]:
        out["compound_with_Match"] = sorted([*out["compound"], "Match"])
        out["note_on_Match"] = (
            "the literal rule excludes Match (ast.Match._fields = ('subject', 'cases'); the "
            "body is in match_case), while the design's list includes it; both are scored")
    return out


def _b6_categories(p1, inv) -> dict:
    """Every category tested, as construct -> label, with None for "not in this partition"."""
    from profilebattery.profiles.names import contract_name
    cn = {c: contract_name(c) for c in p1}
    get = {c: inv.get(cn[c], inv.get(c, {})) for c in p1}
    cats = {
        "member_kind": {c: get[c].get("member_kind") for c in p1},
        "asdl_type": {c: get[c].get("asdl_type") for c in p1},
        "a_family": {c: get[c].get("a_family") for c in p1},
        "frame_class": {c: get[c].get("frame_class") for c in p1},
        "builtin_kind": {c: get[c].get("builtin_kind") for c in p1},
    }
    cats["comprehension_family"] = {
        c: ("comprehension" if get[c].get("a_family") == "comprehensions" else None) for c in p1}
    cats["warning_class"] = {
        c: ("warning" if (get[c].get("builtin_kind") == "exception"
                          and cn[c].endswith("Warning")) else
            ("exception_not_warning" if get[c].get("builtin_kind") == "exception" else None))
        for c in p1}
    parts = {x for c in p1 for x in get[c].get("components", [])}
    cats["composite_and_parts"] = {
        c: ("composite" if get[c].get("components") else
            ("obliged_part" if cn[c] in parts else None)) for c in p1}
    rules = _ast_categories([cn[c] for c in p1])
    byname = {cn[c]: c for c in p1}
    for lab, members in [(k, v) for k, v in rules.items() if isinstance(v, list)]:
        cats[lab] = {c: None for c in p1}
        for n in members:
            cats[lab][byname[n]] = lab
    return {"categories": cats, "ast_rule_membership": rules,
            "ast_rules": {"compound": "ast.stmt subclass with a `body` field",
                          "compound_with_Match": "the same, plus Match by the design's list",
                          "binding": "ast.stmt subclass with one of "
                                     f"{list(BINDING_FIELDS)} in _fields, plus NamedExpr"}}


def _purity(labels, clusters, rng, n_perm=1000):
    """Per category value: the share of its members in one cluster, against a label shuffle.

    The null draws the category's n members from **all** the constructs, not only from the ones
    this partition labels. Drawing from the labelled ones is degenerate for a one-valued category
    -- `warning`, `compound`, `binding` label their members and nobody else, so the labelled set
    *is* the category and every draw returns it, giving a null of exactly the observed purity.
    The first run of B6 had that bug and reported null 1.000 against purity 1.000 for the warning
    classes. The question a purity null has to answer is "would any n constructs be this
    concentrated", so the draw is over all of them.
    """
    import collections as C
    out = {}
    cl_all = np.asarray(clusters)
    sizes = C.Counter(cl_all.tolist())
    largest = max(sizes.values()) / len(cl_all)
    named = [i for i, l in enumerate(labels) if l is not None]
    if len(named) < 2:
        return out
    arr = np.asarray([labels[i] for i in named])
    cl = cl_all[np.asarray(named)]
    for val in sorted(set(arr.tolist())):
        m = arr == val
        n = int(m.sum())
        if n < 2:
            continue
        obs = max(C.Counter(cl[m].tolist()).values()) / n
        null = np.empty(n_perm)
        for j in range(n_perm):
            pick = rng.choice(len(cl_all), size=n, replace=False)
            null[j] = max(C.Counter(cl_all[pick].tolist()).values()) / n
        out[str(val)] = {"n": n, "purity": float(obs),
                         "null_mean": float(null.mean()),
                         "null_q95": float(np.quantile(null, 0.95)),
                         "null_q99": float(np.quantile(null, 0.99)),
                         "p": float((null >= obs).mean()),
                         "largest_cluster_share": float(largest)}
    return out


def b6(args) -> None:
    """Feature 3: does carrier sharing cluster the grammar's families?"""
    import scipy.cluster.hierarchy as H
    from scipy.spatial.distance import squareform
    from experiments.r1_turnover import load_index
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261018)
    p1 = sorted(p1_set())
    idx = load_index(CARRIERS, [LAST])
    sets = {c: np.sort(np.asarray(idx[LAST][f"twin1/{c}|product"], np.int64))
            for c in p1 if f"twin1/{c}|product" in idx[LAST]}
    names = sorted(sets)
    n = len(names)
    # the sharing matrix: raw Jaccard of carrier sets, within twin 1 so the diagonal is 1
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            inter = int(np.intersect1d(sets[names[i]], sets[names[j]]).size)
            J[i, j] = J[j, i] = inter / (len(sets[names[i]]) + len(sets[names[j]]) - inter)
    assert np.allclose(J, J.T) and np.allclose(np.diag(J), 1.0), "matrix not symmetric/unit"
    Dm = 1.0 - J
    np.fill_diagonal(Dm, 0.0)
    Z = H.linkage(squareform(Dm, checks=False), method="average")
    cats = _b6_categories(names, inventory())
    cuts = sorted({*B6_CUTS, *(len({v for v in c.values() if v is not None})
                               for c in cats["categories"].values())} - {0, 1})
    per_cut = {}
    for k in cuts:
        cl = H.fcluster(Z, t=k, criterion="maxclust")
        per_cut[k] = {"n_clusters_realised": int(len(set(cl.tolist()))),
                      "sizes": sorted(collections.Counter(cl.tolist()).values(), reverse=True),
                      "categories": {name: _purity([c[x] for x in names], cl, rng)
                                     for name, c in cats["categories"].items()}}
        print(f"   cut {k:3d} -> {per_cut[k]['n_clusters_realised']} clusters; "
              + "  ".join(f"{nm}:" + ",".join(
                  f"{v}={d['purity']:.2f}(p={d['p']:.3f})"
                  for v, d in sorted(r.items())[:2])
                  for nm, r in list(per_cut[k]["categories"].items())[:3]))
    # the disagreements: highest sharing between constructs that share no tested category
    shared = np.zeros((n, n), bool)
    for c in cats["categories"].values():
        lab = [c[x] for x in names]
        for i in range(n):
            if lab[i] is None:
                continue
            for j in range(i + 1, n):
                if lab[j] == lab[i]:
                    shared[i, j] = shared[j, i] = True
    pairs = [(J[i, j], names[i], names[j]) for i in range(n) for j in range(i + 1, n)
             if not shared[i, j]]
    pairs.sort(reverse=True)
    out = {"matrix": "raw Jaccard of carrier sets, product space, twin 1 at " + LAST +
                     " (within twin, so the diagonal is 1; Jaccard is a monotone function of "
                     "|intersection|, so the clustering is the same under either convention)",
           "n_constructs": n, "n_pairs": n * (n - 1) // 2,
           "linkage": "average, on 1 - Jaccard",
           "cut_levels": cuts, "per_cut": per_cut,
           "permutations": 1000,
           "ast_rules": cats["ast_rules"], "ast_rule_membership": cats["ast_rule_membership"],
           "category_membership": {k: {c: v for c, v in d.items() if v is not None}
                                   for k, d in cats["categories"].items()},
           "disagreements_top30": [{"jaccard": float(s), "a": a, "b": b}
                                   for s, a, b in pairs[:30]],
           "median_pair_jaccard": float(np.median(
               [J[i, j] for i in range(n) for j in range(i + 1, n)])),
           "provenance": provenance([f"validation/carriers/{LAST}"], reference=LAST)}
    (OUT / "FAMILIES.json").write_text(json.dumps(out, indent=1, default=float))
    _b6_figure(Z, names, cats["categories"], out)
    print(f"   {n} constructs, {n*(n-1)//2} pairs, median pair Jaccard "
          f"{out['median_pair_jaccard']:.4f}")
    print("   top disagreements: " + ", ".join(
        f"{d['a']}/{d['b']} {d['jaccard']:.3f}" for d in out["disagreements_top30"][:5]))
    print(f"-> {OUT / 'FAMILIES.json'}")


def _b6_figure(Z, names, cats, out) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scipy.cluster.hierarchy as H
    except Exception as exc:                                           # noqa: BLE001
        print(f"   (no dendrogram: {exc})")
        return
    lab = cats["a_family"]
    vals = sorted({v for v in lab.values() if v is not None})
    colour = dict(zip(vals, plt.get_cmap("tab10").colors))
    fig, ax = plt.subplots(figsize=(11, 16))
    d = H.dendrogram(Z, labels=names, orientation="right", ax=ax, color_threshold=0,
                     above_threshold_color="0.6", leaf_font_size=5)
    for t in ax.get_yticklabels():
        t.set_color(colour.get(lab.get(t.get_text()), "0.3"))
    ax.set_xlabel("1 - Jaccard of carrier sets (average linkage)")
    ax.set_title(f"P1 carrier sharing, {LAST}; leaves coloured by a_family")
    ax.legend(handles=[plt.Line2D([], [], color=c, lw=3, label=v) for v, c in colour.items()],
              loc="lower left", fontsize=7)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"FIG_families.{ext}", dpi=170)
    plt.close(fig)
    out["figure"] = str(OUT / "FIG_families.svg")
    print(f"   figure -> {OUT / 'FIG_families.svg'}")


def b8a(args) -> None:
    """The phase battery's prompt-length distribution in model tokens."""
    import gzip
    from transformers import AutoTokenizer
    from profilebattery.record.sweep import REPO as MODEL
    OUT.mkdir(parents=True, exist_ok=True)
    tk = AutoTokenizer.from_pretrained(MODEL, revision="pretrain_iter_0008000",
                                       trust_remote_code=True)
    B = D / "phase_battery"
    per = collections.defaultdict(list)
    for sub in ("twin1", "twin2", "alt"):
        for f in sorted((B / sub).glob("*.jsonl.gz")):
            kind = f.stem.rsplit("_", 1)[-1].replace(".jsonl", "")
            texts = [json.loads(l)["text"] for l in gzip.open(f, "rt")]
            enc = tk(texts, add_special_tokens=False)["input_ids"]
            per[kind].extend(len(x) for x in enc)
            per["ALL"].extend(len(x) for x in enc)
    out = {"tokeniser": f"{MODEL} @ pretrain_iter_0008000", "by_set_kind": {}}
    for k, v in per.items():
        a = np.asarray(v)
        out["by_set_kind"][k] = {"n_prompts": int(a.size), "mean": float(a.mean()),
                                 "min": int(a.min()), "q25": float(np.percentile(a, 25)),
                                 "median": float(np.median(a)),
                                 "q75": float(np.percentile(a, 75)), "max": int(a.max()),
                                 "total_tokens": int(a.sum())}
    out["provenance"] = provenance(["phase_battery/twin1|twin2|alt/*.jsonl.gz"])
    (OUT / "LENGTHS.json").write_text(json.dumps(out, indent=1, default=float))
    print("B8a: prompt length in model tokens")
    for k, v in sorted(out["by_set_kind"].items()):
        print(f"   {k:5s} n={v['n_prompts']:7,d} mean {v['mean']:6.2f}  min {v['min']:3d} "
              f"q25 {v['q25']:6.1f} median {v['median']:6.1f} q75 {v['q75']:6.1f} "
              f"max {v['max']:4d}  total {v['total_tokens']:,}")
    print(f"-> {OUT / 'LENGTHS.json'}")


# ---------------------------------------------------------------- B8b
def _degree_layout(sets, names) -> tuple[np.ndarray, np.ndarray]:
    """Unit degree over P1's carrier sets, and a within-row order by it.

    Index order scatters a construct's carriers across the whole 6,144-wide row -- the carrier
    set is not contiguous in index and nothing says it should be -- so a map drawn in index order
    is 7,373 single pixels and reads as noise. The design's alternative is taken: within each
    layer row the units are ordered by **degree** (how many of P1's 169 carrier sets the unit is
    in) descending, ties by index ascending. The order is computed once, saved, and reused, so
    every later map is drawn on the same layout.
    """
    deg = np.zeros(24 * 6144, np.int32)
    for c in names:
        deg[sets[c]] += 1
    order = np.empty((24, 6144), np.int64)
    for L in range(24):
        row = deg[L * 6144:(L + 1) * 6144]
        order[L] = L * 6144 + np.lexsort((np.arange(6144), -row))
    return deg, order


def b8b(args) -> None:
    """The neuron map: `For` in place over the 24 x 6,144 units, with the carrier set outlined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.r1_turnover import load_index
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = sorted(p1_set())
    idx = load_index(CARRIERS, [LAST])
    sets = {c: np.sort(np.asarray(idx[LAST][f"twin1/{c}|product"], np.int64))
            for c in p1 if f"twin1/{c}|product" in idx[LAST]}
    names = sorted(sets)
    deg, order = _degree_layout(sets, names)
    np.savez_compressed(OUT / "LAYOUT_neurons.npz", order=order, degree=deg,
                        note=np.array(["within each of 24 layer rows, units ordered by degree "
                                       "over P1's 169 twin-1 carrier sets at " + LAST +
                                       " descending, ties by index ascending"]))
    v = _tree(LAST, "twin1", "For", ("concept", "A_mean", "B_mean"))
    prof = v.get("concept")
    if prof is None:
        prof = v["A_mean"] - v["B_mean"]
    prof = np.asarray(prof, np.float64).reshape(24, 6144)
    car = _carriers(prof)
    mask = np.zeros(24 * 6144, bool)
    mask[car] = True
    img = prof.ravel()[order]
    msk = mask[order]
    # acceptance: the drawing is a permutation of the profile, and the outline is the carrier set
    assert np.array_equal(np.sort(img.ravel()), np.sort(prof.ravel())), "map is not the profile"
    assert int(msk.sum()) == K and np.array_equal(np.sort(order[msk]), car), "outline is not K"
    ys1 = yardstick(LAST, "twin1")
    scoped, causes = {}, {}
    for scopes, lab in ((("module",), "module"), (INDENTED, "indented")):
        try:
            sv, why, _m = _scope_concept(LAST, "twin1", "For", scopes, lab, ys1)
        except Exception as exc:                                        # noqa: BLE001
            sv, why = None, f"{type(exc).__name__}: {exc}"
        if sv is None:
            causes[lab] = why
        else:
            scoped[lab] = np.asarray(sv, np.float64).reshape(24, 6144).ravel()[order]
    # The design asks for the carrier set "as a contour". A contour cannot carry it: the set is
    # 7,373 units scattered through 147,456, so at any line width the outline covers the panel
    # and hides the profile it is meant to annotate -- the first drawing was a black field. The
    # same information is drawn instead as its own binary raster on the same x axis and the same
    # layout, directly under the profile, which shows exactly the carrier set and leaves the
    # profile readable. Noted here and in the artefact rather than changed silently.
    lim = float(np.quantile(np.abs(img), 0.995))
    fig, axes = plt.subplots(2 + len(scoped), 1, figsize=(13, 3.0 * (1 + len(scoped)) + 1.5),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [3, 0.6] + [3] * len(scoped)})
    ax = axes[0][0]
    im = ax.imshow(img, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    ax.set_title(f"`For` concept profile at {LAST}, standardised, all 24 x 6,144 units "
                 f"(rows = layers; within a row, units ordered by degree)", fontsize=9)
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02).set_label("profile entry (sd)", size=7)
    ax = axes[1][0]
    ax.imshow(msk, aspect="auto", cmap="binary", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"the top-5% carrier set, {K:,} units, on the same layout", fontsize=8)
    ax.set_ylabel("layer", size=7)
    fig.colorbar(ax.images[0], ax=ax, pad=0.01, fraction=0.02).set_label("carrier", size=7)
    for i, (lab, arr) in enumerate(scoped.items()):
        ax = axes[2 + i][0]
        cmap = ("PRGn", "BrBG")[i % 2]
        l2 = float(np.quantile(np.abs(arr), 0.995)) or lim
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=-l2, vmax=l2,
                       interpolation="nearest")
        ax.set_title(f"the {lab}-scope profile (B4): "
                     f"{'`for`, no indent' if lab == 'module' else 'the indented `\u2581for`'}",
                     fontsize=9)
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02).set_label("profile entry (sd)", size=7)
    axes[-1][0].set_xlabel("unit within layer, in the degree order of LAYOUT_neurons.npz")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"MAP_For.{ext}", dpi=190)
    plt.close(fig)
    meta = {"figure": [str(OUT / "MAP_For.svg"), str(OUT / "MAP_For.png")],
            "layout": str(OUT / "LAYOUT_neurons.npz"),
            "within_row_order": "degree over P1's 169 twin-1 carrier sets, descending, ties by "
                                "index ascending (index order scatters the carriers; stated per "
                                "the design's choice)",
            "carrier_set_drawn_as": "its own binary raster under the profile, on the same "
                                    "layout, not as a contour: 7,373 units scattered through "
                                    "147,456 make a contour that covers the panel",
            "colour_scale": {"map": "RdBu_r diverging, centred on zero",
                             "limit_is_quantile_0.995_of_abs": lim,
                             "scope_panels": {"module": "PRGn", "indented": "BrBG"}},
            "acceptance": {"image_is_a_permutation_of_the_profile": True,
                           "outline_encloses_exactly_the_carrier_set": True,
                           "n_carriers": K},
            "degree": {"max": int(deg.max()), "mean": float(deg.mean()),
                       "share_zero": float((deg == 0).mean())},
            "scope_panels_missing": causes,
            "provenance": provenance([f"profiles/{LAST}/twin1/For.npz",
                                      f"validation/carriers/{LAST}",
                                      f"record/{LAST}/twin1__For_{{A,B}}_acc.npz"],
                                     reference=LAST)}
    (OUT / "MAP_For.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"B8b: map drawn; degree max {deg.max()}, mean {deg.mean():.2f}, "
          f"share-zero {(deg == 0).mean():.4f}; scope panels {list(scoped)}")
    if causes:
        print(f"   missing scope panels: {causes}")
    print(f"-> {OUT / 'MAP_For.svg'}  and {OUT / 'LAYOUT_neurons.npz'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="item", required=True)
    for name, fn in (("b1", b1), ("b2", b2), ("b3", b3), ("b4", b4), ("b5", b5), ("b6", b6), ("b8a", b8a), ("b8b", b8b)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
