r"""Part 3 of the companion brief: re-weight B to A's companion shares, in twin units.

`check6_companions.py` finds the 0.15 cap breached for every one of the eight constructs
Table B.3 reports, by 3 to 14 companion pairs each. This asks what that costs the profile, by
the route check 4 uses for the generator's mixture: build a second B profile under different
weights, take `relative_distance` between the two concept profiles at the middle layer, and
divide by the construct's own twin distance.

**Four of the eight can be attempted at all.** Re-weighting is a per-prompt operation, so it
needs per-prompt activations, and those exist for the 31-construct subset only: of the eight,
`For`, `If`, `While` and `With` have them; `ClassDef`, `FunctionDef`, `Match` and `Try` have
accumulators alone, which are cell means, and companion presence is not a cell axis, so there
is nothing in them to re-weight. Those four are reported as not computable, not as zero.

**Two of the four are impossible, and it is proved rather than inferred.** The re-weighting
asks for weights `w >= 0`, `sum w = 1`, with `X'w = t` -- X the B prompts' companion
indicators, t A's shares. A phase-1 LP minimising the total moment violation over the whole
simplex settles it: a minimum of zero is feasibility, and anything above zero is a proof of
infeasibility. `For` and `With` come out at zero. `If` cannot reach `UnaryOp` (A 0.231, present
in 15 of 1,440 B prompts) and falls 0.091 short; `While` cannot reach `Compare` at all, because
every one of its 1,440 B prompts contains one and A's share is 0.692 -- a share of 1.0 cannot
be weighted downwards. No amount of recording fixes either; they are properties of the drawn
sets.

**Where it is feasible, the weights are the maximum-entropy ones.** `w_i` proportional to
`exp(-lambda . x_i)` with lambda solved to hit the moments exactly. Among all weightings that
satisfy the constraints this is the one with the largest ESS, so the ESS reported here is an
upper bound on what any exact re-weighting achieves, and the shift is the best-case one.

**The weighted builder is checked against the unweighted one.** With uniform weights it must
reproduce `profile.build`'s `mean` to floating point; the run refuses if it does not.

    .venv/bin/python experiments/check6_reweight.py
"""
from __future__ import annotations

import gzip
import json
import time

import numpy as np
from scipy.optimize import linprog, minimize

from experiments.atlas import D, provenance
from profilebattery.generate.audit import COMPONENTS, present
from profilebattery.profiles.profile import build, cell_of, recorder_order
from profilebattery.profiles.yardstick import Yardstick
from profilebattery.validation.metrics import in_twin_units, relative_distance
from profilebattery.paths import dataset_repo

BAT = D / "phase_battery"
OUT = D / "validation/paper1"
REF = "pretrain_iter_0484865"
CAP = 0.15
EIGHT = ("ClassDef", "For", "FunctionDef", "If", "Match", "Try", "While", "With")
#: the eight minus those without per-prompt activations
PER_PROMPT = ("For", "If", "While", "With")
TWINS = ("twin1", "twin2")


def shard(name: str, ref: str = REF):
    from huggingface_hub import hf_hub_download
    return np.load(hf_hub_download(dataset_repo(), f"record/{ref}/{name}",
                                   repo_type="dataset"))["primary"]


def yardstick(ref: str = REF):
    from huggingface_hub import hf_hub_download
    return Yardstick.from_npz(hf_hub_download(
        dataset_repo(), f"record/{ref}/twin1__C_cells.npz", repo_type="dataset"))


def targets(c: str, rs: dict, inv: dict,
            names: list[str] | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The over-cap non-exempt companions, A's shares, and B's indicator matrix.

    `names` fixes the companion list instead of selecting it from these prompts. Pass it
    whenever the same intervention has to be applied to more than one set of prompts -- the
    two twins, say: selecting per twin makes the lists differ, and then a twin-against-twin
    comparison is measuring two different re-weightings as well as the noise.
    """
    seen = {k: [present(r["text"], c, 1) for r in rs[k]] for k in ("A", "B")}
    comps = sorted({x for k in seen for st in seen[k] for x in st} | set(names or ()))
    sh = {k: {x: float(np.mean([x in st for st in seen[k]])) for x in comps}
          for k in ("A", "B")}
    ex = set(COMPONENTS.get(c, ())) | set(inv.get(c, {}).get("components") or ())
    names = list(names) if names is not None else \
        [x for x in comps if x not in ex and abs(sh["A"][x] - sh["B"][x]) > CAP]
    X = np.array([[x in st for st in seen["B"]] for x in names], float).T
    return names, np.array([sh["A"][x] for x in names]), X


def usable(rows, ys, construct, n_prompts=None):
    """build()'s own filter -- the prompts whose cell the yardstick supports.

    Needs the rows and the yardstick only, so which prompts survive, and therefore whether
    the re-weighting is feasible, is settled before an activation shard is fetched.
    """
    layers = ys.layers
    keep = [(i, cell_of(r))
            for i, r in enumerate(rows[:n_prompts] if n_prompts else rows)]
    keep = [(i, c) for i, c in keep if all(ys.has(l, c) for l in layers)]
    if not keep:
        raise SystemExit(f"REFUSED: {construct}: no prompt has a supported cell")
    return keep


def standardise(primary, keep, ys):
    """The z-scores of the usable prompts, as build() computes them.

    The per-cell mu and sigma stacks are cached across prompts -- there are 20 cells and up
    to 1,440 prompts -- which changes the arithmetic not at all and the time by a lot.
    """
    n_layers = primary.shape[1]
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, c in keep:
        if c not in cache:
            cache[c] = (np.stack([ys.mu(l, c) for l in range(n_layers)]),
                        np.stack([ys.sigma(l, c) for l in range(n_layers)]))
    return np.stack([(primary[i].astype(np.float64) - cache[c][0]) / cache[c][1]
                     for i, c in keep])


def weighted_mean(z, w):
    return np.tensordot(w / w.sum(), z, axes=(0, 0))


def solve(X, t):
    """Maximum-entropy weights hitting the moments, and the LP proof of feasibility."""
    n, m = X.shape
    cost = np.r_[np.zeros(n), np.ones(2 * m)]
    Aeq = np.zeros((m + 1, n + 2 * m))
    Aeq[:m, :n], Aeq[:m, n:n + m], Aeq[:m, n + m:] = X.T, -np.eye(m), np.eye(m)
    Aeq[m, :n] = 1.0
    lp = linprog(cost, A_eq=Aeq, b_eq=np.r_[t, 1.0],
                 bounds=[(0, None)] * (n + 2 * m), method="highs")
    if not lp.success:
        return None, {"lp": lp.message}
    if lp.fun > 1e-7:
        w = lp.x[:n]
        u, v = lp.x[n:n + m], lp.x[n + m:]
        return None, {"infeasible": True, "total_moment_violation": float(lp.fun),
                      "unreachable": [{"j": int(j), "target": float(t[j]),
                                       "b_share": float(X[:, j].mean()),
                                       "best_attainable": float(X[:, j] @ w),
                                       "shortfall": float(u[j] + v[j])}
                                      for j in np.argsort(-(u + v)) if u[j] + v[j] > 1e-7]}

    def obj(lam):
        q = -X @ lam
        mx = q.max()
        e = np.exp(q - mx)
        s = e.sum()
        return np.log(s) + mx + lam @ t, t - (X.T @ (e / s))

    r = minimize(obj, np.zeros(m), jac=True, method="L-BFGS-B",
                 options={"maxiter": 8000, "ftol": 1e-16, "gtol": 1e-14})
    q = -X @ r.x
    w = np.exp(q - q.max())
    w /= w.sum()
    err = float(np.abs(X.T @ w - t).max())
    if err > 1e-4:
        return None, {"infeasible": False, "solver_residual": err,
                      "why": "the LP says the moments are attainable but the entropy "
                             "solution did not converge to them"}
    return w, {"infeasible": False, "max_moment_error": err,
               "ess": float(1.0 / (w ** 2).sum()), "n": int(len(w)),
               "ess_fraction": float(1.0 / (w ** 2).sum() / len(w)),
               "heaviest_prompt_weight": float(w.max()),
               "heaviest_10_together": float(np.sort(w)[-10:].sum())}


def main() -> None:
    t0 = time.time()
    inv = json.loads((D / "inventory/inventory_v2.json").read_text())["constructs"]
    lic = json.loads((OUT / "LICENCE_TABLE.json").read_text())["constructs"]
    lic = lic if isinstance(lic, dict) else {r["construct"]: r for r in lic}
    ys = yardstick(REF)

    doc = {"what": "re-weighting B to A's companion shares, and the profile's shift in twin "
                   "units, for the eight constructs Table B.3 reports",
           "definition": {
               "targets": f"the non-exempt companions whose |share(A) - share(B)| exceeds "
                          f"{CAP}; exemptions are the UNION of audit.COMPONENTS and the "
                          f"inventory's `components`, the more generous of the two lists",
               "weights": "maximum-entropy weights on B's prompts hitting A's shares exactly; "
                          "the largest-ESS weighting that satisfies the moments, so the shift "
                          "is a best case",
               "feasibility": "settled by a phase-1 LP over the whole simplex, not by an "
                              "optimiser's residual",
               "shift": "relative_distance between the base concept profile (A - B) and the "
                        "re-weighted one (A - B_rw) at the middle layer, divided by the "
                        "construct's own twin distance -- check 4's path",
               "reference": REF,
               "counts": "both twins pooled, as the battery files are read"},
           "fixed_before_computing": True, "per_construct": {}}

    for c in EIGHT:
        if c not in PER_PROMPT:
            doc["per_construct"][c] = {
                "outcome": "NOT COMPUTABLE",
                "why": "no per-prompt activations: this construct is outside the "
                       "31-construct subset, and its accumulators are cell means, in which "
                       "companion presence is not an axis"}
            continue
        rows = {k: [r for t in TWINS for r in recorder_order(
                    [json.loads(l) for l in
                     gzip.open(BAT / t / f"{c}_{k}.jsonl.gz", "rt")])]
                for k in ("A", "B")}
        keep = {k: usable(rows[k], ys, c) for k in ("A", "B")}
        rs = {k: [rows[k][i] for i, _ in keep[k]] for k in ("A", "B")}

        seen = {k: [present(r["text"], c, 1) for r in rs[k]] for k in ("A", "B")}
        comps = sorted({x for k in seen for st in seen[k] for x in st})
        sh = {k: {x: float(np.mean([x in st for st in seen[k]])) for x in comps}
              for k in ("A", "B")}
        ex = set(COMPONENTS.get(c, ())) | set(inv.get(c, {}).get("components") or ())
        names = [x for x in comps if x not in ex and abs(sh["A"][x] - sh["B"][x]) > CAP]
        X = np.array([[x in st for st in seen["B"]] for x in names], float).T
        t = np.array([sh["A"][x] for x in names])

        w, info = solve(X, t)
        entry = {"n_a_usable": len(rs["A"]), "n_b_usable": len(rs["B"]),
                 "n_targets": len(names),
                 "targets": [{"companion": x, "a": sh["A"][x], "b": sh["B"][x],
                              "diff": sh["A"][x] - sh["B"][x]} for x in names],
                 **{k: v for k, v in info.items() if k != "unreachable"}}
        if "unreachable" in info:
            entry["unreachable"] = [{**u, "companion": names[u["j"]]}
                                    for u in info["unreachable"]]
        if w is None:
            entry["outcome"] = "IMPOSSIBLE"
            entry["why"] = ("no weighting of B's prompts can reach A's companion shares; the "
                            "LP's minimum total moment violation is "
                            f"{info.get('total_moment_violation', float('nan')):.4f}")
            doc["per_construct"][c] = entry
            print(f"  {c:<12s} IMPOSSIBLE -- " + ", ".join(
                f"{u['companion']} wants {u['target']:.3f}, B has {u['b_share']:.3f}, "
                f"best {u['best_attainable']:.3f}" for u in entry.get("unreachable", [])[:2]))
            continue

        # feasible: only now is it worth moving the activations
        zs = {}
        for k in ("A", "B"):
            prim = np.concatenate([shard(f"{t}__{c}_{k}.npz") for t in TWINS])
            zs[k] = standardise(prim, keep[k], ys)
            ref = build(prim, rows[k], ys, c, k)
            got = weighted_mean(zs[k], np.ones(len(zs[k])))
            if not np.allclose(got, ref.mean, rtol=0, atol=1e-9):
                raise SystemExit(
                    f"REFUSED: {c}/{k}: the weighted builder does not reproduce "
                    f"profile.build at uniform weights (max |diff| "
                    f"{np.abs(got - ref.mean).max():.3e})")
        n_layers = zs["A"].shape[1]
        mid = n_layers // 2
        a_mean = weighted_mean(zs["A"], np.ones(len(zs["A"])))
        b_mean = weighted_mean(zs["B"], np.ones(len(zs["B"])))
        b_rw = weighted_mean(zs["B"], w)
        base, rw = a_mean[mid] - b_mean[mid], a_mean[mid] - b_rw[mid]
        dist = relative_distance(base, rw)
        twin = (lic.get(c, {}).get("check1_twin_unit") or {}).get("median_distance")
        cos = float(base @ rw / (np.linalg.norm(base) * np.linalg.norm(rw)))
        entry.update({
            "outcome": "run", "layer": mid, "layers": n_layers,
            "relative_distance": dist,
            "twin_distance": twin,
            "shift_in_twin_units": in_twin_units(dist, twin) if twin else None,
            "cosine_base_vs_reweighted": cos,
            "norm_base": float(np.linalg.norm(base)),
            "norm_reweighted": float(np.linalg.norm(rw)),
            "b_share_shift_max": float(np.abs(X.T @ w - X.mean(0)).max())})
        doc["per_construct"][c] = entry
        print(f"  {c:<12s} shift {entry['shift_in_twin_units']:5.2f} twin units  "
              f"(distance {dist:.4f}, twin {twin:.4f})  cosine {cos:.4f}  "
              f"ESS {info['ess']:.0f}/{info['n']} = {info['ess_fraction']:.1%}")

    run = {k: v for k, v in doc["per_construct"].items() if v.get("outcome") == "run"}
    doc["summary"] = {
        "n_reported": len(run),
        "n_impossible": sum(1 for v in doc["per_construct"].values()
                            if v.get("outcome") == "IMPOSSIBLE"),
        "n_not_computable": sum(1 for v in doc["per_construct"].values()
                                if v.get("outcome") == "NOT COMPUTABLE"),
        "shift_in_twin_units": {k: v["shift_in_twin_units"] for k, v in run.items()},
        "median_shift_in_twin_units": (float(np.median(
            [v["shift_in_twin_units"] for v in run.values()])) if run else None),
        "reading": "the shift is what correcting the companion asymmetry would do to the "
                   "concept profile, in units of what redrawing the battery does. Above 1 "
                   "means the asymmetry moves the profile by more than the instrument's own "
                   "noise."}
    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance(
        [f"record/{REF}/twin{{1,2}}__{{For,With,If,While}}_{{A,B}}.npz",
         f"record/{REF}/twin1__C_cells.npz",
         "phase_battery/twin{1,2}/*_{A,B}.jsonl.gz",
         "validation/paper1/LICENCE_TABLE.json",
         "inventory/inventory_v2.json"], reference=REF)
    (OUT / "CHECK6_REWEIGHT.json").write_text(json.dumps(doc, indent=1, default=float))
    s = doc["summary"]
    print(f"\n  {s['n_reported']} reported, {s['n_impossible']} impossible, "
          f"{s['n_not_computable']} not computable")
    if s["median_shift_in_twin_units"] is not None:
        print(f"  median shift {s['median_shift_in_twin_units']:.2f} twin units")
    print(f"  {doc['minutes']:.1f} min")
    print(f"-> {OUT / 'CHECK6_REWEIGHT.json'}")


if __name__ == "__main__":
    main()
