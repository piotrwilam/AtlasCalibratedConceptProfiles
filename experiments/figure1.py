r"""Figure 1 — the transformer's neurons in place, with `For`'s concept profile on them.

The paper's first figure: all 24 x 6,144 MLP units drawn as the architecture lays them out, one
row a layer; `For`'s concept profile at the last checkpoint as colour; the carrier set as its own
raster beneath; and the two scopes of `for` -- the indented and the module-level token -- in two
hues below that.

Two things it must satisfy, both asserted in code before the file is written: every drawn cell's
colour traces to a profile entry, and the carrier raster holds exactly the 7,373 units of the
carrier set.

Drawn on the **recorded 20-cell yardstick**, which is paper 1's (Piotr, 21 Oct); the scope panels
come from the record's per-scope accumulators.

    python experiments/figure1.py
"""
from __future__ import annotations

import json

import numpy as np

from experiments.atlas import (INDENTED, K, LAST, OUT, _carriers, _scope_concept, _tree,
                                p1_set, provenance, yardstick)

CAPTION = (
    "Figure 1. The concept profile of the `for` statement on the model's own neurons. All 24 x "
    "6,144 MLP neurons of OpenCoder-1.5B are drawn in place, one row per layer, at the end of "
    "pretraining (2,034 B tokens). **Top:** the carrier set -- the 5% of neurons with the "
    "largest standardised entries -- in the colour of its entry pi(A) - pi(B), in units of the "
    "yardstick's sigma0; every other neuron is grey. Each coloured cell is its own profile "
    "entry, and the grey set is exactly the complement of the carrier set. **Bottom:** one "
    "role, two tokens. The indented `for` and the module-level `for` are two unrelated "
    "vocabulary entries carrying the same grammatical role; each neuron is coloured by which of "
    "their two carrier sets it belongs to. The sets overlap by 0.494, against their own twin "
    "ceilings of 0.837 (module) and 0.850 (indented) -- 0.59 of the ceiling, ten times the "
    "chance level of 0.050, and well short of one representation. Within a layer the neurons "
    "are ordered by degree, how many of P1's 169 carrier sets a neuron belongs to, because "
    "index order scatters a construct's carriers across the whole row; the order is fixed once "
    "and reused so later maps are comparable. Drawn on the recorded 20-cell yardstick.")


def load_map() -> dict:
    """Everything the neuron map is drawn from, loaded once.

    Figure 1 (the small one), Figure C.1 (the full map) and the original two-panel figure all
    read exactly this, so a change to the layout or the carrier set cannot reach one of them
    and miss another.
    """
    from experiments.r1_turnover import load_index
    from experiments.atlas import CARRIERS, _degree_layout
    from experiments.atlas import _carriers as _car
    p1 = sorted(p1_set())
    idx = load_index(CARRIERS, [LAST])
    sets = {c: np.sort(np.asarray(idx[LAST][f"twin1/{c}|product"], np.int64))
            for c in p1 if f"twin1/{c}|product" in idx[LAST]}
    deg, order = _degree_layout(sets, sorted(sets))
    v = _tree(LAST, "twin1", "For", ("concept", "A_mean", "B_mean"))
    prof = v.get("concept")
    if prof is None:
        prof = v["A_mean"] - v["B_mean"]
    prof = np.asarray(prof, np.float64).reshape(24, 6144)
    car = _carriers(prof)
    mask = np.zeros(24 * 6144, bool)
    mask[car] = True
    img, msk = prof.ravel()[order], mask[order]
    assert np.array_equal(np.sort(img.ravel()), np.sort(prof.ravel())), "not the profile"
    assert int(msk.sum()) == K and np.array_equal(np.sort(order[msk]), car), "not the carriers"

    ys1 = yardstick(LAST, "twin1")
    scoped, missing = {}, {}
    for scopes, lab in ((("module",), "module"), (INDENTED, "indented")):
        try:
            sv, why, _m = _scope_concept(LAST, "twin1", "For", scopes, lab, ys1)
        except Exception as exc:                                       # noqa: BLE001
            sv, why = None, f"{type(exc).__name__}: {exc}"
        if sv is None:
            missing[lab] = why
        else:
            scoped[lab] = np.asarray(sv, np.float64).reshape(24, 6144).ravel()[order]

    cats, overlap = None, None
    if len(scoped) == 2:
        cm = np.zeros(24 * 6144, np.int8)
        cmod = set(_car(scoped["module"].reshape(24, 6144)).tolist())
        cind = set(_car(scoped["indented"].reshape(24, 6144)).tolist())
        # scoped[] is already in layout order, so the sets index the layout directly
        for i in cmod:
            cm[i] = 1
        for i in cind:
            cm[i] = 3 if cm[i] == 1 else 2
        cats = cm.reshape(24, 6144)
        overlap = len(cmod & cind) / K
    # the layout reshaped to (24, 6144), so a column is one degree rank across all layers
    return {"deg": deg, "order": order, "img": img.reshape(24, 6144),
            "msk": msk.reshape(24, 6144), "cats": cats, "overlap": overlap,
            "scoped": scoped, "missing": missing}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    M = load_map()
    deg, order, cats, overlap = M["deg"], M["order"], M["cats"], M["overlap"]
    scoped, missing = M["scoped"], M["missing"]
    img, msk = M["img"], M["msk"]

    # two panels of equal width; the colour bar gets its own axes beneath the first, so it
    # cannot steal width from it and the two x-axes stay aligned
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13, 8.6))
    # A density strip above panel (a). The carriers are 307 of 6,144 units a layer, so at page
    # width each is a hairline and the concentration the degree ordering produces -- the whole
    # reason for that ordering -- is invisible in the raster. The strip states it as a number.
    gs = GridSpec(4, 1, height_ratios=[0.55, 3.0, 0.12, 3.0], hspace=0.52, figure=fig,
                  left=0.06, right=0.985, top=0.945, bottom=0.16)
    ad = fig.add_subplot(gs[0])
    a0, acb, a1 = fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3])

    # --- the density strip -----------------------------------------------------------------
    BINS = 96
    per_col = msk.sum(axis=0).astype(float)                     # carriers per column, over 24
    edges = np.linspace(0, msk.shape[1], BINS + 1)
    dens = np.array([per_col[int(edges[i]):int(edges[i + 1])].mean()
                     for i in range(BINS)])
    ad.fill_between(np.arange(BINS) * msk.shape[1] / BINS, 0, dens, step="mid",
                    color="#444444", linewidth=0)
    # the even-spread level: K carriers over 6,144 columns, counted down all 24 layers, is
    # K/6144 = 1.2 a column. The first version wrote `24 * K / msk.size * 24`, which is 24 and
    # put the line off the top of the data.
    even = K / msk.shape[1]
    ad.axhline(even, color="#b35806", lw=0.9, ls="--")
    ad.set_ylim(0, max(dens.max(), even) * 1.35)
    ad.set_ylabel("carriers\nper unit", fontsize=7)
    ad.tick_params(labelsize=7)
    ad.set_xticklabels([])
    ad.margins(x=0)
    ad.set_title("`for` statement, end of pretraining: the carrier set on the model's own "
                 "neurons", fontsize=11)
    ad.text(0.995, 0.92, "dashed: the 5% rate if carriers were spread evenly",
            transform=ad.transAxes, ha="right", va="top", fontsize=7, color="#b35806")

    # --- panel (a): carriers in the profile's colour, every other unit mid-light grey -------
    # Every cell is in place; every **coloured** cell is its own profile entry; the grey set is
    # exactly the complement of the carrier set. Colouring all 147,456 entries made the 5% that
    # matter indistinguishable from the 95% that do not.
    lim = float(np.quantile(np.abs(img[msk]), 0.99))
    shown = np.ma.masked_array(img, mask=~msk)
    a0.set_facecolor("0.72")
    im = a0.imshow(shown, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    a0.set_title("(a)  each carrier in the colour of its entry pi(A) - pi(B); every other "
                 "neuron grey", fontsize=10, loc="left")
    a0.set_ylabel("layer")
    a0.set_yticks([0, 5, 10, 15, 20, 23])          # 24 rows: the last one was unticked
    a0.set_xticklabels([])
    cb = fig.colorbar(im, cax=acb, orientation="horizontal")
    # the label goes ABOVE the bar: below it, it collided with panel (b)'s title
    cb.ax.set_title("panel (a): profile entry, in units of the yardstick's sigma0",
                    fontsize=8, pad=4)
    cb.ax.tick_params(labelsize=7)
    # the bar belongs to panel (a), so it spans only the left half rather than floating the
    # full width between the two panels
    bb = acb.get_position()
    acb.set_position([bb.x0, bb.y0, bb.width * 0.5, bb.height])

    # --- panel (b): which scope's carrier set each unit belongs to --------------------------
    if cats is not None:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        cols = ["0.72", "#1b7837", "#762a83", "#b35806"]
        a1.imshow(cats, aspect="auto", cmap=ListedColormap(cols), vmin=0, vmax=3,
                  interpolation="nearest")
        a1.set_title("(b)  one role, two tokens: which scope's carrier set each neuron "
                     f"belongs to \u2014 they share {overlap:.3f}, which is "
                     f"{overlap / 0.8435:.2f} of the twin ceiling and {overlap / 0.05:.0f}x "
                     "chance", fontsize=10, loc="left")
        a1.set_ylabel("layer")
        a1.set_yticks([0, 5, 10, 15, 20, 23])
        a1.legend(handles=[Patch(color=cols[1], label="module scope only"),
                           Patch(color=cols[2], label="indented scope only"),
                           Patch(color=cols[3], label="both"),
                           Patch(color=cols[0], label="neither")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4, fontsize=8,
                  frameon=False)
    a1.set_xlabel("neuron within layer, ordered by degree over P1's 169 carrier sets "
                  "(high degree at the left)")
    a0.set_xlim(a1.get_xlim())
    ad.set_xlim(a1.get_xlim())
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"FIG1_neuron_map.{ext}", dpi=300)
    plt.close(fig)
    meta = {"figure": "Figure 1", "caption": CAPTION,
            "files": [str(OUT / f"FIG1_neuron_map.{e}") for e in ("svg", "png")],
            "reference": LAST, "reference_described_as": "end of pretraining (2,034 B tokens)",
            "construct": "For", "twin": "twin1",
            "yardstick": "the recorded 20-cell split (paper 1's, Piotr 21 Oct)",
            "layout": str(OUT / "LAYOUT_neurons.npz"),
            "acceptance": {"every_coloured_cell_traces_to_a_profile_entry": True,
                           "grey_set_is_exactly_the_complement_of_the_carrier_set": True,
                           "raster_holds_exactly_the_carrier_set": True, "n_carriers": K,
                           "scope_overlap_share_of_K": (overlap if cats is not None else None)},
            "degree": {"max": int(deg.max()), "mean": float(deg.mean()),
                       "share_zero": float((deg == 0).mean())},
            "scope_panels": list(scoped), "scope_panels_missing": missing,
            "provenance": provenance([f"profiles/{LAST}/twin1/For.npz",
                                      f"validation/carriers/{LAST}",
                                      f"record/{LAST}/twin1__For_{{A,B}}_acc.npz"],
                                     reference=LAST)}
    (OUT / "FIG1_neuron_map.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"Figure 1: two panels, scopes {list(scoped)}; degree max {deg.max()}, "
          f"mean {deg.mean():.2f}, share-zero {(deg == 0).mean():.4f}")
    if missing:
        print("  missing scope panels:", missing)
    print(f"-> {OUT / 'FIG1_neuron_map.svg'}")


if __name__ == "__main__":
    main()
