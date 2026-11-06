r"""Figure 1, single column: where the carriers are, by degree and by layer.

The full neuron map moved to Appendix C.5 as Figure C.1 in the length pass of 1 Nov (Piotr,
option b). What the body keeps is the part of it a reader can act on at a quarter of the size:
the carrier set **rises with degree throughout** and is **present in every layer**, and both
statements are read against the same 5% line -- the share that would hold everywhere if the
7,373 carriers were spread evenly over the 147,456 neurons.

Both panels are therefore drawn in the SAME unit, a share of the neurons present, so that the
one dashed line at 5% means the same thing in each. Panel (a)'s quantity is the density strip
of the old figure restated that way: the strip counted carriers per column down all 24 layers,
which is the same number divided by 24.

Drawn from `figure1.load_map()`, which is what Figure C.1 reads too.

    python experiments/figure1_small.py
"""
from __future__ import annotations

import json

import numpy as np

from experiments.figure1 import load_map
from experiments.atlas import K, LAST, OUT, provenance

BINS = 96
CAPTION = (
    "Figure 1. Where the `for` statement's carriers sit among the model's neurons, at the end "
    "of pretraining (2,034 B tokens). The carrier set is the 5% of the 24 x 6,144 MLP neurons "
    "with the largest standardised entries, 7,373 of them; the dashed line is the 5% share "
    "that would hold everywhere if they were spread evenly. **(a)** By degree rank: neurons "
    "are ordered within each layer by degree, how many of P1's 169 carrier sets a neuron "
    "belongs to, and the curve is the share of neurons at that rank, over all 24 layers, that "
    "carry `for`. The share rises with degree throughout -- from none at all among the 15,763 "
    "neurons that belong to no other carrier set, to 18% among those belonging to 50 or more. "
    "**(b)** By layer: the share of each layer's 6,144 neurons that carry `for`. Every layer "
    "has some. The full map, with each carrier in the colour of its entry and the two scopes "
    "of `for` side by side, is Figure C.1.")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    M = load_map()
    msk = M["msk"]                                   # (24, 6144), in layout order
    n_layers, width = msk.shape
    even = K / msk.size                              # the 5% line, as a share

    # (a) share of the neurons at each degree rank, over all 24 layers
    edges = np.linspace(0, width, BINS + 1).astype(int)
    dens = np.array([msk[:, edges[i]:edges[i + 1]].mean() for i in range(BINS)])
    xs = (edges[:-1] + edges[1:]) / 2
    # (b) share of each layer's neurons
    per_layer = msk.mean(axis=1)

    assert abs(msk.mean() - even) < 1e-12, "the two panels and the 5% line disagree"

    # Two thirds of the former 2.7in height. The margins are fractions of the figure, so
    # they are restated here: the axis labels take the same absolute space as before, which
    # is a larger share of a shorter figure, and keeping bottom at 0.23 clipped the xlabel.
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 1.8),
                               gridspec_kw={"wspace": 0.30, "left": 0.095, "right": 0.985,
                                            "top": 0.99, "bottom": 0.30})
    a.fill_between(xs, 0, dens * 100, step="mid", color="#444444", linewidth=0)
    a.axhline(even * 100, color="#b35806", lw=0.9, ls="--")
    a.set_ylim(0, max(dens.max(), even) * 100 * 1.25)
    a.set_xlim(0, width)
    a.set_xlabel("neuron within a layer, by degree (most shared at the left)", fontsize=7.4)
    a.set_ylabel("carriers\n(% of neurons)", fontsize=7.8)
    a.tick_params(labelsize=6.6)
    a.text(0.985, 0.90, "(a)", transform=a.transAxes, ha="right", va="top",
           fontsize=9, fontweight="bold")
    a.text(0.985, 0.62, "even spread", transform=a.transAxes, ha="right", va="top",
           fontsize=6.5, color="#b35806")

    b.bar(np.arange(n_layers), per_layer * 100, color="#1b4f72", width=0.82, linewidth=0)
    b.axhline(even * 100, color="#b35806", lw=0.9, ls="--")
    b.set_xlim(-0.7, n_layers - 0.3)
    b.set_ylim(0, max(per_layer.max(), even) * 100 * 1.25)
    b.set_xticks([0, 5, 10, 15, 20, 23])
    b.set_xlabel("layer", fontsize=7.8)
    b.set_ylabel("carriers\n(% of neurons)", fontsize=7.8)
    b.tick_params(labelsize=6.6)
    b.text(0.985, 0.90, "(b)", transform=b.transAxes, ha="right", va="top",
           fontsize=9, fontweight="bold")

    # `bbox_inches="tight"` crops to the drawn content, so the white band above the axes
    # and below the x-label goes rather than being guessed at through figure-fraction
    # margins -- which have to be re-tuned every time the figure's height changes.
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"FIG1_carriers_in_place.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    meta = {
        "figure": "Figure 1", "caption": CAPTION,
        "files": [str(OUT / f"FIG1_carriers_in_place.{e}") for e in ("png", "svg")],
        "reference": LAST, "construct": "For", "twin": "twin1",
        "single_column": False, "panels": "side by side", "no_title": True,
        "figsize_inches": [7.2, 1.8],
        "cropped": "bbox_inches=tight: no whitespace above or below the axes",
        "unit": "share of the neurons present, per cent, in both panels, so that the one "
                "dashed 5% line means the same in each",
        "even_spread_share": even, "n_carriers": int(msk.sum()),
        "by_degree": {"n_bins": BINS, "max_share": float(dens.max()),
                      "min_share": float(dens.min()),
                      "share_in_the_top_bin": float(dens[0]),
                      "ratio_top_bin_to_even": float(dens[0] / even)},
        "by_degree_value": None,     # filled below
        "by_layer": {"n_layers": int(n_layers),
                     "min_share": float(per_layer.min()), "max_share": float(per_layer.max()),
                     "argmin_layer": int(per_layer.argmin()),
                     "argmax_layer": int(per_layer.argmax()),
                     "n_layers_with_none": int((per_layer == 0).sum()),
                     "per_layer_counts": msk.sum(axis=1).astype(int).tolist()},
        "provenance": provenance([f"profiles/{LAST}/twin1/For.npz",
                                  f"validation/carriers/{LAST}"], reference=LAST)}
    # the monotone claim is about degree VALUE, not rank, so it is stored that way: a rank
    # bin mixes degrees, and the draft's "present at every degree" turned out to be false
    # precisely at degree 0, which no rank bin shows on its own.
    dv, mv = M["deg"][M["order"]].ravel(), msk.ravel()
    bands, prev = [], -1.0
    for lo, hi, lab in ((0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"),
                        (10, 24, "10-24"), (25, 49, "25-49"), (50, 169, "50+")):
        sel = (dv >= lo) & (dv <= hi)
        if not sel.any():
            continue
        share = float(mv[sel].mean())
        bands.append({"degree": lab, "n_neurons": int(sel.sum()),
                      "n_carriers": int(mv[sel].sum()), "share": share})
        prev = share
    meta["by_degree_value"] = {
        "bands": bands,
        "monotone_increasing": all(bands[i]["share"] <= bands[i + 1]["share"]
                                   for i in range(len(bands) - 1)),
        "share_at_degree_zero": bands[0]["share"],
        "n_degree_ranks_with_no_carrier": int((msk.sum(axis=0) == 0).sum()),
        "note": "the draft's C.5 says the carriers are 'present at every degree'. They are "
                "not: no neuron of degree 0 carries `for`, and 2,327 of the 6,144 degree "
                "ranks hold none. Present in every LAYER is true."}
    (OUT / "FIG1_carriers_in_place.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"Figure 1 (single column): even spread {even:.3%}; "
          f"by degree, the top bin is {dens[0]:.1%} "
          f"({dens[0] / even:.1f}x even) and the lowest {dens.min():.1%}; "
          f"by layer, {per_layer.min():.1%} (layer {per_layer.argmin()}) to "
          f"{per_layer.max():.1%} (layer {per_layer.argmax()}), "
          f"{int((per_layer == 0).sum())} layers with none")
    print(f"-> {OUT / 'FIG1_carriers_in_place.png'}")


if __name__ == "__main__":
    main()
