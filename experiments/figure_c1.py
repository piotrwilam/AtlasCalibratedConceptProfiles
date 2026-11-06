r"""Figure C.1 (Appendix C.5) -- the full neuron map, full page landscape, with insets.

This is the figure that was Figure 1 until the length pass of 1 Nov moved it to the appendix.
Two changes from that version: the page is landscape, so the 6,144 columns get roughly twice
the width they had, and **each panel carries an inset enlarging the 300 highest-degree
neurons** -- the part of the map where the carriers are densest and where, at page width, a
single entry is a hairline.

The density strip that sat above panel (a) is gone: it is now Figure 1(a) in the body, drawn
at a share rather than a count.

The caption lives in the draft's Appendix C.5 and is not drawn here; the panels carry only
their letters, which that caption refers to.

Reads `figure1.load_map()`, the same loader Figure 1 uses, so the two cannot disagree about
the layout or the carrier set.

    python experiments/figure_c1.py
"""
from __future__ import annotations

import json

import numpy as np

from experiments.figure1 import load_map
from experiments.atlas import K, LAST, OUT, provenance

#: the inset enlarges this many of the highest-degree columns
INSET_COLS = 300
#: panel (b)'s four categories, in the order the category array codes them
CAT_COLOURS = ["0.72", "#1b7837", "#762a83", "#b35806"]
CAT_LABELS = ["neither", "module scope only", "indented scope only", "both"]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    OUT.mkdir(parents=True, exist_ok=True)

    M = load_map()
    img, msk, cats, overlap = M["img"], M["msk"], M["cats"], M["overlap"]
    if cats is None:
        raise SystemExit(
            f"REFUSED: panel (b) needs both scope panels and only {list(M['scoped'])} loaded "
            f"({M['missing']}). A one-panel Figure C.1 is not the figure C.5 describes.")
    n_layers, width = msk.shape

    lim = float(np.quantile(np.abs(img[msk]), 0.99))
    shown = np.ma.masked_array(img, mask=~msk)
    cmap_b = ListedColormap(CAT_COLOURS)

    # A4 landscape. Four rows, not two: the inset gets its OWN axes beneath each map rather
    # than floating on top of it. Placed inside the axes it covered the columns from about
    # 3,700 rightward -- a third of the map the figure exists to show -- and its connectors
    # crossed the whole panel. Beneath, nothing is hidden and no connector is needed; the
    # marked rectangle at the left edge says where the enlargement comes from.
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(11.2, 7.6))
    gs = GridSpec(4, 1, height_ratios=[3.0, 1.15, 3.0, 1.15], hspace=0.40, figure=fig,
                  left=0.055, right=0.90, top=0.965, bottom=0.095)
    a0, i0, a1, i1 = (fig.add_subplot(gs[r]) for r in range(4))

    def draw_a(ax):
        ax.set_facecolor(CAT_COLOURS[0])
        return ax.imshow(shown, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                         interpolation="nearest")

    def draw_b(ax):
        ax.set_facecolor(CAT_COLOURS[0])
        return ax.imshow(cats, aspect="auto", cmap=cmap_b, vmin=0, vmax=3,
                         interpolation="nearest")

    def mark(ax):
        """Outline the columns the inset enlarges, at the left edge of the map."""
        ax.add_patch(Rectangle((-0.5, -0.5), INSET_COLS, n_layers,
                               fill=False, edgecolor="#111111", linewidth=1.3, zorder=5))

    def style_map(ax, letter):
        ax.set_ylabel("layer", fontsize=9)
        ax.set_yticks([0, 5, 10, 15, 20, 23])
        ax.tick_params(labelsize=8)
        ax.text(-0.048, 1.03, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=12, fontweight="bold")
        mark(ax)

    def style_inset(ax):
        ax.set_xlim(-0.5, INSET_COLS - 0.5)
        ax.set_ylim(n_layers - 0.5, -0.5)
        ax.set_xticks([0, 100, 200, 300])
        ax.set_yticks([0, 11, 23])
        ax.tick_params(labelsize=7, length=2)
        ax.set_ylabel("layer", fontsize=7.5)
        for sp in ax.spines.values():
            sp.set_linewidth(1.3)
            sp.set_edgecolor("#111111")
        ax.set_title(f"the {INSET_COLS} highest-degree neurons (the outlined strip above), "
                     f"enlarged", fontsize=8, pad=3)

    im = draw_a(a0)
    style_map(a0, "(a)")
    draw_a(i0)
    style_inset(i0)
    draw_b(a1)
    style_map(a1, "(b)")
    draw_b(i1)
    style_inset(i1)

    for ax in (a0, i0, a1):
        ax.set_xticklabels([])
    a0.set_xlim(a1.get_xlim())
    i1.set_xlabel("neuron within layer, ordered by degree (the 300 most shared)", fontsize=8)
    a1.set_xlabel("neuron within layer, ordered by degree over P1's 169 carrier sets "
                  "(high degree at the left)", fontsize=9)

    # the colour bar gets its own axes at the right of panel (a), so its title cannot be
    # clipped by the figure edge as it was when matplotlib stole the space
    cax = fig.add_axes([0.915, gs[0].get_position(fig).y0, 0.012,
                        gs[0].get_position(fig).height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title("profile entry\n(units of sigma0)", fontsize=7.5, pad=6)
    cb.ax.tick_params(labelsize=7)
    i1.legend(handles=[Patch(color=CAT_COLOURS[i], label=CAT_LABELS[i]) for i in (1, 2, 3, 0)],
              loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=4, fontsize=9,
              frameon=False)

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"FIGC1_neuron_map.{ext}", dpi=300)
    plt.close(fig)

    # what the inset actually contains, so the caption's claim about it is checkable
    head = msk[:, :INSET_COLS]
    meta = {
        "figure": "Figure C.1",
        "caption": "in the draft's Appendix C.5; not drawn into the image",
        "files": [str(OUT / f"FIGC1_neuron_map.{e}") for e in ("png", "svg")],
        "reference": LAST, "reference_described_as": "end of pretraining (2,034 B tokens)",
        "construct": "For", "twin": "twin1", "orientation": "full page landscape",
        "figsize_inches": [11.2, 7.3], "no_title": True,
        "inset": {"columns": INSET_COLS,
                  "what": "the highest-degree neurons, enlarged, one inset per panel",
                  "n_neurons_in_inset": int(head.size),
                  "n_carriers_in_inset": int(head.sum()),
                  "carrier_share_in_inset": float(head.mean()),
                  "share_of_all_carriers_in_inset": float(head.sum() / msk.sum())},
        "acceptance": {"raster_holds_exactly_the_carrier_set": bool(int(msk.sum()) == K),
                       "n_carriers": int(msk.sum()),
                       "scope_overlap_share_of_K": overlap,
                       "both_scope_panels_present": True},
        "panel_b_categories": CAT_LABELS,
        "degree": {"max": int(M["deg"].max()), "mean": float(M["deg"].mean()),
                   "share_zero": float((M["deg"] == 0).mean())},
        "provenance": provenance([f"profiles/{LAST}/twin1/For.npz",
                                  f"validation/carriers/{LAST}",
                                  f"record/{LAST}/twin1__For_{{A,B}}_acc.npz"],
                                 reference=LAST)}
    (OUT / "FIGC1_neuron_map.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"Figure C.1: landscape {11.2}x{7.3} in, two panels, inset of {INSET_COLS} columns")
    print(f"  the inset holds {head.sum():,} carriers of {head.size:,} neurons "
          f"({head.mean():.1%}), {head.sum() / msk.sum():.1%} of the whole carrier set")
    print(f"  scope overlap {overlap:.4f} of K")
    print(f"-> {OUT / 'FIGC1_neuron_map.png'}")


if __name__ == "__main__":
    main()
