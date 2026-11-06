r"""Figure 1 — the calibrated concept profile end to end, in six steps.

The schematic the draft reserves at the head of section 2, for the reader who wants the method
in two minutes. Six steps, left to right, with `for` as the running example throughout so the
same construct can be followed from a prompt to a reading across checkpoints.

**One row, six boxes, consecutive arrows.** Each step takes the one before it: the generator
makes the batteries, the model turns them into raw readings, the yardstick calibrates those
into a profile, a twin battery gives the unit the profile is read in, the nine checks license
it, and the whole thing repeats at every checkpoint. Reading left to right is reading the
method in order, with nothing to track up or down the page.

The coordinates are isotropic -- 100 x 23.1 units in 7.4 x 1.71 inches -- so a box's drawn
proportions are its declared ones. At six boxes across the text width each is about an inch
wide, so every line of body text is held under eighteen characters and the type is set large
enough to survive the scaling to \textwidth.

Nothing here is measured; it is a diagram of the method, and every figure in it that could be
mistaken for a result is instead a label.

    python experiments/figure_method.py
"""
from __future__ import annotations

import json
import time

from experiments.atlas import OUT, provenance

INK = "#1a1a1a"
MUTED = "#6b6b6b"
ACCENT = "#b35806"
SPINE = "#1b4f72"
PAPER = "#f4f1ec"

CAPTION = (
    "Figure 1. The calibrated concept profile in six steps, left to right. (1) A generator "
    "with a declared distribution draws the batteries: set A (the construct in its role), "
    "set B (its token out of role) and set C (ordinary code). (2) The model at one "
    "checkpoint is run on all three, giving raw activations at each prompt's read site. "
    "(3) Per neuron, A minus B in units of that neuron's own spread on C at the same kind of "
    "position -- the yardstick -- gives the calibrated profile. (4) Steps 1 to 3 are repeated "
    "on a second battery drawn from the same distribution, and the spread between the two "
    "profiles is the noise unit every later figure is read in. (5) Nine checks on the "
    "profiles give the licence: two must pass, seven are reported. (6) The same battery and "
    "the same process at every checkpoint make readings comparable across training, with a "
    "stated resolution.")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 1.71))
    ax.set_xlim(0, 100)
    ax.set_ylim(-5.0, 18.1)
    ax.axis("off")

    W, GAP, Y, H = 14.0, 2.8, 1.3, 14.5
    xs = [1.0 + k * (W + GAP) for k in range(6)]

    def box(x, title, lines, edge=SPINE, fill="white"):
        ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0.4,rounding_size=0.9",
                                    linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=2))
        cy = Y + H - 1.8
        for ln in title:
            ax.text(x + W / 2, cy, ln, ha="center", va="top", fontsize=7.0,
                    fontweight="bold", color=INK, zorder=3)
            cy -= 2.5
        cy -= 0.5
        for ln in lines:
            ax.text(x + W / 2, cy, ln, ha="center", va="top", fontsize=6.8,
                    color=MUTED, zorder=3)
            cy -= 2.25

    box(xs[0], ["prompt generator"],
        ["batteries A, B, C", "are generated"])
    box(xs[1], ["raw readings"],
        ["the model is run", "on A, B, C;", "raw readings"])
    box(xs[2], ["the profile"],
        ["calibrated readings:", "A − B in units", "of C's spread"])
    box(xs[3], ["the noise unit"],
        ["steps 1–3 repeated", "on a twin battery;", "the spread between", "the two profiles"],
        edge=ACCENT)
    box(xs[4], ["the licence"],
        ["nine checks", "on the profiles"], edge=ACCENT)
    box(xs[5], ["readings across", "training"],
        ["the same battery", "and process at", "every checkpoint"],
        edge=SPINE, fill=PAPER)

    for k, x in enumerate(xs):
        ax.add_patch(plt.Circle((x + 0.7, Y + H + 0.3), 1.15, facecolor=SPINE,
                                edgecolor="none", zorder=4))
        ax.text(x + 0.7, Y + H + 0.3, str(k + 1), ha="center", va="center", fontsize=6.0,
                color="white", fontweight="bold", zorder=5)

    # consecutive, 1->2 ... 5->6: every step takes the one before it
    for k in range(5):
        ax.add_patch(FancyArrowPatch((xs[k] + W + 0.5, Y + H / 2),
                                     (xs[k + 1] - 0.5, Y + H / 2),
                                     arrowstyle="-|>", mutation_scale=7, linewidth=1.0,
                                     color=SPINE, zorder=1))

    # 6 back to 2, dashed: the battery and the process are fixed, so what repeats across
    # training is the recording onwards -- not the generator, which runs once. Drawn as an
    # explicit three-segment path; matplotlib's "bar" connector puts the crossbar a fraction
    # of the span away, which at a 70-unit span took it off the figure entirely.
    DASH, y_bar = (0, (2.4, 1.8)), Y - 3.2
    x6, x2 = xs[5] + W / 2, xs[1] + W / 2
    ax.plot([x6, x6], [Y - 0.4, y_bar], ls=DASH, lw=0.9, color=MUTED, zorder=1)
    ax.plot([x6, x2], [y_bar, y_bar], ls=DASH, lw=0.9, color=MUTED, zorder=1)
    ax.add_patch(FancyArrowPatch((x2, y_bar), (x2, Y - 0.5), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=0.9, linestyle=DASH,
                                 color=MUTED, zorder=1))
    ax.text((x2 + x6) / 2, y_bar - 0.7, "at each checkpoint", ha="center", va="top",
            fontsize=6.4, color=MUTED, style="italic")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"FIG1_method.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    meta = {"figure": "Figure 1", "caption": CAPTION,
            "files": [str(OUT / f"FIG1_method.{e}") for e in ("png", "svg")],
            "what": "a schematic of the method; nothing in it is measured",
            "steps": ["prompt generator", "raw readings", "the profile",
                      "the noise unit", "the licence", "readings across training"],
            "layout": "one row of six, arrows consecutive 1->2 ... 5->6, and a dashed "
                      "return 6 -> 2 labelled 'at each checkpoint'",
            "running_example": "for",
            "figsize_inches": [7.4, 1.71],
            "axes_units": [100, 23.1],
            "isotropic": True,
            "cropped": "bbox_inches=tight: no whitespace above or below",
            "provenance": provenance(["none: a diagram, not a reading"])}
    (OUT / "FIG1_method.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"Figure 1: six steps, {(time.time() - t0):.1f}s")
    print(f"-> {OUT / 'FIG1_method.png'}")


if __name__ == "__main__":
    main()
