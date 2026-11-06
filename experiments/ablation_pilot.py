r"""D3's ablation pilot — item 25: does zeroing a construct's carrier set move its loss?

    python experiments/ablation_pilot.py --constructs For If ListComp \
        --references pretrain_iter_0001000 pretrain_iter_0020000

D3's gate, in the design's own words: *"The pass is built only if the pilot moves the loss beyond
the random control, and with whichever unit set does."* So this is not a measurement of the
construct -- it is a measurement of whether an instrument is worth building, and its output is a
yes or a no plus which unit set earns it.

**Three unit sets, because the design names three and expects them to disagree.** The carrier set
at the **verdict layer** (k = 307 units), the carrier set over the **product space** (k = 7,373
over 24 x 6,144), and a **size-matched random set** drawn from the same space. The design's own
reason for expecting the layer version to fail: *"The concept's norm mass is spread across layers
(centre near layer 10 on the pilot), so a single-layer ablation may be a measurement null; the
product space is the right unit set for ablation even though it changes nothing for the turnover
curve."*

**What is zeroed, and where.** The carrier set is a set of (layer, unit) pairs in the MLP's gated
product -- the same site every profile is read at, the pre-hook on `mlp.down_proj` whose `args[0]`
is the gated product. Zeroing is done there, in the forward pass, so the ablation propagates
through `down_proj` and everything after it, which is what makes it an intervention rather than a
re-read.

**The reliance criterion.** C2's clause: the ablation effect must exceed the size-matched random
set's by **more than one twin unit** at both ends of the span. A pilot cannot supply a twin unit
-- that needs the second battery -- so it reports the effect in nats and as a multiple of the
random control's own spread across draws, and names what the sweep would have to beat.

Writes `validation/ablation/PILOT_<reference>.json`.
"""

from profilebattery.paths import data_root
import argparse
import gzip
import json
import os
import pathlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from profilebattery.record.sweep import REPO

D = data_root()
OUT = D / "validation/ablation"
TOK_REV = "pretrain_iter_0008000"
K_FRAC = 0.05
N_RANDOM_DRAWS = 5


PROFILES = D / "profiles"


def verdict_layers() -> dict:
    """`{construct: layer}` from the carrier index, which is where the rule's own answer lives.

    `VERDICT_LAYERS.json` is the argmax of the concept-profile norm over layers, computed from the
    tree at `pretrain_iter_0484865` with no fallback -- the plan's rule. A38's sensitivity measured
    it to move for 70 of 169 constructs when the last layer is excluded, so it is fragile; that is
    a reason to report the layer ablation beside the product-space one, not a reason to substitute
    a constant for it.
    """
    for p in (D / "validation/carriers/VERDICT_LAYERS.json",
              D / "validation/carriers/VERDICT_LAYERS.json"):
        if p.exists():
            lay = json.loads(p.read_text()).get("layer")
            if isinstance(lay, dict) and lay:
                return lay
            raise ValueError(f"{p} carries no per-construct `layer` map")
    # **Refused, not defaulted.** The first attempt at this returned `{}` and the caller fell back
    # to layer 15 for every construct -- which is exactly the run it was meant to replace, and it
    # cost a GPU session and wrote a file whose `verdict_layer` field claimed per-construct layers
    # it had not used. A layer test at the wrong layer is worse than no layer test.
    raise FileNotFoundError(
        "VERDICT_LAYERS.json is not present, so no construct's own verdict layer is known. "
        "Fetch validation/carriers/VERDICT_LAYERS.json from the dataset, or pass "
        "--verdict-layer N to test one layer for every construct deliberately.")


def carrier_sets(construct, reference, twin, n_layers, width, verdict_layer, rng):
    """The three unit sets D3 names, as boolean masks over (layer, unit).

    **The carrier set is read from the recorded concept profile, not drawn.** It is the top
    `K_FRAC` of the squared mass of pi(A) - pi(B), which is C0's own definition, and it exists
    only where a reference has been recorded. The first version of this script fell back to a
    seeded draw when it could not find one, and the smoke test showed what that costs: ablating
    a random 5% against another random 5% gave carrier -0.0218 against control +0.0159 ± 0.0122,
    a null by construction and not D3's question. A stand-in carrier cannot pilot an ablation.

    So the pilot runs on the **pilot record**, whose profiles exist at both of D3's checkpoints.
    Nothing in the design ties the pilot to the phase battery -- it is three constructs at two
    checkpoints -- and the pilot keeps this gate independent of a phase recording that does not
    exist and that this gate is meant to precede.
    """
    f = PROFILES / reference / twin / f"{construct}.npz"
    if not f.exists():
        raise FileNotFoundError(
            f"no recorded profile at {f}: the carrier set is the top {K_FRAC:.0%} of the "
            f"concept profile and cannot be invented")
    concept = np.load(f)["concept"]
    if concept.shape != (n_layers, width):
        raise ValueError(f"{f}: concept is {concept.shape}, expected {(n_layers, width)}")
    mass = concept.astype(np.float64) ** 2
    k_product = max(1, int(round(K_FRAC * n_layers * width)))
    k_layer = max(1, int(round(K_FRAC * width)))
    product = np.zeros((n_layers, width), dtype=bool)
    product.flat[np.argsort(mass.ravel())[::-1][:k_product]] = True
    layer = np.zeros((n_layers, width), dtype=bool)
    layer[verdict_layer, np.argsort(mass[verdict_layer])[::-1][:k_layer]] = True
    randoms = []
    for _ in range(N_RANDOM_DRAWS):
        m = np.zeros((n_layers, width), dtype=bool)
        m.flat[rng.choice(n_layers * width, size=k_product, replace=False)] = True
        randoms.append(m)
    return {"carrier_product": product, "carrier_layer": layer}, randoms


class Ablator:
    """Zeroes a mask of (layer, unit) in the gated product, at the read site itself."""

    def __init__(self, model, n_layers):
        self.mask = None
        self.handles = []
        for i in range(n_layers):
            self.handles.append(
                model.layers[i].mlp.down_proj.register_forward_pre_hook(self._pre(i)))

    def _pre(self, i):
        def hook(_mod, inputs):
            if self.mask is None:
                return None
            row = self.mask[i]
            if not row.any():
                return None
            x = inputs[0].clone()
            x[..., torch.from_numpy(np.where(row)[0])] = 0.0
            return (x,) + tuple(inputs[1:])
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def mark_positions(rows, construct, tok):
    """Derive the diagnostic positions for prompts that carry none.

    **The pilot battery predates item 87**, which is what put position marks into a prompt, so
    its records have no `positions` field and D3's loss -- read at the diagnostic positions --
    came out NaN on every one of them. The phase battery has the marks and no recording; the
    pilot has the recording and no marks. `positions_for` resolves it: it works on any text, so
    the marks are derived here at runtime, with the position finder as it stands today --
    including the corrections of 1 October, which the stored marks of either battery predate in
    part.
    """
    from profilebattery.generate.positions import positions_for
    out = 0
    for r in rows:
        if r.get("positions"):
            continue
        pieces = [tok.convert_ids_to_tokens(i)
                  for i in tok(r["text"], add_special_tokens=False)["input_ids"]]
        ct = r.get("site_construct_token")
        if ct is None:
            continue
        got = positions_for(r["text"], construct, pieces, ct)
        r["positions"] = [p.as_dict() for p in got or []]
        out += len(r["positions"])
    return out


@torch.no_grad()
def loss_at_positions(model, tok, rows, device, batch=8):
    """Mean next-token loss over each prompt's marked obliged and dependent positions."""
    total, n = 0.0, 0
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        texts = [r["text"] for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        out = model(ids).logits.float()
        logp = torch.log_softmax(out[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for b, r in enumerate(chunk):
            idx = [m["token_index"] for m in (r.get("positions") or [])
                   if m.get("token_index") is not None
                   and 0 < m["token_index"] <= tgt.shape[1]
                   and m.get("matched") is not False]
            if not idx:
                continue
            take = torch.tensor([i - 1 for i in idx], device=device)
            total += float(-picked[b].index_select(0, take).sum())
            n += len(idx)
    return total / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constructs", nargs="*", default=["For", "If", "ListComp"])
    ap.add_argument("--references", nargs="*",
                    default=["pretrain_iter_0001000", "pretrain_iter_0020000"])
    # the PILOT battery by default: its profiles exist at both of D3's checkpoints,
    # and the carrier set has to be read from a recording rather than drawn
    ap.add_argument("--battery", default=str(D / "battery"))
    ap.add_argument("--per-set", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=N_RANDOM_DRAWS,
                    help="size-matched control draws; 5 of the 8 passes per "
                         "construct are these, so it is the knob that matters")
    ap.add_argument("--verdict-layer", type=int, default=None,
                    help="one layer for every construct. Omit it and each construct's OWN "
                         "verdict layer is used, read from the carrier index's "
                         "VERDICT_LAYERS.json -- which is the only form of this test that means "
                         "anything: the first pilot run used 15 for all three, and 15 is the "
                         "verdict layer of `For` alone (`If` is 4, `ListComp` is 5), so the "
                         "layer null it reported was established for one construct and untested "
                         "for the other two")
    ap.add_argument("--twin", default="twin1")
    args = ap.parse_args()
    # cuda first: this runs on an A100 for the real pilot, and the MPS-or-CPU form it had
    # would have picked CPU there -- a 1.5B model at fp32 over eight forward passes per
    # construct, which is how the local run spent 33 minutes without finishing one.
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    vlayers = {} if args.verdict_layer is not None else verdict_layers()
    if args.verdict_layer is None:
        missing = [c for c in args.constructs if c not in vlayers]
        if missing:
            raise SystemExit(
                f"no verdict layer recorded for {missing}: VERDICT_LAYERS.json covers "
                f"{len(vlayers)} constructs. Pass --verdict-layer N to test one layer for all "
                f"of them deliberately, or name constructs the index carries.")
        print("verdict layer per construct:", {c: vlayers[c] for c in args.constructs},
              flush=True)
    tok = AutoTokenizer.from_pretrained(REPO, revision=TOK_REV, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    B = pathlib.Path(os.path.expanduser(args.battery))

    globals()['N_RANDOM_DRAWS'] = args.n_random
    for ref in args.references:
        model = AutoModelForCausalLM.from_pretrained(
            REPO, revision=ref, dtype=dtype, trust_remote_code=True).to(device).eval()
        core = model.model
        n_layers = core.config.num_hidden_layers
        width = core.config.intermediate_size
        abl = Ablator(core, n_layers)
        res = {"reference": ref, "device": device, "k_frac": K_FRAC,
               "n_layers": n_layers, "width": width,
               # the layer actually used, per construct -- not a claim about where it came
               # from: the claim is what was wrong in the first `_ownlayer` artefacts
               "verdict_layer_used": {c: (args.verdict_layer if args.verdict_layer is not None
                                          else vlayers[c]) for c in args.constructs
                                      if args.verdict_layer is not None or c in vlayers},
               "verdict_layer_rule": ("one layer for every construct, passed on the command line"
                                      if args.verdict_layer is not None else
                                      "each construct's own, from VERDICT_LAYERS.json"),
               "carrier_source": f"top {K_FRAC:.0%} of the recorded concept profile "
                                 f"at profiles/{ref}/{args.twin}",
               "constructs": {}}
        print(f"\n{ref} | {device} | {n_layers} layers x {width}", flush=True)
        for c in args.constructs:
            f = B / "twin1" / f"{c}_A.jsonl.gz"
            if not f.exists():
                print(f"  {c}: no A set"); continue
            rows = [json.loads(l) for l in gzip.open(f, "rt")][: args.per_set]
            derived = mark_positions(rows, c, tok)
            n_pos = sum(len(r.get("positions") or []) for r in rows)
            if n_pos == 0:
                print(f"  {c}: no diagnostic positions -- skipped")
                continue
            rng = np.random.default_rng(abs(hash(c)) % 2 ** 31 if False else 7)
            sets, randoms = carrier_sets(c, ref, args.twin, n_layers, width,
                                         (args.verdict_layer if args.verdict_layer is not None
                                          else vlayers[c]), rng)
            abl.mask = None
            base = loss_at_positions(model, tok, rows, device, args.batch)
            out = {"baseline_nats": base, "n_prompts": len(rows),
                   "n_positions": n_pos, "positions_derived": derived}
            for name, mask in sets.items():
                abl.mask = mask
                out[name] = loss_at_positions(model, tok, rows, device, args.batch) - base
            rnd = []
            for m in randoms:
                abl.mask = m
                rnd.append(loss_at_positions(model, tok, rows, device, args.batch) - base)
            abl.mask = None
            out["random_mean"] = float(np.mean(rnd))
            out["random_sd"] = float(np.std(rnd, ddof=1))
            out["product_over_random_in_sd"] = (
                (out["carrier_product"] - out["random_mean"]) / out["random_sd"]
                if out["random_sd"] > 0 else float("nan"))
            res["constructs"][c] = out
            print(f"  {c:12} base {base:.4f}  product {out['carrier_product']:+.4f}  "
                  f"layer {out['carrier_layer']:+.4f}  random {out['random_mean']:+.4f}"
                  f" ± {out['random_sd']:.4f}", flush=True)
        abl.remove()
        del model
        OUT.mkdir(parents=True, exist_ok=True)
        # **the two layer rules write to different files.** The first pilot run used layer 15
        # for every construct and its artefact is the evidence A43's table rests on; a re-run
        # with each construct's own layer must not overwrite it, or the comparison that shows
        # which layer was tested disappears.
        tag = "" if args.verdict_layer is not None else "_ownlayer"
        p = OUT / f"PILOT_{ref}{tag}.json"
        p.write_text(json.dumps(res, indent=1))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
