"""The three hooks of §4, and why each is where it is.

**Primary — the inner MLP units.** A forward *pre*-hook on `model.layers.{i}.mlp.down_proj`,
capturing `args[0]`: the gated product act(W_gate x) * (W_up x), 6,144 wide, the vector the
down-projection consumes. Verified against a recomputation from the weights with a maximum
absolute difference of 0.0. NOT `act_fn`'s output, which is act(W_gate x) without the up
factor — a different vector, and not the one zeroing which ablates a neuron.

**Secondary — m_k.** A forward hook on `.mlp`, capturing its 2,240-wide output. It is
accumulated in its own right rather than derived, because m_k = W_down @ h carries through for
the *mean* but not for the spread: the variance of a linear combination needs the covariance
of h, not its diagonal. It is also the v1 basis, so the bridge reads it.

**The robustness arm — the residual stream.** A forward hook on the decoder layer, capturing
its output.

All three are captured at the same positions in the same pass, so nothing is re-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Capture:
    """Sliced activations for one batch: {layer -> tensor[batch, sites, width]}."""

    h: dict[int, torch.Tensor] = field(default_factory=dict)
    mk: dict[int, torch.Tensor] = field(default_factory=dict)
    resid: dict[int, torch.Tensor] = field(default_factory=dict)
    overflow: dict[int, int] = field(default_factory=dict)


FP16_MAX = 65504.0


class Recorder:
    """Attaches the three hooks and slices to the requested positions in-hook.

    Slicing inside the hook is not an optimisation, it is the difference between feasible and
    not: keeping the full sequence for 24 layers at batch 8 and 6,144 units is 580 MB per
    forward pass, against 14 MB for three sites.

    The slices stay **on device** until the pass completes and are then moved in one stacked
    copy per stream. Moving each layer's slice inside its own hook meant 72 device-to-host
    transfers per forward pass — 24 layers times three streams — and each one is an MPS
    synchronisation point. The arithmetic was never the cost; the synchronisation was.
    """

    def __init__(self, model, n_layers: int, want_resid: bool = True,
                 want_mk: bool = False):
        """`want_mk` defaults to **False**, because D1a drops it.

        *"What is dropped, and why each costs nothing that is read: the `mk` stream everywhere
        ... every call site passes `stream="h"`."* The Recorder captured it regardless, and
        `sweep_anchored` wrote `mk|...` accumulators from it, so the record carried two streams
        where the design keeps one. That doubled the accumulator volume -- A21's measured 363 GB
        becomes 726 GB, past option C's whole 702 GB budget -- and doubled the cost of
        `collect()`, which profiling showed to be the remaining bottleneck once the per-layer
        site transfer was fixed. Kept as a flag rather than deleted, because the method paper's
        read-site comparison may want it on a few sets.
        """
        self.model, self.n_layers, self.want_resid = model, n_layers, want_resid
        self._rows = None
        self.want_mk = want_mk
        self.cap = Capture()
        self.sites: torch.Tensor | None = None      # [batch, 3] token indices
        self.handles = []
        layers = model.layers if hasattr(model, "layers") else model.model.layers
        for i, layer in enumerate(layers):
            self.handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(self._pre_h(i)))
            if want_mk:
                self.handles.append(layer.mlp.register_forward_hook(self._post_mk(i)))
            if want_resid:
                self.handles.append(layer.register_forward_hook(self._post_resid(i)))

    def set_sites(self, sites: torch.Tensor, device=None) -> None:
        """Set the read sites for the next forward pass.

        `device` moves the tensor **once**, here. `_gather` used to call `sites.to(t.device)`
        itself, which transferred this tiny tensor from host to device on **every layer of
        every batch** -- 24 times a batch -- and each transfer is a synchronisation point, so
        the GPU stalled 24 times per batch and never pipelined. Profiled on one set: 2.717 s of
        2.865 s, **95% of recording time**, and it is what put an A100 at 1.3% of its bf16 peak
        and the sweep at 548 hours against D5's budget of 7.
        """
        self.cap = Capture()
        self.sites = sites.to(device) if device is not None else sites

    def pause(self) -> None:
        """Make the hooks inert without detaching them.

        The C sweep attaches its own hooks to the same modules and reads every position, not
        three. Left armed, these hooks fire during that pass with the *previous* anchored
        batch's site indices — a batch of 4 with 3 sites gathering from a batch of 2 with 52
        positions — and the run dies 76 sets in. Hook state outliving the pass it belongs to
        is the second bug of this kind, so pausing is explicit rather than implied.
        """
        self.sites = None
        self.cap = Capture()

    @property
    def armed(self) -> bool:
        return self.sites is not None

    def _gather(self, t: torch.Tensor) -> torch.Tensor:
        # t: [batch, seq, width]; sites: [batch, n_sites]
        # No `.to()` here: the tensor is moved once in `set_sites`. `clamp` rather than
        # `clamp_`, because clamping in place would mutate the caller's tensor -- harmless when
        # a fresh copy arrived per layer, a bug now that one tensor is shared across layers.
        idx = self.sites
        if idx.device != t.device:
            idx = idx.to(t.device)
        idx = idx.clamp(max=t.shape[1] - 1)
        # **Advanced indexing, not `gather` with an expanded index.** `gather` needs an index
        # the shape of the *output*, so `expand(-1, -1, 6144)` materialises a [B, 3, 6144]
        # int64 tensor -- 19 MB a layer, 0.45 GB a batch across 24 layers -- to read three
        # positions. Indexing by row takes the same three and allocates nothing: measured
        # 55.5 ms -> 11.0 ms per batch of 24 layers, with identical output.
        if self._rows is None or self._rows.shape[0] != idx.shape[0] \
                or self._rows.device != t.device:
            self._rows = torch.arange(idx.shape[0], device=t.device).unsqueeze(1)
        return t[self._rows, idx]

    def _pre_h(self, i):
        def fn(_m, args):
            if self.sites is None:
                return None
            # **not `.float()`**: widening bf16 to fp32 here doubles every tensor on the
            # device and doubles the bytes crossing PCIe, for a quantity D1a stores
            # in fp16 anyway. The widening happens on the host, after the transfer.
            self.cap.h[i] = self._gather(args[0]).detach()
            return None
        return fn

    def _post_mk(self, i):
        def fn(_m, _args, out):
            if self.sites is None:
                return None
            t = out[0] if isinstance(out, tuple) else out
            self.cap.mk[i] = self._gather(t).detach()
            return None
        return fn

    def _post_resid(self, i):
        def fn(_m, _args, out):
            if self.sites is None:
                return None
            t = out[0] if isinstance(out, tuple) else out
            self.cap.resid[i] = self._gather(t).detach()
            return None
        return fn

    def collect(self) -> dict[str, "np.ndarray"]:
        """One stacked host copy per stream, and the fp16 overflow count for h.

        **Profiled: this was 2.6 s of 2.9 s** once the per-layer site transfer was fixed, and on
        an A100 it held a batch of 128 to 5.3 s where the forward pass should be about 50 ms.
        Three costs, all removed here.

        *The widening.* The hooks no longer call `.float()`, so the tensors stay in the model's
        own dtype and the stack is half the size.

        *The overflow check.* `(stacked.abs() > FP16_MAX).sum()` allocated an `abs` copy and a
        56-million-element boolean, then `int()` forced a host synchronisation. `min` and `max`
        are two scalar reductions with no allocation, and `|min|` against `max` bounds the
        magnitude exactly as well. It no longer counts *how many* values overflow, only whether
        any do -- and the pilot measured **zero at every reference, layer and site**, so the
        count was always zero and the question is whether the record is usable, not by how much.

        *The transfer.* Cast to fp16 before crossing PCIe rather than after: D1a stores means in
        fp16, so the fp32 bytes were being moved only to be narrowed later.
        """
        import numpy as np

        out = {}
        for key, store in (("h", self.cap.h), ("mk", self.cap.mk),
                           ("resid", self.cap.resid)):
            if not store:
                continue
            stacked = torch.stack([store[i] for i in range(self.n_layers)], dim=1)
            if key == "h":
                lo = float(stacked.min())
                hi = float(stacked.max())
                if max(abs(lo), abs(hi)) > FP16_MAX:
                    self.cap.overflow[-1] = self.cap.overflow.get(-1, 0) + 1
            host = stacked.to(torch.float16).cpu().numpy()
            out[key] = host.astype(np.float32, copy=False)
        return out

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
