r"""The sweep's orchestrator — items 67 and 89, built against amendment A21.

    python experiments/run_colab.py --plan licence      # Stage 3: 80, 66's activation half, 12
    python experiments/run_colab.py --plan sweep        # Stage 4: 59 references in D5's order

`run_sweep.py` records **one** reference. This drives it across the plan, and adds the five
things the design asks of a long run and `run_sweep` has none of:

**1 · The device is checked at session start and the run aborts if it is wrong.** Compute units
buy a budget, not a reserved GPU. A 14-hour run on a T4 that should have been an A100 is the
failure mode, and the check costs a second. *`ablation_pilot` selected `mps` or `cpu` and would
have run a 1.5B model at fp32 on CPU here — which is why this is a function and not a comment.*

**2 · Three watched numbers, aborting out of range.** Twin agreement (a collapse means the
battery is broken), the concept norm at the verdict layer (near zero means the read site is
wrong), the carrier-set overlap with the previous reference (near 1.0 or 0.0 means a set is
being compared with itself or with noise). Out of range at reference three, not 702 GB of
unusable record at reference fifty-nine.

**3 · Resumable upload, one reference at a time.** Each reference is uploaded to HuggingFace as
it is recorded and the next records while the previous sends; Drive holds one reference. A
reference already present upstream is skipped, so a dropped session resumes rather than restarts.

**4 · The upload rate is measured on the first reference.** This is Cowork's condition on
amendment A21's option C: option C adds 203 GB of upload, and if the measured rate makes it
binding the run falls back to option B's narrower scope key with the regret logged. Measured,
then reported, before the second reference starts.

**5 · Write after every unit** — item 89's second lesson. A long pass that produces nothing
until it ends is the failure the design exists to prevent, so each set's shard is written and
flushed as it completes rather than held for the reference.

Storage follows A21: the per-prompt hidden stream only for the 32 constructs of
`validation/battery/PER_PROMPT_SUBSET.json`, the residual stream for all of P₁, four checkpoints.
"""

from profilebattery.paths import data_root, dataset_repo
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

from profilebattery.record.remote import quiet_hub
from profilebattery.record.sweep import RESID_REFS

D = data_root()
#: the recorder writes here -- **local disk**, never the Drive mount (see run_sweep's note).
#: Set the same variable the recorder reads, so the two cannot disagree.
REC = pathlib.Path(os.environ.setdefault("PB_RECORD_DIR",
                                         "/content/record" if pathlib.Path("/content").exists()
                                         else str(D / "record")))
#: where a finished reference is moved to, if anywhere, once it is uploaded
ARCHIVE = D / "record"
#: **On the shared drive, not `/content`.** The progress file exists for visibility *after a
#: disconnect*, and it was being written to the local disk a reclaim destroys -- so the one
#: artefact whose whole purpose is surviving a lost session did not. The shared units ledger
#: lives here for the same reason, and because sibling sessions have to see each other's.
RUNDIR = D / "runs"
#: the battery every reference must have been recorded against
BATTERY = D / "phase_battery"
#: how many sets a complete reference holds, read from the battery rather than assumed
EXPECTED_SETS = len(json.loads((BATTERY / "CONTENT_HASHES.json").read_text())) \
    if (BATTERY / "CONTENT_HASHES.json").exists() else 0
#: the four checkpoints at which per-prompt vectors are kept (D1, A21)
#: D1's four checkpoints, imported rather than restated. They were written out here *and*
#: in `run_sweep`, and the two lists governed different things — this one whether per-prompt
#: data is kept, that one whether the residual stream is recorded at all — so a change to one
#: would have silently given a reference per-prompt residual vectors with no residual stream.
PER_PROMPT_REFS = RESID_REFS
MIN_TWIN_AGREEMENT = 0.30
MIN_CONCEPT_NORM = 1e-3
OVERLAP_BOUNDS = (0.02, 0.995)


def check_device(allow=("A100", "L4", "H100"), smoke=0):
    """Abort unless the assigned GPU is one the budget was costed on.

    `smoke` lifts the requirement, because the orchestrator could otherwise only be exercised
    on an A100 — and every defect of this week was in the orchestration and was found by
    running it. A smoke run forces `--no-upload` and a small `--limit-sets`, so it cannot be
    mistaken for the real thing, and says so on every line.
    """
    import torch
    if smoke:
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"SMOKE RUN on {dev}: {smoke} sets per reference, no upload. Not a sweep.",
              flush=True)
        return f"smoke:{dev}"
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: the sweep is costed on an A100 and will not run here")
    name = torch.cuda.get_device_name(0)
    if not any(a.lower() in name.lower() for a in allow):
        raise SystemExit(f"device is {name!r}, not one of {allow}: aborting rather than "
                         f"spending the month on the wrong GPU")
    free, total = torch.cuda.mem_get_info()
    print(f"device {name}  memory {total/1e9:.0f} GB, {free/1e9:.0f} free", flush=True)
    return name


def plan_references(plan: str):
    """The references to record, in D5's order.

    D5 puts the **second seed first and the last pretraining checkpoint second**, so the licence
    checks and C0.4's both-seeds requirement are satisfied on the record's own opening rather
    than by a separate pass. That is why Stage 3's A100 work is the beginning of the sweep and
    not a prelude to it.
    """
    sched = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    refs = sched["reference_set"]
    if plan == "licence":
        # item 12's second seed, then the last pretraining checkpoint: what 66's activation
        # half and C0.4 need, and nothing more
        return ["untrained_seed2", f"pretrain_iter_{sched['pretrain_steps'][-1]:07d}"]
    dense = [f"pretrain_iter_{s:07d}" for s in refs["dense"]]
    tail = [f"pretrain_iter_{s:07d}" for s in refs["tail"]]
    anneal = [f"anneal_iter_{s:07d}" for s in refs["anneal"]]
    last = f"pretrain_iter_{sched['pretrain_steps'][-1]:07d}"
    # D5's order, then **deduplicated while preserving it**: the last pretraining checkpoint is
    # named explicitly because D5 puts it second, and it is also the final member of the tail,
    # so the naive concatenation listed it twice and made the plan 62 references where the
    # reference set is 59 plus the two untrained seeds.
    ordered = ["untrained_seed2", last, "untrained"] + dense + tail + anneal
    seen, out = set(), []
    for r in ordered:
        if r not in seen:
            seen.add(r)
            out.append(r)
    expected = refs["n"] + 2          # the 59 of the reference set, plus both untrained seeds
    if len(out) != expected:
        raise ValueError(f"plan has {len(out)} references, expected {expected} "
                         f"({refs['n']} in the reference set plus two untrained seeds)")
    return out


def per_prompt_for(reference: str) -> bool:
    return reference in PER_PROMPT_REFS


def subset_constructs():
    p = D / "validation/battery/PER_PROMPT_SUBSET.json"
    return set(json.loads(p.read_text())["subset"]) if p.exists() else None


#: one fetch per reference: `already_recorded` and `watched` both want the manifest, and
#: downloading it twice printed the same line twice and asked the Hub twice.
_MANIFEST_CACHE: dict[str, dict] = {}


def local_or_remote_manifest(reference: str, repo: str | None = None) -> dict | None:
    """This reference's manifest, from local disk or else from the dataset.

    **The local copy is deleted on purpose.** Colab's disk is ~110 GB against a 702 GB sweep,
    so `main` removes a reference's directory once the upload is confirmed. The resume check
    then had nothing to read and would re-record a reference that is safely upstream — five
    hours of A100 per reference, silently, on any rerun after a disconnect. That is the
    expensive half of this: the whole point of a resume check is the run that was interrupted.

    So: local manifest first, and the dataset's copy as the fallback. The manifest is a few
    kilobytes, and `hf_hub_download` on one file is cheap enough to do per reference. A failure
    to reach the Hub is reported and treated as "not recorded", which costs compute and never
    costs a record.
    """
    cached = _MANIFEST_CACHE.get(reference)
    if cached is not None:
        return cached
    m = REC / reference / "MANIFEST.json"
    if m.exists():
        try:
            return json.loads(m.read_text())
        except (OSError, json.JSONDecodeError):
            print(f"  {reference}: local manifest unreadable, trying the dataset", flush=True)
    if repo is None:
        return None
    try:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(repo_id=repo, repo_type="dataset",
                            filename=f"record/{reference}/MANIFEST.json",
                            token=os.environ.get("HF_TOKEN"))
        d = json.loads(pathlib.Path(f).read_text())
        print(f"  {reference}: manifest read from the dataset (local copy was freed)",
              flush=True)
        _MANIFEST_CACHE[reference] = d
        return d
    except Exception as e:                      # noqa: BLE001 -- any Hub or parse failure
        print(f"  {reference}: no local manifest and the dataset's could not be read "
              f"({type(e).__name__}); treating as not recorded", flush=True)
        return None


def already_recorded(reference: str, repo: str | None = None) -> bool:
    """Whether this reference is recorded **against the battery now in use**.

    The check was the mere existence of a `MANIFEST.json`, and that is how a wrong record passed
    as done: an aborted run had recorded `untrained_seed2` against the *pilot* battery's 212
    sets, left a manifest behind, and the next run skipped it and uploaded 2.6 GB of the wrong
    artefact to the dataset under the right name. A manifest is evidence that *something* was
    recorded, not that the right thing was.

    So the manifest's own `battery_hashes` path and set count are compared with the battery
    being used. A mismatch is reported and the reference re-recorded rather than trusted.
    """
    d = local_or_remote_manifest(reference, repo)
    if d is None:
        return False
    want = str(BATTERY / "CONTENT_HASHES.json")
    got = d.get("battery_hashes")
    if got != want:
        print(f"  {reference}: recorded against {got!r}, not {want!r} -- re-recording")
        return False
    n_sets = len(d.get("sets") or {})
    if n_sets < EXPECTED_SETS:
        print(f"  {reference}: {n_sets} sets recorded, expected {EXPECTED_SETS} "
              f"-- incomplete, re-recording")
        return False
    # **And whether it carries the per-prompt arms this reference is supposed to carry.**
    # The same failure as the pilot-battery one, one layer along: a manifest written before
    # A21's decisions 2 and 3 were implemented is complete, hashed to the right battery, and
    # *missing C9's readability-probe input* on all 169 constructs of P1 -- and this function
    # would have skipped the reference and left the gap permanent, since a per-prompt vector
    # not recorded cannot be recovered by any later compute.
    #
    # An old-form record is accepted where the plan asks nothing of it that it lacks: a
    # reference outside the four checkpoints keeps no per-prompt data in either form, so its
    # oversized accumulators are waste and not a gap. One inside them is re-recorded.
    if per_prompt_for(reference):
        # **Both forms.** `per_prompt` is a list of arm names in manifests written after
        # 06295ad and the *shapes dict* in those written before it, because `stats` used to
        # overwrite `info` under that key. Reading only the list form made reference 2 of the
        # licence pass -- complete, correct and 107 GB -- look as though it had no per-prompt
        # arms, and it would have been re-recorded: 2.5 h and ~17 units for nothing.
        arms = set()
        for v in (d.get("sets") or {}).values():
            pp = v.get("per_prompt")
            if isinstance(pp, (list, tuple, dict)):
                arms |= set(pp)
            shapes = v.get("per_prompt_shapes")
            if isinstance(shapes, dict):
                arms |= set(shapes)
        if not {"h", "resid"} <= arms:
            print(f"  {reference}: one of D1's four checkpoints, recorded with per-prompt "
                  f"arms {sorted(arms) or 'none'} -- A21 decisions 2 and 3 need both "
                  f"'h' (32 constructs) and 'resid' (all of P1), and the residual arm is "
                  f"C9's only possible input. Re-recording.", flush=True)
            return False
    return True


def watched(reference: str, repo: str | None = None) -> dict:
    """The three numbers, read from the reference's own manifest after it is recorded.

    **Two defects, found on 4 Oct when the licence pass aborted at reference 1.**

    The manifest was read from local disk, exactly as `already_recorded` used to be, so a
    reference whose local copy had been freed after a confirmed upload failed the gate with "no
    manifest" -- and the gate aborts the run. That is the fix here: the same local-or-dataset
    lookup.

    The deeper one: **`run_sweep` writes no `watched` key**, so `d.get("watched")` has always
    been `None`, `bad` has always been empty, and this gate has always returned `ok=True`
    vacuously. The three numbers that were to abort the sweep at reference one rather than at
    fifty-nine have never been computed on any reference. Computing them needs profiles built
    against the C yardstick, which is `run_validation`'s work and not a line in the recorder, so
    it is item 103 and not a silent patch. Until then the absence is **reported and passed**,
    which is what the code already did -- the change is that it now says so instead of looking
    like a check.
    """
    d = local_or_remote_manifest(reference, repo)
    if d is None:
        return {"ok": False, "why": "no manifest, locally or in the dataset"}
    if "watched" not in d:
        # a reference recorded before item 103 landed. Reported, and passed, because the
        # alternative is refusing to proceed past the pilot and the licence pass.
        return {"ok": True, "values": None,
                "why": "NOT CHECKED: recorded before item 103; no watched numbers in the "
                       "manifest. Every reference from the sweep on carries them."}
    w = d["watched"]
    if not w.get("n_sampled"):
        # **Zero constructs sampled is not three numbers in range.** This gate returned
        # ok=True on every reference ever recorded because the key was absent and the loop
        # over it was empty; it must not now do the same because the key is present and the
        # numbers are None.
        return {"ok": False, "values": w,
                "why": f"watched numbers could not be computed over any construct"
                       f"{': ' + w['error'] if w.get('error') else ''}"}
    bad = []
    for name, v in w.items():
        if name == "twin_agreement" and v is not None and v < MIN_TWIN_AGREEMENT:
            bad.append(f"twin agreement {v:.3f} < {MIN_TWIN_AGREEMENT}")
        if name == "concept_norm" and v is not None and v < MIN_CONCEPT_NORM:
            bad.append(f"concept norm {v:.2e} < {MIN_CONCEPT_NORM}")
        if name == "carrier_overlap" and v is not None and not (
                OVERLAP_BOUNDS[0] <= v <= OVERLAP_BOUNDS[1]):
            bad.append(f"carrier overlap {v:.3f} outside {OVERLAP_BOUNDS}")
    return {"ok": not bad, "why": "; ".join(bad), "values": d.get("watched")}


def upload(reference: str, repo: str, token: str | None) -> dict:
    """Upload one reference's directory, and time it. Skips what is already upstream."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    src = REC / reference
    if not src.is_dir():
        # the reference was already upstream and its local copy freed, so there is nothing
        # here to send. `upload_folder` raises on a missing directory, which is how this
        # aborted the licence pass on 4 Oct.
        return {"reference": reference, "bytes": 0, "seconds": None, "MB_per_s": None,
                "note": "nothing local to upload; already in the dataset"}
    size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    if size < 1e7:
        # the recorder's own flushes already sent the shards; only the manifest is left, and
        # timing a 20 kB upload would report a meaningless rate into the option-C condition
        api.upload_folder(folder_path=str(src), path_in_repo=f"record/{reference}",
                          repo_id=repo, repo_type="dataset")
        return {"reference": reference, "bytes": size, "seconds": None,
                "MB_per_s": None, "note": "flushed during recording"}
    t0 = time.time()
    api.upload_folder(folder_path=str(src), path_in_repo=f"record/{reference}",
                      repo_id=repo, repo_type="dataset")
    dt = time.time() - t0
    rate = size / dt / 1e6 if dt else 0.0
    return {"reference": reference, "bytes": size, "seconds": round(dt, 1),
            "MB_per_s": round(rate, 2)}


def report_upload_rate(first: dict, total_gb: float = 702.0):
    """Cowork's condition on A21's option C, answered with the measured rate."""
    rate = first["MB_per_s"]
    if rate <= 0:
        print("upload rate not measurable; option C stands and this is re-checked", flush=True)
        return
    hours = total_gb * 1000 / rate / 3600
    print(f"\nupload rate {rate:.1f} MB/s -> {total_gb:.0f} GB in {hours:.1f} h of transfer",
          flush=True)
    # the sweep itself is days, and uploads run behind the recording, so the question is
    # whether transfer outlasts the recording rather than whether it is slow in itself
    if hours > 72:
        print("  BINDING: option C's 203 GB of extra upload does not fit behind the "
              "recording. Amendment A21's fallback is option B (499 GB), scope key on the "
              "statement-initial constructs, two-use sets and controls, with the regret "
              "logged: register C3's role-or-context question becomes unanswerable for 160 "
              "constructs without a re-record.", flush=True)
    else:
        print("  not binding: option C stands as amended.", flush=True)


def record(reference: str, batch: int, keep_per_prompt: bool, extra: list[str],
           repo: str | None = None, flush_every: int = 0,
           min_free_gb: float = 20.0) -> int:
    # `sys.executable`, not "python": the recorder must run in the same interpreter as the
    # orchestrator, or it can silently pick up a different environment -- and on a machine
    # where `python` is not on PATH at all it simply fails.
    cmd = [sys.executable, "-m", "experiments.run_sweep", "--reference", reference,
           "--batch", str(batch)] + extra
    if not keep_per_prompt:
        cmd.append("--no-per-prompt")
    # **The recorder uploads as it goes.** A reference is 118 to 121 GB against a Colab disk of
    # 110 to 166 GB, and uploading only at the end both risks the disk and loses the whole
    # reference to a reclaimed runtime. With this on, the upload below finds almost nothing
    # left to send -- it stays as the belt to the recorder's braces.
    if repo and flush_every > 0:
        cmd += ["--repo", repo, "--flush-every", str(flush_every),
                "--min-free-gb", str(min_free_gb)]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return _call(cmd, hours=RECORD_TIMEOUT_H, what=f"the recorder for {reference}")


#: A hung subprocess is the expensive failure, not a crashing one: `subprocess.call` waits for
#: ever, so a stalled upload or a wedged CUDA call would hold an idle A100 at 6.85 units an hour
#: until someone noticed. A reference is ~2.1 h of recording and ~0.4 h of loss probe, so these
#: are generous multiples -- they exist to catch a hang, not a slow run. Killing a hung recorder
#: costs at most the 50 sets since its last flush.
RECORD_TIMEOUT_H = 5.0
LOSS_TIMEOUT_H = 2.0


def _call(cmd: list[str], hours: float, what: str) -> int:
    """Run a subprocess with a ceiling, so a hang cannot quietly spend the balance."""
    try:
        return subprocess.run(cmd, timeout=hours * 3600).returncode
    except subprocess.TimeoutExpired:
        print(f"\n** {what} exceeded {hours:g} h and was killed. That is a hang, not a slow "
              f"run: the record is safe upstream and a rerun resumes from the last flush. "
              f"Aborting so the runtime is released rather than left idle. **", flush=True)
        return 124


def loss_done(reference: str, repo: str | None) -> bool:
    """Whether this reference already has its losses, locally or in the dataset."""
    if (REC / reference / "LOSS.json").exists():
        return True
    if not repo:
        return False
    try:
        from huggingface_hub import HfApi
        return f"record/{reference}/LOSS.json" in set(
            HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo, repo_type="dataset"))
    except Exception:                           # noqa: BLE001 -- re-probing costs 22 min, not a record
        return False


def loss_pass(reference: str, batch: int, limit: int = 0) -> int:
    """C2's losses for one reference, into the reference's own directory.

    **Run per reference, immediately after its activations, and into `REC/<ref>/LOSS.json`** so
    that a reference is atomic: its activations, its losses and its manifest land together and
    "reference 24 done" means one thing. The sweep will certainly be interrupted -- Colab only
    allows a top-up below 500 units, which falls around reference 25 -- and three artefacts that
    can each be half-present is the state a resume cannot reason about.

    A second model load rather than a shared one, deliberately. The recorder uses `AutoModel`
    and the probe needs `AutoModelForCausalLM`: making the activation pass carry the head would
    compute logits over a 96,640-wide vocabulary on every batch, which costs far more than the
    ~30 s load it would save. 59 extra loads is about 4.5 compute units, and it keeps the
    recording path untouched.

    Cowork's item 105 decision: **all 188 constructs**, ~22 min a reference. The pilot's 30 hold
    only 13 in P1, so a loss curve on them cannot give C2 "the share of P2 that is flat".
    """
    out = REC / reference / "LOSS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "experiments.loss_probe", "--reference", reference,
           "--batch", str(batch), "--out", str(out)]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return _call(cmd, hours=LOSS_TIMEOUT_H, what=f"the loss probe for {reference}")


def main():

    from profilebattery.record.store import refuse_retired_record_dir

    refuse_retired_record_dir(REC)   # the recorder may create its tree, but not at the retired path (A50)

    # Hub progress bars off before anything touches the Hub: three bars a file,
    # redrawn every second, bury the lines a human actually reads a sweep by. The
    # environment variable this sets is inherited by the subprocesses too.
    quiet_hub()
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=("licence", "sweep"), required=True)
    # `$ATLAS_DATASET` -- the HuggingFace **dataset**, which carries the same name as
    # the GitHub **repository** by Piotr's choice, so the programme reads the same on both
    # platforms. Provisional on both: the programme's name is undecided and both platforms
    # keep a redirect after a rename, so the two are renamed together at publication.
    #
    # The one consequence to carry: a provenance line must say which platform it means, since
    # the name alone no longer distinguishes 702 GB of activations from 8,000 lines of code.
    ap.add_argument("--repo", default=dataset_repo(),
                    help="HuggingFace DATASET repo id (same name as the GitHub repository)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--flush-every", type=int, default=50,
                    help="the recorder uploads and frees every N sets (0 = off). 50 sets is "
                         "about 2 GB under A21's option C and about fifteen minutes of "
                         "recording, so a reclaimed runtime costs fifty sets rather than a "
                         "reference.")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--shard", default=None, metavar="J/K",
                    help="record only every Kth reference starting at the Jth, 1-based: run "
                         "`--shard 1/2` in one session and `--shard 2/2` in another and they "
                         "cover the plan between them without overlapping. Compute units burn "
                         "per GPU-hour, so this halves the wall clock and saves no units. "
                         "**Stride, not blocks**: each shard is spread across the dense block, "
                         "the tail and the annealing branch, so a session that dies leaves the "
                         "record thin everywhere rather than absent over a whole branch.")
    ap.add_argument("--units-start", type=float, default=0.0, metavar="N",
                    help="compute units in hand at the start. With --units-rate this prints an "
                         "estimated balance before every reference, so the top-up is seen "
                         "coming rather than discovered. Colab exposes no balance to query, so "
                         "this is wall clock x rate and is an estimate, labelled as one.")
    ap.add_argument("--units-rate", type=float, default=6.85, metavar="PER_HOUR",
                    help="compute units an hour for the attached GPU. 6.85 is the A100, "
                         "measured on Piotr's account on 4 Oct.")
    ap.add_argument("--units-warn", type=float, default=520.0, metavar="N",
                    help="warn below this. The default is a little above Colab's 500-unit "
                         "top-up threshold, so the warning arrives before the purchase can.")
    ap.add_argument("--no-loss", dest="loss", action="store_false",
                    help="skip C2's loss pass. It is ~22 min a reference against 2.1 h of "
                         "recording, and Cowork's item 105 decision needs it at every "
                         "reference, so skipping it means a second pass over the month later.")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="exercise the whole orchestrator on N sets per reference, on any "
                         "device, without uploading. For checking the plan, the resume, the "
                         "scope schedule, the watched numbers and the progress file before "
                         "committing a month of A100 to them.")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the storage decision, record nothing")
    args, extra = ap.parse_known_args()

    refs = plan_references(args.plan)
    shard_tag = ""
    if args.shard:
        try:
            j, k = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard {args.shard!r}: expected J/K, as in 1/2")
        if not 1 <= j <= k:
            raise SystemExit(f"--shard {args.shard!r}: need 1 <= J <= K")
        whole, refs = refs, refs[j - 1::k]
        shard_tag = f"_shard{j}of{k}"
        print(f"shard {j} of {k}: {len(refs)} of {len(whole)} references", flush=True)
    sub = subset_constructs()
    print(f"plan {args.plan}: {len(refs)} reference(s)")
    print(f"per-prompt kept at: {[r for r in refs if per_prompt_for(r)] or 'none in this plan'}")
    # Printed here, **applied in `run_sweep`** (A21 decisions 2 and 3). It was printed and
    # nowhere applied until 3 Oct, so every reference that kept per-prompt data kept it for
    # all 188 constructs and for the hidden stream only.
    print(f"per-prompt subset: {len(sub) if sub else 'ALL'} constructs, "
          f"hidden stream (A21 2); residual stream: all of P1 (A21 3)")
    if args.dry_run:
        for i, r in enumerate(refs, 1):
            print(f"  {i:3d}. {r:28} per-prompt {per_prompt_for(r)}")
        return

    check_device(smoke=args.smoke)
    if args.smoke:
        args.no_upload = True
        extra = [a for a in extra if not a.startswith("--limit-sets")]
        extra += ["--limit-sets", str(args.smoke)]
    token = os.environ.get("HF_TOKEN")
    rates = []
    t_run = time.time()
    for i, ref in enumerate(refs, 1):
        print(f"\n=== [{i}/{len(refs)}] {ref}", flush=True)
        if args.units_start:
            # **An estimate, and labelled one.** Colab exposes no balance to query, so this is
            # wall clock times the rate -- summed across every session of this plan, because
            # with `--shard` the balance falls K times faster than one session's clock shows.
            used = units_used(args.plan, args.units_rate,
                              (time.time() - t_run) / 3600, shard_tag)
            left = args.units_start - used
            print(f"  units: ~{used:.0f} used, ~{left:.0f} left of {args.units_start:.0f} "
                  f"(estimated at {args.units_rate}/h)", flush=True)
            if left < args.units_warn:
                print(f"  ** below {args.units_warn:.0f} units: top up now. The sweep resumes "
                      f"from the last flush, so stopping here costs at most 50 sets. **",
                      flush=True)
        # **`fresh` gates everything downstream.** Three functions in a row assumed the
        # reference's local directory exists -- `watched`, `upload`, and the manifest read in
        # the free block -- and all three are wrong for a reference that is already upstream
        # with its local copy freed. Fixing them one at a time produced three aborted runs on
        # 4 Oct; the structural fix is that a reference nothing was recorded for has nothing to
        # upload and nothing to free.
        fresh = not already_recorded(ref, args.repo)
        if not fresh:
            print("  already recorded against this battery, skipping", flush=True)
        else:
            rc = record(ref, args.batch, per_prompt_for(ref), extra, args.repo,
                        args.flush_every, args.min_free_gb)
            if rc != 0:
                raise SystemExit(f"{ref}: recorder exited {rc}")

        # **The loss pass is judged on its own artefact, not on whether the activations were
        # recorded in this run.** References 1 and 2 of the licence pass are already recorded
        # and are skipped above, so a loss pass inside that branch would leave them without
        # losses -- and Cowork's item 105 decision is losses at *every* reference. Keyed on
        # `LOSS.json` instead, which makes it idempotent and fills those two in.
        if args.loss and not loss_done(ref, args.repo):
            rc = loss_pass(ref, args.batch, limit=args.smoke * 20 if args.smoke else 0)
            if rc != 0:
                raise SystemExit(f"{ref}: loss probe exited {rc}")
        elif args.loss:
            print("  losses already present, skipping the probe", flush=True)

        w = watched(ref, args.repo)
        print(f"  watched: {w.get('values') if w.get('values') is not None else w['why']}",
              flush=True)
        if not w["ok"]:
            raise SystemExit(f"{ref}: watched number out of range -- {w['why']}. Aborting at "
                             f"reference {i} rather than at fifty-nine.")
        # **Not gated on `fresh`.** It was, to stop `upload_folder` raising on the missing
        # directory of an already-recorded reference -- but `upload` now no-ops on that, and
        # the gate meanwhile discarded the one artefact such a reference does produce: the
        # `LOSS.json` the probe had just spent twenty minutes of A100 computing. Caught on
        # 5 Oct with two shards mid-pass. What decides whether anything is freed is whether
        # anything was sent, not whether this run did the recording.
        if args.repo and not args.no_upload:
            u = upload(ref, args.repo, token)
            rates.append(u)
            if u.get("MB_per_s"):
                print(f"  uploaded {u['bytes']/1e9:.1f} GB in {u['seconds']}s "
                      f"({u['MB_per_s']} MB/s)", flush=True)
            else:
                print(f"  {u.get('note', 'uploaded')}; {u['bytes']/1e6:.1f} MB left to send",
                      flush=True)
            if len(rates) == 1 and u.get("MB_per_s"):
                report_upload_rate(u)
            # **Upload, verify, then free the local copy.** Colab's local disk is ~110 GB and
            # the sweep writes ~702 GB, so a reference cannot be kept after it is safely
            # upstream. Deleting only on a confirmed upload is the point: deleting on an
            # unconfirmed one is how a record is lost.
            if u.get("bytes"):
                # **the manifest stays.** It is a few kilobytes and it is what the resume
                # check reads; deleting it with the activations is what made a safely
                # uploaded reference look unrecorded.
                man = REC / ref / "MANIFEST.json"
                kept = man.read_bytes() if man.exists() else None
                import shutil
                shutil.rmtree(REC / ref, ignore_errors=True)
                if kept is not None:
                    (REC / ref).mkdir(parents=True, exist_ok=True)
                    man.write_bytes(kept)
                print(f"  freed the local copy of {ref}, kept its manifest", flush=True)
            else:
                print("  nothing local to upload or free", flush=True)
        write_progress(args.plan, refs, i, ref, w.get("values"), rates, shard_tag)
    REC.mkdir(parents=True, exist_ok=True)
    out = REC / f"UPLOAD_RATES_{args.plan}.json"
    out.write_text(json.dumps(rates, indent=1))
    print(f"\ndone. wrote {out}")


def units_used(plan: str, rate: float, mine_hours: float, shard_tag: str) -> float:
    """Units burnt by **every session of this plan**, not just this one.

    With `--shard` the balance falls K times faster than any one session's clock shows, and a
    session that trusted its own figure could run the account to nothing without warning. Each
    session records its own elapsed hours on the shared drive and every session sums them all.

    Best effort: a Drive hiccup returns this session's own figure rather than raising. Over-
    reporting a balance is the dangerous direction, so a failure is said out loud.
    """
    RUNDIR.mkdir(parents=True, exist_ok=True)
    mine = RUNDIR / f"UNITS_{plan}{shard_tag}.json"
    try:
        mine.write_text(json.dumps({"hours": round(mine_hours, 4), "rate": rate,
                                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        total = 0.0
        for f in sorted(RUNDIR.glob(f"UNITS_{plan}*.json")):
            d = json.loads(f.read_text())
            total += float(d.get("hours", 0.0)) * float(d.get("rate", rate))
        return total
    except Exception as e:                      # noqa: BLE001 -- a warning, never a stopped run
        print(f"  (units ledger unreadable: {type(e).__name__}; this session only)", flush=True)
        return mine_hours * rate


def write_progress(plan, refs, done, current, watched_values, rates, shard_tag=""):
    """Append the run's state to a file on the data volume after every reference.

    **Why this and not Weights & Biases.** The thing worth watching across a 14-hour run is the
    three numbers that abort it, and an aborted run has already told you. What is actually
    missing is *visibility after a disconnect*: Colab drops the log, and the question becomes
    "where did it get to". This answers that from the shared drive, with no account, no
    dependency and -- the part that matters -- **no new way for the run to die**. An inline
    reporter that raises on a network blip would take a 14-hour pass with it, and D5's own rule
    is that nothing has to be computed inline for a released record.
    """
    p = RUNDIR / f"PROGRESS_{plan}{shard_tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "plan": plan, "total": len(refs), "done": done, "current": current,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "watched": watched_values, "uploads": rates,
    }, indent=1))


if __name__ == "__main__":
    main()
