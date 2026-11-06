"""The position-keyed loss pass and the paired role gap — S16 rebuilt."""
import numpy as np
import pytest

import experiments.position_loss as pl


def test_the_gap_is_read_only_at_dependent_classes():
    """Cowork's call 1, 10 Oct. A control's header is a plain assignment, so A's obliged
    positions have no counterpart in it and the gap there does not exist by construction.
    """
    assert set(pl.DEPENDENT) == {"governed", "scope"}
    assert "obliged" not in pl.DEPENDENT


def _row(pid, core, positions, control_of=None):
    return {"core": core, "control_of": control_of, "positions": positions}


def _pos(cls, role, tok_i, ctrl_i, loss, matched=True):
    return {"class": cls, "role": role, "token_index": tok_i, "control_index": ctrl_i,
            "matched": matched, "forced": False, "floor_nats": None, "bound_name": "x",
            "piece": "p", "loss": loss, "why": None}


def test_a_control_position_pairs_by_control_index_not_by_order():
    """The positions sit at different indices because the replaced header is a different length.

    `For`'s governed uses are at 30 and 43 in the A prompt and 29 and 42 in the control, and a
    control record's `control_index` names its counterpart's index. Sharing an index, or
    pairing by order, would read the wrong token and return a plausible number.
    """
    a = {"A1": _row("A1", "c0", [_pos("governed", "governed_use", 30, 29, 1.0),
                                 _pos("governed", "governed_use", 43, 42, 3.0)])}
    c = {"C1": _row("C1", "c0", [_pos("governed", "governed_use", 29, 30, 1.5),
                                 _pos("governed", "governed_use", 42, 43, 3.25)],
                    control_of="A1")}
    g = pl.role_gap(a, c)
    st = g["by_class_role"]["governed|governed_use"]
    # 1.5-1.0 = 0.5 and 3.25-3.0 = 0.25 -> core mean 0.375
    assert st["mean"] == pytest.approx(0.375)
    assert st["n"] == 2 and st["cores"] == 1
    assert g["skipped"] == {}


def test_an_unmatched_or_unresolvable_position_is_counted_not_dropped():
    """A silent drop makes a thin reading look like a complete one."""
    a = {"A1": _row("A1", "c0", [_pos("governed", "governed_use", 30, 29, 1.0)])}
    c = {"C1": _row("C1", "c0", [
             _pos("governed", "governed_use", 29, 30, 1.5),
             _pos("governed", "governed_use", 31, 99, 2.0),            # no A at 99
             _pos("scope", "scope_use", 33, 29, 2.0, matched=False),   # not matched
         ], control_of="A1"),
         "C2": _row("C2", "c0", [_pos("governed", "governed_use", 29, 30, 9.0)],
                    control_of="MISSING")}
    g = pl.role_gap(a, c)
    assert g["skipped"]["no A position at control_index"] == 1
    assert g["skipped"]["position not matched"] == 1
    assert g["skipped"]["no paired A prompt"] == 1
    assert g["by_class_role"]["governed|governed_use"]["n"] == 1


def test_the_obliged_positions_never_enter_the_gap():
    a = {"A1": _row("A1", "c0", [_pos("obliged", "in", 20, 19, 1.0)])}
    c = {"C1": _row("C1", "c0", [_pos("obliged", "in", 19, 20, 5.0)], control_of="A1")}
    assert pl.role_gap(a, c)["by_class_role"] == {}


def test_core_stats_is_between_core_and_keeps_the_core_means():
    """The SE every reading in the plan is in: spread of core means over sqrt(cores)."""
    st = pl.core_stats({"c0": [1.0, 1.0], "c1": [3.0], "c2": []})
    assert st["cores"] == 2 and st["n"] == 3
    assert st["mean"] == pytest.approx(2.0)
    assert st["se"] == pytest.approx(np.std([1.0, 3.0], ddof=1) / np.sqrt(2))
    assert set(st["core_means"]) == {"c0", "c1"}


def test_core_stats_of_nothing_is_none_not_zero():
    """Zero is a measurement; nothing read is not."""
    st = pl.core_stats({})
    assert st["mean"] is None and st["n"] == 0 and st["se"] is None


def test_aggregate_counts_what_it_could_not_read():
    detail = {"P1": _row("P1", "c0", [
        _pos("governed", "governed_use", 30, 29, 1.0),
        {"class": "obliged", "role": "in", "token_index": 99, "control_index": None,
         "matched": True, "forced": True, "floor_nats": 0.5, "bound_name": None,
         "piece": None, "loss": None, "why": "token_index 99 outside 1..45"}])}
    agg = pl.aggregate(detail)
    assert "governed|governed_use" in agg["by_class_role"]
    assert "obliged|in" not in agg["by_class_role"], "an unread position is not a reading"
    assert agg["unread"] == {"token_index 99 outside 1..45": 1}
    assert "governed|governed_use|p" in agg["by_class_role_piece"]


def test_the_piece_is_carried_so_the_expected_arm_needs_no_offset_join():
    """A27's expected arm reads each realised piece's curve.

    Without the piece here, `POSITIONS.json` could not replace `by_offset_and_piece` and the
    offset join would survive for part of the primary reading — which is what the wider pass
    is bought to retire.
    """
    import inspect
    src = inspect.getsource(pl._forward)
    assert "convert_ids_to_tokens" in src and '"piece"' in src
    agg = pl.aggregate({"P": _row("P", "c0", [_pos("obliged", "in", 20, 19, 1.0)])})
    assert list(agg["by_class_role_piece"]) == ["obliged|in|p"]


def test_the_forward_runs_under_no_grad():
    """`loss_probe.losses_for` carries `@torch.no_grad()`; I copied the body without it.

    Without it the logits keep a graph, `ce.cpu().numpy()` raises "Can't call numpy() on Tensor
    that requires grad" — which is how cell 2 died on the A100 — and before raising it holds a
    96,640-wide vocabulary's activations per batch. Third time today something just outside a
    function failed to come across with it: `-m` instead of a script path, the xet retry, this.
    """
    import inspect
    src = inspect.getsource(pl.losses_at_positions)
    assert "no_grad" in src, "the forward must not build a graph"
    # and the inner loop must be reached only through that wrapper
    assert "_forward(" in src
    inner = inspect.getsource(pl._forward)
    assert "model(" in inner and "no_grad" not in inner


def _battery(tmp_path, sets):
    """sets: {name: [rows]} where a row is a dict; written as the battery lays them out."""
    import gzip
    import json
    d = tmp_path / "twin1"
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        with gzip.open(d / f"{name}.jsonl.gz", "wt") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return tmp_path


def test_only_sets_that_name_positions_are_read(tmp_path):
    """Only 27 of 188 A sets carry positions; the rest are 74% of the tokens and yield nothing.

    The first timed reference showed it as a column of `0 class/role`, and `verify` would have
    refused the run — correctly. The set list was wrong, not the guard.
    """
    b = _battery(tmp_path, {
        "For_A": [{"positions": [{"class": "obliged", "role": "in", "token_index": 5}]}],
        "abs_A": [{"positions": []}, {"positions": []}],
        "zip_A": [{}],                                    # no `positions` key at all
    })
    inc, exc = pl.sets_with_positions(b)
    assert inc == ["For"] and exc == ["abs", "zip"]


def test_a_set_whose_later_rows_have_positions_is_still_included(tmp_path):
    """The probe reads the first rows, so it must not miss a set that starts empty."""
    rows = [{"positions": []}] * 5 + [{"positions": [{"class": "obliged", "role": "x",
                                                      "token_index": 3}]}]
    b = _battery(tmp_path, {"Odd_A": rows})
    assert pl.sets_with_positions(b, probe_rows=20)[0] == ["Odd"]
    # and with too small a probe it would be missed, which is why the window is 20 not 1
    assert pl.sets_with_positions(b, probe_rows=2)[0] == []


def test_the_probe_agrees_with_an_exhaustive_scan_of_the_real_battery():
    """Probing rather than scanning is only safe if the distinction is structural."""
    import gzip
    import json
    import pathlib
    b = pl.battery_dir()
    if not (b / "twin1").exists():
        pytest.skip("the phase battery is not on this machine")
    inc, exc = pl.sets_with_positions(b)
    full = []
    for f in sorted((b / "twin1").glob("*_A.jsonl.gz")):
        if any((json.loads(l).get("positions") or []) for l in gzip.open(f, "rt")):
            full.append(f.name[:-len("_A.jsonl.gz")])
    assert inc == full, set(inc) ^ set(full)
    assert len(inc) == 27 and len(exc) == 161


def test_the_skipped_sets_are_recorded_in_the_output():
    """An absent set must never look like a forgotten one."""
    import pathlib
    src = pathlib.Path(pl.__file__).read_text()
    assert '"skipped_no_positions": skipped' in src
