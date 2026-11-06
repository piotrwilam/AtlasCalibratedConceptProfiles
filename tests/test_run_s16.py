"""S16's driver, rebuilt around the position-keyed pass."""
import json
import pathlib

import pytest

import experiments.run_s16 as rs


def _set(means):
    return {"by_class_role": {k: {"mean": v} for k, v in means.items()}}


def test_verify_refuses_a_set_that_read_nothing():
    """What failed on 10 Oct: the check counted that eleven keys arrived and never looked in.

    S16's first pass wrote 11 `_CTL` entries with empty contents at 54 references and the
    guard passed every one.
    """
    doc = {"sets": {"For|A": _set({"obliged|in": 1.0}), "For|CTL": {"by_class_role": {}}},
           "role_gap": {"For": _set({"governed|governed_use": 0.3})}}
    with pytest.raises(SystemExit, match="read nothing at any position"):
        rs.verify(doc, {"For"})


def test_verify_refuses_a_set_whose_every_mean_is_none():
    doc = {"sets": {"For|A": _set({"obliged|in": None})},
           "role_gap": {"For": _set({"governed|governed_use": 0.3})}}
    with pytest.raises(SystemExit, match="no finite mean"):
        rs.verify(doc, {"For"})


def test_verify_refuses_a_missing_or_empty_role_gap():
    doc = {"sets": {"For|A": _set({"obliged|in": 1.0})}, "role_gap": {}}
    with pytest.raises(SystemExit, match="no role gap for"):
        rs.verify(doc, {"For"})

    doc2 = {"sets": {"For|A": _set({"obliged|in": 1.0})},
            "role_gap": {"For": _set({"governed|governed_use": None})}}
    with pytest.raises(SystemExit, match="role gap is empty"):
        rs.verify(doc2, {"For"})


def test_verify_passes_a_complete_pass():
    doc = {"sets": {"For|A": _set({"obliged|in": 1.0}),
                    "For|CTL": _set({"governed|governed_use": 2.0})},
           "role_gap": {"For": _set({"governed|governed_use": 0.3})}}
    rs.verify(doc, {"For"})


def test_nothing_is_merged_into_loss_json():
    """The hazard is removed structurally, not guarded against.

    `POSITIONS.json` is a new artefact, so there is no fetch-modify-reupload of the 376-set
    offset table that cost about 22 minutes of A100 a reference to produce.
    """
    src = pathlib.Path(rs.__file__).read_text()
    assert "POSITIONS.json" in src
    assert "LOSS.json" not in src.split('"""')[2], "the driver must not touch LOSS.json"
    assert "fetch_loss" not in src


def test_the_model_cache_is_purged_between_references(monkeypatch):
    """54 cached revisions at 3.81 GB filled the disk at reference 55 on 10 Oct."""
    import profilebattery.record.remote as remote
    seen = []
    monkeypatch.setattr(remote, "purge_repo_cache",
                        lambda target: seen.append(target) or (2, "ok"))
    seq = iter([10.0, 48.0])
    monkeypatch.setattr(rs, "free_gb", lambda path="/": next(seq))
    assert rs.purge_model_cache() == pytest.approx(38.0)
    assert seen == [rs.MODEL_TARGET]
    assert rs.MODEL_TARGET.startswith("model/") and rs.MODEL_TARGET != rs.REPO


def test_go_refuses_without_a_timed_reference_on_this_host(monkeypatch, tmp_path):
    """Every inherited figure for this pass has been wrong, so the rate is measured here."""
    monkeypatch.setattr(rs, "TIMED_RECORD", tmp_path / "S16_TIMED.json")
    monkeypatch.setattr(rs, "OUT", tmp_path / "out")
    monkeypatch.setattr(rs, "controls", lambda: {"For"})
    monkeypatch.setattr("sys.argv", ["run_s16.py", "--go"])
    import profilebattery.record.remote as remote
    monkeypatch.setattr(remote, "hub_token", lambda: "t")
    with pytest.raises(SystemExit, match="run --timed first"):
        rs.main()

    (tmp_path / "S16_TIMED.json").write_text(json.dumps({"host_kind": "linux-NVIDIA A100"}))
    with pytest.raises(SystemExit, match="rate is the host's"):
        rs.main()


def test_the_token_count_is_for_the_sets_that_actually_read_something():
    """2,135,922 tokens: the 27 A sets that name positions, plus the 11 controls.

    It was 8,366,284 for all 188 A sets, and the first timed reference on an A100 showed why
    that was wrong — 161 of them printed `0 class/role` because builtins and exception names
    have no obliged, governed or scope position by design. Those 161 were 74% of the tokens and
    produced nothing: 4.4 of every 6.1 minutes a reference. The projection in `--timed` rests
    on this number, so it tracks what the pass now reads.
    """
    assert rs.TOKENS_PER_REFERENCE == 2_135_922


def test_a_battery_without_controls_stops_the_run(monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "controls", lambda: set())
    monkeypatch.setattr(rs, "OUT", tmp_path / "out")
    monkeypatch.setattr(rs, "TIMED_RECORD", tmp_path / "x.json")
    monkeypatch.setattr("sys.argv", ["run_s16.py", "--timed"])
    import profilebattery.record.remote as remote
    monkeypatch.setattr(remote, "hub_token", lambda: "t")
    with pytest.raises(SystemExit, match="role gap cannot be read"):
        rs.main()


def test_the_timed_reference_is_a_real_checkpoint():
    """`references()[-1]` is `untrained_seed2`, which `load_lm` builds from config with a seed.

    No download and no real load, so the projection would omit the 25-40 s every other
    reference spends fetching a 3.81 GB revision — understating a 3.4-minute reference by
    15-20%. The rate would still be right; the total would not.
    """
    assert rs.TIMED_REFERENCE == "pretrain_iter_0484865"
    assert not rs.TIMED_REFERENCE.startswith("untrained")
    src = pathlib.Path(rs.__file__).read_text()
    assert "references(token)[-1]" not in src
