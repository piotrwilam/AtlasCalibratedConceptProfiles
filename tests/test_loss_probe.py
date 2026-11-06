"""The loss probe's set selection and its merge guard — S16."""
import json
import pathlib

import pytest

import experiments.loss_probe as lp


def battery(tmp_path, names):
    d = tmp_path / "twin1"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / f"{n}.jsonl.gz").write_bytes(b"")
    return tmp_path


def test_the_default_kinds_are_a_and_b_only():
    """The pass that produced the 61 LOSS.json files took these, which is why S16 exists."""
    import inspect
    sig = inspect.signature(lp.set_files)
    assert sig.parameters["kinds"].default is inspect.Parameter.empty  # explicit at the call
    src = pathlib.Path(lp.__file__).read_text()
    assert '"--kinds", default="A,B"' in src


def test_ctl_is_selectable_and_a_b_are_not_dragged_in(tmp_path):
    """S16 runs the 11 controls alone: the A and B sets are already measured and cost 22 min each."""
    b = battery(tmp_path, ["For_A", "For_B", "For_CTL", "With_A", "With_CTL", "If_A"])
    got = [f.name for f in lp.set_files(b, "CTL")]
    assert got == ["For_CTL.jsonl.gz", "With_CTL.jsonl.gz"], got

    both = [f.name for f in lp.set_files(b, "A,B")]
    assert all("_CTL" not in n for n in both) and len(both) == 4


def test_a_kind_with_no_sets_fails_loudly(tmp_path):
    """A typo must not read as "nothing to do, success"."""
    b = battery(tmp_path, ["For_A"])
    with pytest.raises(SystemExit, match="nothing to do"):
        lp.set_files(b, "CTRL")          # the kind is `CTL`


def test_constructs_still_narrows_within_a_kind(tmp_path):
    b = battery(tmp_path, ["For_CTL", "With_CTL", "Import_CTL"])
    got = [f.name for f in lp.set_files(b, "CTL", "For,Import")]
    assert got == ["For_CTL.jsonl.gz", "Import_CTL.jsonl.gz"], got


def test_carry_over_keeps_the_sets_a_ctl_pass_does_not_recompute(tmp_path):
    """S16 must add to `LOSS.json`, not replace it."""
    out = tmp_path / "LOSS.json"
    out.write_text(json.dumps({"limit": None, "sets": {"For|A": {}, "For|B": {}}}))
    assert set(lp.carry_over(out, None, resume=True)) == {"For|A", "For|B"}


def test_carry_over_refuses_when_the_limit_differs(tmp_path):
    """The file is rewritten whole, so an unmatched limit would silently drop 376 sets.

    Those sets cost about 22 minutes of A100 each. Nothing in the original code compared the
    two set lists: the write was unconditional.
    """
    out = tmp_path / "LOSS.json"
    out.write_text(json.dumps({"limit": None, "sets": {"For|A": {}}}))
    with pytest.raises(SystemExit, match="would not carry them over"):
        lp.carry_over(out, 40, resume=True)


def test_carry_over_refuses_no_resume_over_existing_sets(tmp_path):
    out = tmp_path / "LOSS.json"
    out.write_text(json.dumps({"limit": None, "sets": {"For|A": {}}}))
    with pytest.raises(SystemExit, match="--no-resume was given"):
        lp.carry_over(out, None, resume=False)


def test_carry_over_on_an_absent_or_empty_file_is_empty(tmp_path):
    assert lp.carry_over(tmp_path / "gone.json", None, resume=True) == {}
    out = tmp_path / "LOSS.json"
    out.write_text(json.dumps({"limit": 40, "sets": {}}))
    assert lp.carry_over(out, None, resume=True) == {}   # no sets to lose, so no refusal


def test_n_sets_describes_the_file_after_the_merge():
    """A `--kinds CTL` pass over a 376-set file must not record `n_sets: 11`.

    That would make a complete file look truncated to every later reader, including the audit
    that checks the record is whole.
    """
    src = pathlib.Path(lp.__file__).read_text()
    assert 'res["n_sets"] = len(res["sets"])' in src
