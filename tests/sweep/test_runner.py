import pytest
import torch

from syncytium.eval import perplexity
from syncytium.sweep import ResultLog, plan, run_conditions
from syncytium.sweep import runner as runner_module
from syncytium.sweep.config import GridConfig
from syncytium.sweep.runner import calibrate

# The tiny model's narrowest input dimension is 32, so groups of 32 fit every weight.
GRID = GridConfig(bits=(4,), group_sizes=(32,), children=(2, 4), max_missing=1, rtn_bits=(4,))


@pytest.fixture
def batches():
    return torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(0))


@pytest.fixture
def log(tmp_path):
    return ResultLog(tmp_path / "results.jsonl")


def count_evaluations(monkeypatch):
    calls = []

    def counting_perplexity(model, batches):
        calls.append(1)
        return perplexity(model, batches)

    monkeypatch.setattr(runner_module, "perplexity", counting_perplexity)
    return calls


def test_logs_every_condition(tiny_model, batches, log):
    conditions = plan(GRID)
    run_conditions(conditions, tiny_model, batches, log)
    assert set(log.completed()) == {c.id for c in conditions}


def test_evaluates_each_distinct_model_once(tiny_model, batches, log, monkeypatch):
    calls = count_evaluations(monkeypatch)
    conditions = plan(GRID)
    run_conditions(conditions, tiny_model, batches, log)
    assert len(calls) == len({c.key for c in conditions})


def test_reused_rows_copy_their_source(tiny_model, batches, log):
    run_conditions(plan(GRID), tiny_model, batches, log)
    rows = log.completed()
    reused = [row for row in rows.values() if row["reused_from"]]
    assert reused
    for row in reused:
        assert row["perplexity"] == rows[row["reused_from"]]["perplexity"]


def test_resumes_without_repeating_work(tiny_model, batches, log, monkeypatch):
    run_conditions(plan(GRID), tiny_model, batches, log)
    calls = count_evaluations(monkeypatch)
    run_conditions(plan(GRID), tiny_model, batches, log)
    assert calls == []


def test_bf16_row_matches_the_untouched_model(tiny_model, batches, log):
    expected = perplexity(tiny_model, batches)
    run_conditions(plan(GRID), tiny_model, batches, log)
    assert log.completed()["bf16"]["perplexity"] == pytest.approx(expected)


def test_scaled_conditions_run_with_calibration(tiny_model, batches, log):
    scaled_grid = GridConfig(
        bits=(4,), group_sizes=(32,), children=(2,), scaling=(True,), rtn_bits=(4,)
    )
    run_conditions(plan(scaled_grid), tiny_model, batches, log, calibrate(tiny_model, batches))
    rows = log.completed()
    assert "rtn/b4-g32-n1-p0.5-s/k0" in rows
    assert all(row["scaling"] for row in rows.values() if row["kind"] != "bf16")


def test_scaled_conditions_need_calibration(tiny_model, batches, log):
    scaled_grid = GridConfig(bits=(4,), group_sizes=(32,), children=(2,), scaling=(True,))
    with pytest.raises(ValueError, match="calibration"):
        run_conditions(plan(scaled_grid), tiny_model, batches, log)
