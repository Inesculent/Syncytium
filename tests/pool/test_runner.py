import json
from dataclasses import replace

from syncytium.eval import perplexity
from syncytium.pool import plan, run_states
from syncytium.pool import runner as runner_module
from syncytium.pool.runner import write_layout
from syncytium.sweep import ResultLog
from syncytium.sweep.runner import calibrate


def count_evaluations(monkeypatch):
    calls = []

    def counting_perplexity(model, batches):
        calls.append(1)
        return perplexity(model, batches)

    monkeypatch.setattr(runner_module, "perplexity", counting_perplexity)
    return calls


def test_logs_every_condition(tiny_model, batches, grid, tmp_path):
    log = ResultLog(tmp_path / "results.jsonl")
    conditions = plan(grid)
    run_states(conditions, tiny_model, batches, log, grid.stages)
    rows = log.completed()
    assert set(rows) == {c.id for c in conditions}
    assert rows["full/lost-none"]["lost"] == []


def test_evaluates_each_distinct_state_once(tiny_model, batches, grid, tmp_path, monkeypatch):
    calls = count_evaluations(monkeypatch)
    conditions = plan(grid)
    run_states(conditions, tiny_model, batches, ResultLog(tmp_path / "r.jsonl"), grid.stages)
    assert len(calls) == len({c.key for c in conditions})


def test_resumes_without_repeating_work(tiny_model, batches, grid, tmp_path, monkeypatch):
    log = ResultLog(tmp_path / "results.jsonl")
    run_states(plan(grid), tiny_model, batches, log, grid.stages)
    calls = count_evaluations(monkeypatch)
    run_states(plan(grid), tiny_model, batches, log, grid.stages)
    assert calls == []


def test_scaled_grid_runs_with_calibration(tiny_model, batches, grid, tmp_path):
    scaled = replace(grid, scaling=True)
    log = ResultLog(tmp_path / "results.jsonl")
    scales = calibrate(tiny_model, batches)
    run_states(plan(scaled), tiny_model, batches, log, scaled.stages, scales)
    rows = log.completed()
    assert all(row["scaling"] for row in rows.values() if row["kind"] != "bf16")


def test_layout_records_device_and_model_memory(tiny_model, grid, tmp_path):
    write_layout(tmp_path / "layout.json", tiny_model, grid)
    layout = json.loads((tmp_path / "layout.json").read_text(encoding="utf-8"))
    assert len(layout["device_gb"]) == grid.stages + 1
    assert max(layout["device_gb"]) < layout["dense_gb"]
