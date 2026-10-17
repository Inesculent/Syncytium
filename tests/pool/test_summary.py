import math

import pandas as pd
import pytest

from syncytium.pool import (
    Kind,
    additivity,
    load_pool_results,
    plan,
    run_states,
    to_markdown,
    write_pool_report,
)
from syncytium.pool.runner import write_layout
from syncytium.pool.summary import per_estimator, r_squared
from syncytium.sweep import ResultLog


@pytest.fixture
def results_dir(tiny_model, batches, grid, tmp_path):
    run_states(plan(grid), tiny_model, batches, ResultLog(tmp_path / "results.jsonl"), grid.stages)
    write_layout(tmp_path / "layout.json", tiny_model, grid)
    return tmp_path


def test_additivity_has_one_row_per_random_pattern(results_dir, grid):
    results = load_pool_results(results_dir / "results.jsonl")
    fits = additivity(results)
    assert len(fits) == sum(c.kind == Kind.RANDOM for c in plan(grid))
    assert fits[["measured", "predicted"]].notna().all().all()


def test_rows_valid_for_any_estimator_join_every_group():
    frame = pd.DataFrame({"estimator": ["mean", "any", "midpoint"], "value": [1, 2, 3]})
    expanded = per_estimator(frame, ["mean", "midpoint"])
    assert sorted(expanded.loc[expanded["estimator"] == "mean", "value"]) == [1, 2]
    assert sorted(expanded.loc[expanded["estimator"] == "midpoint", "value"]) == [2, 3]
    assert "any" not in set(expanded["estimator"])
    assert per_estimator(frame, []).equals(frame)


def test_r_squared():
    measured = pd.Series([0.1, 0.2, 0.4])
    assert r_squared(measured, measured) == pytest.approx(1.0)
    assert r_squared(measured, pd.Series([0.2, 0.3, 0.5])) < 1
    assert math.isnan(r_squared(measured[:1], measured[:1]))


def test_markdown_reports_the_tables_and_checks(results_dir):
    text = to_markdown(load_pool_results(results_dir / "results.jsonl"))
    assert text.startswith("bf16 perplexity: **")
    for heading in ("One device lost", "One stage on a single child", "Additivity", "H6b", "H6c"):
        assert heading in text
    assert "(any)" not in text


def test_report_writes_summary_and_fits(results_dir, tmp_path):
    out = tmp_path / "report"
    write_pool_report(results_dir, out)
    assert "largest device" in (out / "summary.md").read_text(encoding="utf-8")
    assert (out / "additivity.csv").exists()
