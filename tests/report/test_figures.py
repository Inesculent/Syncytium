import pytest
import torch
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

from syncytium.report import all_figures, load_results, summarize, write_report
from syncytium.report.figures import _mark
from syncytium.sweep import ResultLog, plan, run_conditions
from syncytium.sweep.config import GridConfig
from syncytium.sweep.runner import calibrate


@pytest.fixture(scope="module")
def results(results_dir):
    return load_results(results_dir / "results.jsonl")


def test_one_figure_set_per_phase_and_rotation(results):
    names = set(all_figures(results))
    for suffix in ("p0-rot-off", "p0-rot-on", "p0.5-rot-off", "p0.5-rot-on"):
        assert {f"recovery-{suffix}", f"pareto-{suffix}"} <= names
    assert "offsets" in names


def test_a_run_of_scaled_settings_has_no_offsets_figure(results):
    scaled = results.copy()
    scaled.loc[scaled["kind"] != "bf16", "scaling"] = "on"
    names = set(all_figures(scaled))
    assert "offsets" not in names
    assert any(name.startswith("recovery-") for name in names)


def test_recovery_has_one_panel_per_bit_width(results):
    figure = all_figures(results)["recovery-p0-rot-off"]
    assert [axis.get_title() for axis in figure.axes] == ["3-bit children", "4-bit children"]


def test_every_chart_uses_a_single_log_scale_axis(results):
    for figure in all_figures(results).values():
        for axis in figure.axes:
            assert axis.get_yscale() == "log"


def test_more_than_three_series_is_refused():
    with pytest.raises(ValueError, match="at most 3 series"):
        _mark(3)


def test_report_writes_tables_and_figures(results_dir, tmp_path):
    write_report(results_dir, tmp_path)
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("bf16 perplexity")
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "offsets.svg").exists()
    assert (tmp_path / "pareto-p0-rot-on.png").exists()


@pytest.fixture(scope="module")
def mixed_results(tmp_path_factory, tiny_config):
    """A log with scaled and unscaled settings of the same grid."""
    directory = tmp_path_factory.mktemp("mixed")
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tiny_config).eval()
    batches = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(0))
    grid = GridConfig(
        bits=(4,), group_sizes=(32,), children=(4,), scaling=(False, True), rtn_bits=(4, 6)
    )
    log = ResultLog(directory / "results.jsonl")
    run_conditions(plan(grid), model, batches, log, calibrate(model, batches))
    return load_results(directory / "results.jsonl")


def test_scaled_settings_get_their_own_rows_and_figures(mixed_results):
    assert sorted(summarize(mixed_results)["scaling"]) == ["off", "on"]
    names = set(all_figures(mixed_results))
    assert {"recovery-p0-rot-off", "recovery-p0-rot-off-scaled"} <= names
    assert {"baselines-p0-rot-off", "baselines-p0-rot-off-scaled"} <= names


def test_logs_without_scaling_read_as_unscaled(results):
    assert set(results["scaling"]) == {"off"}
