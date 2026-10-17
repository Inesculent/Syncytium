import pytest

from holograph.report import all_figures, load_results, write_report
from holograph.report.figures import _mark


@pytest.fixture(scope="module")
def results(results_dir):
    return load_results(results_dir / "results.jsonl")


def test_one_figure_set_per_phase_and_rotation(results):
    names = set(all_figures(results))
    for suffix in ("p0-rot-off", "p0-rot-on", "p0.5-rot-off", "p0.5-rot-on"):
        assert {f"recovery-{suffix}", f"pareto-{suffix}"} <= names
    assert "offsets" in names


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
