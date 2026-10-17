import json

import pandas as pd
import pytest

from syncytium.report import bf16_perplexity, load_results, summarize, to_markdown


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def row(kind, perplexity, indices=(0,), **settings):
    base = {"bits": 4, "group_size": 64, "num_children": 2, "phase": 0.0}
    return {
        "id": f"{kind}/{indices}",
        "kind": kind,
        "indices": list(indices),
        "perplexity": perplexity,
        "bits_per_weight": 4.5,
        **(base | settings),
    }


@pytest.fixture
def handmade(tmp_path):
    rows = [
        {"id": "bf16", "kind": "bf16", "indices": [], "perplexity": 10.0, "bits_per_weight": 16},
        row("child", 12.0, (0,)),
        row("child", 14.0, (1,)),
        row("rebuild", 10.3, (0, 1)),
        row("child", 30.0, (0,), num_children=4),
        row("rebuild", 11.0, (0, 1, 2, 3), num_children=4),
        row("partial", 31.0, (0, 1, 2), num_children=4),
    ]
    return load_results(write_rows(tmp_path / "results.jsonl", rows))


def test_rows_without_rotation_seed_count_as_unrotated(handmade):
    assert set(handmade["rotation"]) == {"off"}


def test_summary_has_one_row_per_setting(handmade):
    assert list(summarize(handmade)["num_children"]) == [2, 4]


def test_summary_statistics(handmade):
    first = summarize(handmade).iloc[0]
    assert (first["child_best"], first["child_mean"], first["child_worst"]) == (12, 13, 14)
    assert first["rebuild_gap"] == pytest.approx(0.3)


def test_hypothesis_checks(handmade):
    summary = summarize(handmade).set_index("num_children")
    assert summary.loc[2, "h3"]
    assert not summary.loc[4, "h3"]
    assert not summary.loc[4, "h4"]  # a partial rebuild worse than the best child
    assert pd.isna(summary.loc[2, "h4"])  # no partial rebuilds were logged for N = 2


def test_markdown_states_the_bf16_reference(handmade):
    text = to_markdown(summarize(handmade), bf16_perplexity(handmade))
    assert text.startswith("bf16 perplexity: **10.000**")
    assert "| rebuild |" not in text and "rebuild_gap" in text


def test_real_sweep_log_summarizes(results_dir):
    results = load_results(results_dir / "results.jsonl")
    summary = summarize(results)
    assert set(summary["rotation"]) == {"off", "on"}
    assert summary["rebuild"].notna().all()
