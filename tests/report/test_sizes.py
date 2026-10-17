import json
import shutil

import pandas as pd
import pytest
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

from syncytium.eval import ParameterCounts, count_parameters
from syncytium.models import quantized_weights
from syncytium.report.cli import write_comparison
from syncytium.report.sizes import (
    Sweep,
    check_counts,
    equal_memory,
    frontier,
    load_sweep,
    single_models,
    to_markdown,
)


def write_runs(directory, *models):
    lines = [
        json.dumps({"config": f'name = "x"\noutput = "x"\n[model]\npath = "{m.as_posix()}"\n'})
        for m in models
    ]
    (directory / "runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def model_dir(tmp_path, tiny_config):
    directory = tmp_path / "tiny-model"
    tiny_config.save_pretrained(directory)
    return directory


@pytest.fixture
def sweep_dir(tmp_path, results_dir, model_dir):
    """The tiny sweep log next to a runs.jsonl naming the tiny model's config."""
    directory = tmp_path / "sweep"
    directory.mkdir()
    shutil.copy(results_dir / "results.jsonl", directory)
    write_runs(directory, model_dir)
    return directory


def test_counts_come_from_the_model_config(sweep_dir, tiny_config):
    model = Qwen3_5ForCausalLM(tiny_config)
    assert load_sweep(sweep_dir).counts == count_parameters(model, quantized_weights(model))


def test_wrong_counts_are_refused(sweep_dir):
    sweep = load_sweep(sweep_dir)
    wrong = Sweep(sweep.model, sweep.results, ParameterCounts(sweep.counts.quantized, 0))
    with pytest.raises(ValueError, match="bits per weight"):
        check_counts(wrong)


def test_runs_of_different_models_are_refused(sweep_dir, model_dir, tmp_path):
    write_runs(sweep_dir, model_dir, tmp_path / "other")
    with pytest.raises(ValueError, match="2 models"):
        load_sweep(sweep_dir)


@pytest.mark.parametrize(
    ("model", "label"),
    [
        ("Qwen/Qwen3.5-0.8B-Base", "0.8B"),
        ("Qwen/Qwen3.5-9B-Base", "9B"),
        ("local/tiny", "tiny"),
        (r"C:\models\tiny", "tiny"),
    ],
)
def test_label_is_the_model_size(model, label):
    assert Sweep(model, pd.DataFrame(), ParameterCounts(0, 0)).label == label


def handmade_sweep(tmp_path, model, quantized, perplexities):
    """A sweep with bf16, round-to-nearest at 4 and 8 bits, and one 4-bit setting with N = 4."""
    rows = [{"id": "bf16", "kind": "bf16", "indices": [], "perplexity": perplexities["bf16"]}]
    for bits in (4, 8):
        rows.append(
            {
                "id": f"rtn{bits}",
                "kind": "rtn",
                "bits": bits,
                "group_size": 64,
                "num_children": 1,
                "phase": 0.5,
                "indices": [0],
                "perplexity": perplexities[f"rtn{bits}"],
            }
        )
    setting = {"bits": 4, "group_size": 64, "num_children": 4, "phase": 0.5}
    for kind, indices, value in [
        ("child", [0], perplexities["child"]),
        ("partial", [0, 1, 2], perplexities["partial"]),
        ("rebuild", [0, 1, 2, 3], perplexities["rebuild"]),
    ]:
        rows.append(
            {
                "id": f"{kind}{indices}",
                "kind": kind,
                "indices": indices,
                "perplexity": value,
                **setting,
            }
        )
    for row in rows:
        row["bits_per_weight"] = 0.0  # not used by the comparison
    path = tmp_path / f"{model.replace('/', '-')}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    from syncytium.report import load_results

    return Sweep(model, load_results(path), ParameterCounts(quantized, 0))


@pytest.fixture
def two_sizes(tmp_path):
    # In memory: small 4-bit < small 8-bit < small bf16 < big 4-bit < big 8-bit < big bf16.
    small = handmade_sweep(
        tmp_path,
        "org/m-1B",
        64_000,
        {"bf16": 12, "rtn4": 14, "rtn8": 12.5, "child": 15, "partial": 12.4, "rebuild": 12.2},
    )
    big = handmade_sweep(
        tmp_path,
        "org/m-4B",
        256_000,
        {"bf16": 8, "rtn4": 9, "rtn8": 8.1, "child": 9.5, "partial": 8.3, "rebuild": 8.2},
    )
    return [small, big]


def test_frontier_keeps_models_that_beat_everything_smaller(two_sizes):
    labels = list(frontier(single_models(two_sizes))["label"])
    assert labels == ["1B 4-bit", "1B 8-bit", "1B bf16", "4B 4-bit", "4B 8-bit", "4B bf16"]


def test_best_single_model_is_the_best_that_fits(two_sizes):
    table = equal_memory(two_sizes).set_index("model")
    # A small child fits only the small 4-bit model; a big child fits every small model
    # and big 4-bit, which is the best of them.
    assert table.loc["1B", "best_single"] == "1B 4-bit"
    assert table.loc["4B", "best_single"] == "4B 4-bit"
    assert table.loc["4B", "best_single_perplexity"] == 9
    assert (table.loc["4B", "child_worst"], table.loc["4B", "fleet_minus_one"]) == (9.5, 8.3)
    assert table.loc["4B", "fleet_all"] == 8.2


def test_markdown_lists_the_models(two_sizes):
    text = to_markdown(equal_memory(two_sizes), two_sizes)
    assert "1B (0.00B parameters, bf16 perplexity 12.000)" in text
    assert "best_single" in text


def test_comparison_writes_tables_and_figures(sweep_dir, tmp_path):
    out = tmp_path / "sizes"
    write_comparison([sweep_dir], out)
    assert (out / "sizes.md").exists()
    assert (out / "sizes.csv").exists()
    assert (out / "sizes-p0.5-rot-on.png").exists()
