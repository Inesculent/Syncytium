from pathlib import Path

import pytest

from syncytium.sweep import load_config

MINIMAL = """
name = "test"
output = "results/test"

[model]
path = "some/model"

[grid]
bits = [4]
group_sizes = [64]
children = [2, 4]
"""


def write(tmp_path, text):
    path = tmp_path / "sweep.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_values_and_defaults(tmp_path):
    config = load_config(write(tmp_path, MINIMAL))
    assert config.output == Path("results/test")
    assert config.grid.children == (2, 4)
    assert config.grid.phases == (0.0,)
    assert config.grid.rotation == (False,)
    assert config.grid.scaling == (False,)
    assert (config.evaluation.calibration_window, config.evaluation.calibration_windows) == (
        512,
        64,
    )
    assert config.grid.seed == 0
    assert config.evaluation.window == 2048
    assert config.evaluation.max_windows is None
    assert config.model.dtype == "bfloat16"


def test_rejects_unknown_keys(tmp_path):
    with pytest.raises(TypeError):
        load_config(write(tmp_path, MINIMAL + "typo_key = 1\n"))


def test_rejects_empty_lists(tmp_path):
    with pytest.raises(ValueError, match=r"grid\.bits"):
        load_config(write(tmp_path, MINIMAL.replace("bits = [4]", "bits = []")))


SWEEP_CONFIGS = [
    p
    for p in sorted(Path("configs").glob("*.toml"))
    if not p.stem.startswith(("pool-", "swarm-", "speed-"))
]


@pytest.mark.parametrize("path", SWEEP_CONFIGS, ids=lambda p: p.stem)
def test_repository_configs_are_valid(path):
    config = load_config(path)
    assert config.name == path.stem
    assert config.output == Path("results") / path.stem
