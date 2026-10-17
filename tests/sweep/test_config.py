from pathlib import Path

import pytest

from holograph.sweep import load_config

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
    assert config.evaluation.window == 2048
    assert config.evaluation.max_windows is None
    assert config.model.dtype == "bfloat16"


def test_rejects_unknown_keys(tmp_path):
    with pytest.raises(TypeError):
        load_config(write(tmp_path, MINIMAL + "typo_key = 1\n"))


def test_rejects_empty_lists(tmp_path):
    with pytest.raises(ValueError, match=r"grid\.bits"):
        load_config(write(tmp_path, MINIMAL.replace("bits = [4]", "bits = []")))


@pytest.mark.parametrize("name", ["screen-0.8b", "pilot-0.8b"])
def test_repository_configs_are_valid(name):
    config = load_config(Path("configs") / f"{name}.toml")
    assert config.name == name
