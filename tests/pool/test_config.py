from pathlib import Path

import pytest

from syncytium.pool import PoolGridConfig, load_pool_config
from syncytium.quant import Scheme

MINIMAL = """
name = "test"
output = "results/test"

[model]
path = "some/model"

[pool]
bits = 4
group_size = 64
children = 4
stages = 4
estimators = ["mean"]
"""


def write(tmp_path, text):
    path = tmp_path / "pool.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_values_and_defaults(tmp_path):
    config = load_pool_config(write(tmp_path, MINIMAL))
    assert config.output == Path("results/test")
    assert config.pool.estimators == ("mean",)
    assert config.pool.max_lost == 4
    assert config.pool.scheme == Scheme(bits=4, group_size=64, num_children=4)


@pytest.mark.parametrize("path", sorted(Path("configs").glob("pool-*.toml")), ids=lambda p: p.stem)
def test_repository_configs_are_valid(path):
    config = load_pool_config(path)
    assert config.name == path.stem
    assert config.output == Path("results") / path.stem


def test_rotation_uses_the_seed():
    grid = PoolGridConfig(bits=4, group_size=64, children=4, stages=4, rotation=True, seed=3)
    assert grid.scheme.rotation_seed == 3


def test_rejects_unknown_keys(tmp_path):
    with pytest.raises(TypeError):
        load_pool_config(write(tmp_path, MINIMAL + "typo = 1\n"))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"children": 1}, "children"),
        ({"stages": 0}, "stages"),
        ({"estimators": ()}, "estimators"),
        ({"estimators": ("median",)}, "median"),
        ({"max_lost": 0}, "max_lost"),
        ({"max_lost": 7}, "max_lost"),  # 2 stages + 1, each can lose 2 of 3 children
        ({"samples": -1}, "samples"),
    ],
)
def test_rejects_invalid_grids(changes, message):
    settings = {"bits": 4, "group_size": 32, "children": 3, "stages": 2} | changes
    with pytest.raises(ValueError, match=message):
        PoolGridConfig(**settings)
