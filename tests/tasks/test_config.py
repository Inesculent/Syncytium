from pathlib import Path

import pytest

from syncytium.tasks import load_task_config

CONFIGS = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize("name", [p.stem for p in sorted(CONFIGS.glob("swarm-*.toml"))])
def test_shipped_configs_load(name):
    config = load_task_config(CONFIGS / f"{name}.toml")
    assert config.name == name
    assert config.output == Path("results") / name
    assert config.setting.scheme.num_children == 4
    assert config.gsm8k is not None
    assert config.mmlu is not None


def test_fixed_configs_run_the_pre_registered_setting():
    for name in ("swarm-9b", "swarm-4b"):
        config = load_task_config(CONFIGS / f"{name}.toml")
        scheme = config.setting.scheme
        assert (scheme.bits, scheme.group_size, scheme.phase) == (3, 64, 0.5)
        assert scheme.rotation_seed == 0
        assert scheme.scaling
        assert config.gsm8k.split == config.mmlu.split == "test"
        assert config.gsm8k.limit is None
        assert config.gsm8k.max_new_tokens == 256


def test_smoke_config_never_touches_test_splits():
    config = load_task_config(CONFIGS / "swarm-smoke-0.8b.toml")
    assert config.gsm8k.split == "train"
    assert config.mmlu.split == "dev"


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


BASE = """
name = "x"
output = "results/x"
[model]
path = "m"
[setting]
bits = 3
group_size = 64
children = 4
"""


def test_a_config_needs_a_task(tmp_path):
    with pytest.raises(ValueError, match="names no task"):
        load_task_config(write(tmp_path, BASE))


def test_unknown_estimator_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_task_config(write(tmp_path, BASE + 'estimator = "median"\n[gsm8k]\n'))


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(TypeError):
        load_task_config(write(tmp_path, BASE + "[gsm8k]\nshots = 4\n"))
