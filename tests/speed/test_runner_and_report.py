import json
from pathlib import Path

import pytest
import torch

from syncytium.eval import ParameterCounts
from syncytium.speed import AcceptanceConfig, Round, SpeedConfig, load_speed_config
from syncytium.speed import report as report_module
from syncytium.speed.runner import ACCEPTANCE, measure_agreement
from syncytium.sweep.config import EvalConfig, ModelConfig
from syncytium.tasks import SwarmSetting

CONFIGS = Path(__file__).parents[2] / "configs"


def config(tmp_path):
    return SpeedConfig(
        name="tiny",
        output=tmp_path,
        model=ModelConfig(path="unused"),
        evaluation=EvalConfig(),
        # The tiny model's narrowest input dimension is 32, so groups of 32 fit every weight.
        setting=SwarmSetting(bits=4, group_size=32, children=4, phase=0.5),
        acceptance=AcceptanceConfig(drafts=(1, 2, 4, 8)),
    )


def rows(path):
    with (path / ACCEPTANCE).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def test_every_child_is_measured_once(tiny_model, tmp_path):
    batches = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(0))
    measure_agreement(tiny_model, batches, config(tmp_path))
    found = rows(tmp_path)
    assert [r["child"] for r in found] == [0, 1, 2, 3]
    for r in found:
        assert 0 <= r["agreement"] <= 1
        assert r["positions"] == 32
        assert set(r["tokens_per_cycle"]) == {"1", "2", "4", "8"}
    measure_agreement(tiny_model, batches, config(tmp_path))
    assert len(rows(tmp_path)) == 4


@pytest.mark.parametrize("size", ["9b", "4b"])
def test_shipped_configs_use_the_swarm_setting(size):
    speed = load_speed_config(CONFIGS / f"speed-{size}.toml")
    swarm = (CONFIGS / f"swarm-{size}.toml").read_text(encoding="utf-8")
    assert speed.output == Path("results") / f"speed-{size}"
    assert f'path = "{speed.model.path}"' in swarm
    scheme = speed.setting.scheme
    assert (scheme.bits, scheme.group_size, scheme.num_children, scheme.phase) == (3, 64, 4, 0.5)
    assert scheme.scaling
    assert scheme.rotation_seed == 0


def test_report_judges_h9_on_the_worst_drafter(tmp_path, monkeypatch):
    rounds = [Round(("embed",), 4096)] + [Round((f"l{i}",), 12_000) for i in range(128)]
    rounds.append(Round(("head",), 248_320, head=True))
    monkeypatch.setattr(report_module, "model_rounds", lambda _: rounds)
    monkeypatch.setattr(
        report_module, "parameter_counts", lambda _: ParameterCounts(8_000_000_000, 64_000)
    )
    tpc = {str(k): 1 + 0.8 * k for k in range(1, 9)}
    lines = [
        {"id": f"child/k{k}", "child": k, "agreement": a, "positions": 10, "tokens_per_cycle": tpc}
        for k, a in enumerate([0.9, 0.7, 0.8, 0.95])
    ]
    (tmp_path / ACCEPTANCE).write_text("".join(json.dumps(r) + "\n" for r in lines), "utf-8")
    text = report_module.report(config(tmp_path))
    assert "130 rounds per fused token" in text
    assert "with child 1." in text
    assert "**H9: " in text
    assert text.count("| 0.2 ms | 1 Gbit/s | 100 GB/s |") == 1
