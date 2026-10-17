import pytest
import torch
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

from holograph.sweep import ResultLog, plan, run_conditions
from holograph.sweep.config import GridConfig

GRID = GridConfig(
    bits=(3, 4),
    group_sizes=(32,),
    children=(2, 4),
    phases=(0.0, 0.5),
    rotation=(False, True),
    rtn_bits=(3, 4, 6),
)


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory, tiny_config):
    """A real sweep log produced by running a small grid on the tiny model."""
    directory = tmp_path_factory.mktemp("sweep")
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tiny_config).eval()
    batches = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(0))
    run_conditions(plan(GRID), model, batches, ResultLog(directory / "results.jsonl"))
    return directory
