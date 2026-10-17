import pytest

from syncytium.eval import count_parameters, dense_footprint
from syncytium.models import quantized_weights
from syncytium.pool import assign_stages, device_footprints, stage_of
from syncytium.quant import Scheme


def test_weights_outside_the_decoder_layers_go_to_stage_zero():
    assert stage_of("model.embed_tokens.weight", num_layers=24, num_stages=4) == 0
    assert stage_of("model.norm.weight", num_layers=24, num_stages=4) == 0


def test_decoder_layers_split_into_contiguous_equal_blocks():
    stages = [stage_of(f"model.layers.{i}.mlp.up_proj.weight", 24, 4) for i in range(24)]
    assert stages == [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6


def test_assigns_every_tiny_model_weight(tiny_model):
    stages = assign_stages(quantized_weights(tiny_model), num_stages=2)
    assert stages["model.embed_tokens.weight"] == 0
    assert stages["model.layers.1.mlp.down_proj.weight"] == 1
    assert stages["model.layers.2.mlp.down_proj.weight"] == 2


def test_more_stages_than_layers_is_refused(tiny_model):
    with pytest.raises(ValueError, match="cannot fill"):
        assign_stages(quantized_weights(tiny_model), num_stages=5)


def test_devices_cover_the_model_and_each_holds_less(tiny_model):
    scheme = Scheme(bits=4, group_size=32, num_children=3)
    devices = device_footprints(tiny_model, scheme, num_stages=2)
    counts = count_parameters(tiny_model, quantized_weights(tiny_model))
    assert len(devices) == 3
    assert sum(device.params for device in devices) == counts.total
    assert all(device.total_bits < dense_footprint(counts).total_bits for device in devices)
