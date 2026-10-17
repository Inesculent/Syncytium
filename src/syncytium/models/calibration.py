"""Activation statistics for channel scaling, gathered from calibration text."""

from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn

from syncytium.models.selection import quantized_weights


@torch.inference_mode()
def input_moments(model: nn.Module, batches: Tensor) -> dict[str, Tensor]:
    """E[x_c²] of each input channel of every quantized linear weight, over ``batches``.

    Keys are the names from :func:`~syncytium.models.selection.quantized_weights`. A weight
    shared by several layers, such as an embedding table tied to the output head, gets the
    statistics of the linear layer that reads it; one that no linear layer reads gets none.

    Args:
        model: The unquantized model.
        batches: Token ids, one window per row.
    """
    names = {id(weight): name for name, weight in quantized_weights(model).items()}
    sums: dict[str, Tensor] = {}
    counts: dict[str, int] = {}

    def record(name: str) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        def hook(_: nn.Module, args: tuple[Tensor, ...]) -> None:
            inputs = args[0].detach().flatten(0, -2).float()
            sums[name] = sums.get(name, 0) + inputs.square().sum(dim=0)
            counts[name] = counts.get(name, 0) + inputs.shape[0]

        return hook

    handles = [
        module.register_forward_pre_hook(record(names[id(module.weight)]))
        for module in model.modules()
        if isinstance(module, nn.Linear) and id(module.weight) in names
    ]
    try:
        device = model.get_output_embeddings().weight.device
        for row in batches:
            model(row.unsqueeze(0).to(device))
    finally:
        for handle in handles:
            handle.remove()
    return {name: (total / counts[name]).cpu() for name, total in sums.items()}


def missing_moments(moments: Mapping[str, Tensor], model: nn.Module) -> list[str]:
    """Quantized weights without statistics; they are left unscaled."""
    return [name for name in quantized_weights(model) if name not in moments]
