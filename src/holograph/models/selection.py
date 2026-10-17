"""Choosing which weights are quantized.

Quantized: the weights of every linear and embedding layer. Kept in full precision: norms,
convolutions and the Gated DeltaNet decay parameters, which are small and sensitive. Tied
weights (Qwen3.5 ties the output head to the embeddings) are listed once.
"""

from torch import nn

QUANTIZED_MODULES = (nn.Linear, nn.Embedding)


def quantized_weights(model: nn.Module) -> dict[str, nn.Parameter]:
    """Map the name of each weight to quantize to its parameter, in module order."""
    selected: dict[str, nn.Parameter] = {}
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, QUANTIZED_MODULES) and id(module.weight) not in seen:
            seen.add(id(module.weight))
            selected[f"{module_name}.weight"] = module.weight
    return selected
