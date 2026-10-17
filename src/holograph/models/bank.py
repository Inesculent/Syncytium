"""Writing simulated weights into a loaded model."""

from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn

Transform = Callable[[Tensor], Tensor]


class WeightBank:
    """Keeps a CPU copy of the original weights and writes transformed versions into the model.

    Every :meth:`apply` starts from the originals, so conditions never compound. Transforms
    run in float32 on the parameter's device; results are cast back to the parameter's dtype
    (bf16 for the models used here), as a dequantized model would be served.

    Args:
        weights: The parameters to manage, from
            :func:`~holograph.models.selection.quantized_weights`.
    """

    def __init__(self, weights: Mapping[str, nn.Parameter]) -> None:
        self._weights = dict(weights)
        self._originals = {name: w.detach().to("cpu", copy=True) for name, w in weights.items()}

    @torch.no_grad()
    def apply(self, transform: Transform) -> None:
        """Replace every weight with ``transform`` of its original value."""
        for name, weight in self._weights.items():
            original = self._originals[name].to(weight.device, torch.float32)
            weight.copy_(transform(original))

    @torch.no_grad()
    def restore(self) -> None:
        """Put the original weights back."""
        for name, weight in self._weights.items():
            weight.copy_(self._originals[name])
