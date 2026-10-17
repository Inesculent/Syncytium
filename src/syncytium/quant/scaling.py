"""Shared activation-aware channel scaling (``docs/method.md``, "Optional channel scaling").

Before quantization, input channel c of a weight is multiplied by a scale s_c, and the
dequantized weight is divided by it again. Channels that see large activations get larger
scales, so their weights are rounded relatively more finely, as in AWQ. Every child uses the
same scales, and the transform is linear, so the rebuild identity still holds.

The scales are s = a^alpha, normalized so that max(s) min(s) = 1, where a_c is the root mean
square of input channel c over calibration text. alpha is chosen per weight from a grid,
minimizing the activation-weighted squared error Σ_c E[x_c²] Σ_r (ŵ_rc - w_rc)² of the
round-to-nearest child at the scheme's bit width: a diagonal approximation of the layer's
output error that needs no stored activations. alpha = 0 (no scaling) is on the grid, so the
search never does worse than leaving the weight unscaled.
"""

from collections.abc import Mapping

import torch
from torch import Tensor

from syncytium.quant.scheme import Scheme
from syncytium.quant.simulate import simulate_child

ALPHAS = tuple(step / 20 for step in range(21))


def candidate_scales(moments: Tensor, alpha: float) -> Tensor:
    """Scales a^alpha for second moments ``moments`` (E[x_c²]), normalized to max · min = 1."""
    rms = moments.clamp_min(torch.finfo(moments.dtype).tiny).sqrt()
    scales = rms**alpha
    return scales / (scales.max() * scales.min()).sqrt()


def search_scales(weight: Tensor, moments: Tensor, scheme: Scheme) -> Tensor:
    """Scales of the alpha on :data:`ALPHAS` that minimize the activation-weighted error.

    Args:
        weight: The weight, input channels along the last dimension.
        moments: E[x_c²] of each input channel over calibration text.
        scheme: Settings; the bit width, group size and rotation are used.
    """
    nearest = Scheme(scheme.bits, scheme.group_size, 1, 0.5, scheme.rotation_seed)
    moments = moments.to(weight)
    best, best_error = None, float("inf")
    for alpha in ALPHAS:
        scales = candidate_scales(moments, alpha)
        error = simulate_child(weight, nearest, 0, scales).sub_(weight).square_()
        total = float((error * moments).sum())
        if total < best_error:
            best, best_error = scales, total
    return best


class ChannelScales:
    """Searched scales of each weight, computed on first use and cached.

    Args:
        moments: E[x_c²] of the input of each weight, keyed by weight name. Weights without
            an entry (an embedding table that no linear layer reads) are left unscaled.
    """

    def __init__(self, moments: Mapping[str, Tensor]) -> None:
        self._moments = dict(moments)
        self._cache: dict[tuple, Tensor | None] = {}

    def __call__(self, name: str, weight: Tensor, scheme: Scheme) -> Tensor | None:
        """Scales for weight ``name`` under ``scheme``, or ``None`` to leave it unscaled."""
        key = (name, scheme.bits, scheme.group_size, scheme.rotation_seed)
        if key not in self._cache:
            moments = self._moments.get(name)
            self._cache[key] = None if moments is None else search_scales(weight, moments, scheme)
        return self._cache[key]
