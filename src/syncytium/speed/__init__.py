"""Speed of fused agents (H9): synchronization rounds, drafting agreement, network model."""

from syncytium.speed.acceptance import agreement_curve, greedy_choices, tokens_per_cycle
from syncytium.speed.config import AcceptanceConfig, SpeedConfig, load_speed_config
from syncytium.speed.network import (
    Network,
    Swarm,
    all_reduce_seconds,
    argmax_seconds,
    fused_pass_seconds,
    fused_tokens_per_second,
    solo_seconds,
)
from syncytium.speed.rounds import Round, model_rounds, trace_rounds

__all__ = [
    "AcceptanceConfig",
    "Network",
    "Round",
    "SpeedConfig",
    "Swarm",
    "agreement_curve",
    "all_reduce_seconds",
    "argmax_seconds",
    "fused_pass_seconds",
    "fused_tokens_per_second",
    "greedy_choices",
    "load_speed_config",
    "model_rounds",
    "solo_seconds",
    "tokens_per_cycle",
    "trace_rounds",
]
