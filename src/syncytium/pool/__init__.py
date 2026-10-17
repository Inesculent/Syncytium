"""Pooled pipeline: stages of the parent held by devices that each store one child."""

from syncytium.pool.conditions import Kind, PoolCondition, plan, random_patterns
from syncytium.pool.config import PoolConfig, PoolGridConfig, load_pool_config
from syncytium.pool.layout import assign_stages, device_footprints, stage_of
from syncytium.pool.runner import run_pool, run_states
from syncytium.pool.summary import additivity, load_pool_results, to_markdown, write_pool_report

__all__ = [
    "Kind",
    "PoolCondition",
    "PoolConfig",
    "PoolGridConfig",
    "additivity",
    "assign_stages",
    "device_footprints",
    "load_pool_config",
    "load_pool_results",
    "plan",
    "random_patterns",
    "run_pool",
    "run_states",
    "stage_of",
    "to_markdown",
    "write_pool_report",
]
