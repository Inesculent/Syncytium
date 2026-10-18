"""Swarm task evaluation: lone and fused children on GSM8K and MMLU (Gate A)."""

from syncytium.tasks.analysis import Check, checks, load_items, report, write_report
from syncytium.tasks.conditions import Kind, TaskCondition, plan
from syncytium.tasks.config import (
    GSM8KConfig,
    MMLUConfig,
    SwarmSetting,
    TaskConfig,
    load_task_config,
)
from syncytium.tasks.generation import Completion, Decoding, continue_after_switch, generate
from syncytium.tasks.runner import TaskData, TaskRun, items_path, load_data, run_tasks

__all__ = [
    "Check",
    "Completion",
    "Decoding",
    "GSM8KConfig",
    "Kind",
    "MMLUConfig",
    "SwarmSetting",
    "TaskCondition",
    "TaskConfig",
    "TaskData",
    "TaskRun",
    "checks",
    "continue_after_switch",
    "generate",
    "items_path",
    "load_data",
    "load_items",
    "load_task_config",
    "plan",
    "report",
    "run_tasks",
    "write_report",
]
