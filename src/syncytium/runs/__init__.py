"""The status of every configured run: what it is, how far it got, whether it is reported."""

from syncytium.runs.status import Kind, Run, State, inspect, inventory, table

__all__ = ["Kind", "Run", "State", "inspect", "inventory", "table"]
