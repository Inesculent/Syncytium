"""The timeline figure of one H18 run, drawn from its event logs.

One panel per system (the swarm, then the replicated pipeline), on its own clock: a lane per
device showing when it was starting, healing (regrowing or copying) and serving, with a tick
where the controller detected its death; and a service lane showing every request by route,
with the spans where the system could not answer shaded red. Dashed lines mark the scheduled
events.
"""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.patches import Patch

from syncytium.live.checks import read_events

LOSSES = ("crashed", "destroyed", "dead")
STATES = {"starting": "#adb5bd", "healing": "#f08c00", "serving": "#2b8a3e"}
ROUTES = {"alone": "#4dabf7", "agreed": "#4dabf7", "pipeline": "#4dabf7", "fused": "#7048e8"}
DOWN = "#e03131"
TITLES = {"swarm": "swarm", "pipeline": "replicated pipeline"}
NETWORKS = {"cluster": "cluster network", "lan": "emulated LAN"}

Interval = tuple[float, float, str]


class _Lanes:
    """Reads events in order and closes each device's intervals as its state changes."""

    def __init__(self, start: float) -> None:
        self.start = start
        self.open: dict[int, tuple[float, str]] = {}
        self.killed: dict[int, float] = {}
        self.intervals: dict[int, list[Interval]] = defaultdict(list)
        self.detections: list[tuple[int, float]] = []

    def read(self, event: dict[str, Any]) -> None:
        t, kind = event["time"] - self.start, event["event"]
        if kind == "registered":
            self.open[event["device"]] = (t, "starting")
        elif kind in ("regrowing", "copying"):
            for k in event.get("lost") or [event["device"]]:
                self._switch(k, t, "healing")
        elif kind == "admitted":
            self._switch(event["device"], t, "serving")
        elif kind in ("crashed", "destroyed") and not event.get("already_down"):
            self.killed[event["device"]] = t
        elif kind == "dead":
            self._die(event["device"], t, event["silence"])

    def finish(self, end: float) -> None:
        for k, (began, state) in self.open.items():
            self.intervals[k].append((began, end, state))

    def _switch(self, k: int, t: float, state: str) -> None:
        if k in self.open:
            began, previous = self.open[k]
            self.intervals[k].append((began, t, previous))
        self.open[k] = (t, state)

    def _die(self, k: int, t: float, silence: float) -> None:
        # Told to die: it ended then. Crashed on its own: at its last heartbeat, which can be
        # up to a heartbeat early, so never before its last change of state.
        ended = self.killed.pop(k, t - silence)
        if k in self.open:
            began, state = self.open.pop(k)
            self.intervals[k].append((began, max(ended, began), state))
        self.detections.append((k, t))


def lanes(events: Sequence[dict[str, Any]]) -> _Lanes:
    """Every device's intervals and every detected death, on the log's own clock."""
    reader = _Lanes(events[0]["time"])
    for event in events:
        reader.read(event)
    reader.finish(events[-1]["time"] - events[0]["time"])
    return reader


def down_spans(events: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    """When the system could not answer: from the loss that took it down to its next answer.

    A span starts at the first death or kill after the last answered request before a request
    went unanswered, and ends when a request is next answered (or at the end of the log).
    """
    start, spans = events[0]["time"], []
    since = None  # when the current run of losses began, if it may have taken the system down
    down = None  # when the current down span began
    for event in events:
        t = event["time"] - start
        if event["event"] in LOSSES and since is None:
            since = t
        if event["event"] != "request":
            continue
        if event["text"] is None and down is None:
            down = t - event["seconds"] if since is None else since
        elif event["text"] is not None:
            if down is not None:
                spans.append((down, t - event["seconds"]))
            since = down = None
    if down is not None:
        spans.append((down, events[-1]["time"] - start))
    return spans


def timeline(directory: Path) -> Figure:
    """The figure of the run in ``directory`` (named ``<scenario>-<network>``)."""
    scenario, network = directory.name.split("-", 1)
    figure = Figure(figsize=(11, 6.5), layout="constrained")
    for ax, system in zip(figure.subplots(2, 1), TITLES, strict=True):
        _draw(ax, read_events(directory / system / "events.jsonl"))
        ax.set_title(f"Scenario {scenario}, {NETWORKS[network]}: {TITLES[system]}")
    legend = [Patch(color=c, label=s) for s, c in STATES.items()]
    legend += [Patch(color=ROUTES["alone"], label="answered alone or by the pipeline")]
    legend += [
        Patch(color=ROUTES["fused"], label="answered fused"),
        Patch(color=DOWN, label="down"),
    ]
    figure.legend(handles=legend, loc="outside lower center", ncols=6, fontsize=8)
    return figure


def _draw(ax: Any, events: Sequence[dict[str, Any]]) -> None:
    start, service = events[0]["time"], -1.5
    devices = lanes(events)
    for k, intervals in devices.intervals.items():
        for began, end, state in intervals:
            ax.broken_barh([(began, end - began)], (k - 0.35, 0.7), color=STATES[state])
    for k, t in devices.detections:
        ax.plot([t, t], [k - 0.45, k + 0.45], color="black", linewidth=1.5)
    for began, end in down_spans(events):
        ax.axvspan(began, end, ymin=0, ymax=0.2, color=DOWN, alpha=0.35, linewidth=0)
    for event in (e for e in events if e["event"] == "request" and e["text"] is not None):
        began = event["time"] - start - event["seconds"]
        colour = ROUTES.get(event["route"], DOWN)
        ax.broken_barh([(began, event["seconds"])], (service - 0.35, 0.7), color=colour)
    steps = [e for e in events if e["event"] == "step" and e["label"].startswith("E")]
    for i, event in enumerate(steps):
        t = event["time"] - start
        ax.axvline(t, color="#495057", linewidth=0.8, linestyle="--")
        ax.text(t, 3.6 + 0.3 * (i % 2), event["label"], fontsize=8, ha="center")
    ax.set_yticks(
        [service, 0, 1, 2, 3], ["service", "device 0", "device 1", "device 2", "device 3"]
    )
    ax.set_ylim(-2.2, 4.2)
    ax.set_xlabel("seconds")
