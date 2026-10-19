"""H18's checks, computed from the runs' event logs and probes (``docs/preregistration-demo.md``).

``write_reference`` computes check (a)'s reference: the simulation, which is the children's mean
weights (the parent) in one model, teacher-forced on the probe prompts. ``write_checks`` reads
every scenario run so far and writes ``checks.json`` (checks (a) to (g) per run, plus the
reported-alongside numbers) and one timeline figure per run.
"""

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from syncytium.live.run import SCENARIOS, LiveConfig, problems
from syncytium.models import load_model, load_tokenizer
from syncytium.shards import MANIFEST_NAME, Manifest, rebuild_from_files, shard_name
from syncytium.speed.network import Network, leader_reduce_seconds
from syncytium.tasks import gsm8k

TAIL = 16  # positions per probe prompt whose logits are compared
AGREEMENT = 0.99  # check (a)
CAP = 8e9  # check (f), bytes
MODEL_FACTOR = 2.0  # check (g)
BASELINE_DOWN = {"A": {"P2"}, "B": {"P2", "P3"}, "C": set()}  # check (e)
REFUSALS = {"A": [], "B": [], "C": ["neighbour"]}  # check (c)


def write_reference(config: LiveConfig, device: str = "cuda") -> Path:
    """Teacher-force the probe prompts through the simulation; save top-1 tokens and tails."""
    manifest = Manifest.load(config.children / MANIFEST_NAME)
    shards = {k: config.children / shard_name(k) for k in range(manifest.scheme.num_children)}
    model = load_model(config.model, device)
    model.load_state_dict(rebuild_from_files(shards, manifest, torch.bfloat16), strict=False)
    tokenizer = load_tokenizer(config.model)
    prompts = []
    for problem in problems(config)[0]:
        ids = torch.tensor([tokenizer.encode(gsm8k.prompt(problem.question))], device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float().cpu()
        prompts.append({"top1": logits.argmax(dim=-1).tolist(), "tail": logits[-TAIL:]})
    path = config.output / "reference.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts": prompts}, path)
    return path


def read_events(path: Path) -> list[dict[str, Any]]:
    """The events of one log, in order."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def check_exact(probe: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Check (a): top-1 agreement of the live probe with the simulation, and logit gaps."""
    same = total = 0
    gaps = []
    for live, simulated in zip(probe["prompts"], reference["prompts"], strict=True):
        a, b = torch.tensor(live["top1"]), torch.tensor(simulated["top1"])
        same += int((a == b).sum())
        total += a.numel()
        gaps.append((live["tail"].float() - simulated["tail"].float()).abs())
    gap = torch.cat([g.flatten() for g in gaps])
    agreement = same / total
    return {
        "passed": agreement >= AGREEMENT,
        "agreement": agreement,
        "max_logit_gap": float(gap.max()),
        "mean_logit_gap": float(gap.mean()),
    }


def check_recovery(first: dict[str, Any], last: dict[str, Any]) -> dict[str, Any]:
    """Check (b): the probe after recovery is bit-identical to the first."""
    same = first["sha256"] == last["sha256"] and first["members"] == last["members"]
    consistent = first["consistent"] and last["consistent"]
    return {"passed": same and consistent, "first": first["sha256"], "last": last["sha256"]}


def check_healing(events: Sequence[dict[str, Any]], scenario: str) -> dict[str, Any]:
    """Check (c): every returning or regrown shard matches; the called-for refusals happen."""
    regrown = [e for e in events if e["event"] == "regrown"]
    admissions = [e for e in events if e["event"] in ("admitted", "refused")]
    refusals = [e["kind"] for e in events if e["event"] == "regrowth-refused"]
    passed = (
        bool(regrown)
        and all(e["matches"] for e in regrown)
        and all(e["event"] == "admitted" for e in admissions)
        and refusals == REFUSALS[scenario]
    )
    return {
        "passed": passed,
        "regrown": [
            {k: e.get(k) for k in ("device", "matches", "bytes", "seconds")} for e in regrown
        ],
        "refusals": refusals,
    }


def expected_requests(scenario: str, per_phase: int) -> int:
    """Requests a scenario serves: every phase's, plus a forced one where a member crashes."""
    steps = SCENARIOS[scenario]
    return sum(per_phase if a == "phase" else a == "crash_during" for _, a, _ in steps)


def check_serving(
    events: Sequence[dict[str, Any]], scenario: str, per_phase: int
) -> dict[str, Any]:
    """Check (d): every request of every phase is answered."""
    requests = [e for e in events if e["event"] == "request"]
    failed = [e["problem"] for e in requests if e["text"] is None]
    expected = expected_requests(scenario, per_phase)
    return {
        "passed": not failed and len(requests) == expected,
        "requests": len(requests),
        "expected": expected,
        "failed": failed,
    }


def check_baseline(events: Sequence[dict[str, Any]], scenario: str) -> dict[str, Any]:
    """Check (e): the baseline answers exactly while every stage has a live copy."""
    wrong = [
        e["problem"]
        for e in events
        if e["event"] == "request"
        and (e["route"] == "down") != (e["phase"] in BASELINE_DOWN[scenario])
    ]
    impossible = [e["device"] for e in events if e["event"] == "restore-impossible"]
    restored_stage_1 = [e for e in events if e["event"] == "copied" and e["stage"] == 0]
    passed = not wrong and (scenario != "B" or (impossible and not restored_stage_1))
    return {"passed": bool(passed), "wrong": wrong, "restore_impossible": impossible}


def check_memory(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Check (f): no device's GPU memory goes above the cap; regrowth's disk copies reported."""
    peaks: dict[int, int] = defaultdict(int)
    for e in events:
        if e["event"] == "peak":
            peaks[e["device"]] = max(peaks[e["device"]], e["bytes"])
    held = [e.get("packed_bytes") for e in events if e["event"] == "admitted"]
    moved = sum(e.get("bytes") or 0 for e in events if e["event"] in ("regrown", "copied"))
    return {
        "passed": bool(peaks) and max(peaks.values()) <= CAP,
        "peak_bytes": dict(peaks),
        "largest_model_bytes": max((b for b in held if b is not None), default=None),
        "bytes_moved": moved,
    }


def check_model(events: Sequence[dict[str, Any]], link: tuple[float, float]) -> dict[str, Any]:
    """Check (g): the mean averaging round is within a factor of two of the leader reduce."""
    network = Network(link[0], link[1], 0.0)
    rounds = measured = predicted = 0.0
    for e in events:
        if e["event"] != "request":
            continue
        for values, live, n, seconds in e.get("timing") or []:
            rounds += n
            measured += seconds
            predicted += n * leader_reduce_seconds(int(values), int(live), network)
    if not rounds:
        return {"passed": False, "rounds": 0}
    ratio = measured / predicted
    return {
        "passed": 1 / MODEL_FACTOR <= ratio <= MODEL_FACTOR,
        "rounds": int(rounds),
        "measured_seconds": measured / rounds,
        "predicted_seconds": predicted / rounds,
        "ratio": ratio,
    }


def phases(events: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per phase: members, answered, fused, accuracy, and tokens per second by route."""
    graded = {(e["problem"]): e["correct"] for e in events if e["event"] == "graded"}
    table: dict[str, dict[str, Any]] = {}
    for e in (e for e in events if e["event"] == "request"):
        row = table.setdefault(
            e["phase"], {"requests": 0, "answered": 0, "correct": 0, "members": [], "routes": {}}
        )
        row["requests"] += 1
        row["answered"] += e["text"] is not None
        row["correct"] += bool(graded.get(e["problem"]))
        row["members"] = e["members"]
        route = row["routes"].setdefault(e["route"], {"requests": 0, "tokens": 0, "seconds": 0.0})
        route["requests"] += 1
        route["tokens"] += e.get("tokens") or 0
        route["seconds"] += e["seconds"]
    for row in table.values():
        for route in row["routes"].values():
            route["tokens_per_second"] = (
                route["tokens"] / route["seconds"] if route["seconds"] else None
            )
    return table


def check_run(
    config: LiveConfig, directory: Path, reference: dict[str, Any] | None
) -> dict[str, Any]:
    """Every check of one scenario on one network."""
    scenario, network = directory.name.split("-", 1)
    swarm = read_events(directory / "swarm" / "events.jsonl")
    baseline = read_events(directory / "pipeline" / "events.jsonl")
    probes = sorted((directory / "swarm").glob("probe-*.pt"))
    first, last = (torch.load(p, weights_only=False) for p in (probes[0], probes[-1]))
    checks = {
        "a": check_exact(first, reference) if reference else {"passed": False, "missing": True},
        "b": check_recovery(first, last),
        "c": check_healing(swarm, scenario),
        "d": check_serving(swarm, scenario, config.per_phase),
        "e": check_baseline(baseline, scenario),
        "f": _both(check_memory(swarm), check_memory(baseline)),
    }
    if network == "lan":
        checks["g"] = check_model(swarm, config.lan)
    detected = [e["seconds"] for e in swarm if e["event"] == "detected"]
    return {
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
        "phases": {"swarm": phases(swarm), "baseline": phases(baseline)},
        "detection_seconds": detected,
    }


def write_checks(config: LiveConfig) -> dict[str, Any]:
    """Check every run in the output directory; write ``checks.json`` and the timelines."""
    path = config.output / "reference.pt"
    reference = torch.load(path, weights_only=False) if path.exists() else None
    runs = {}
    for directory in sorted(d for d in config.output.iterdir() if (d / "pipeline").is_dir()):
        runs[directory.name] = check_run(config, directory, reference)
        from syncytium.live.timeline import timeline  # it imports this module

        figure = timeline(directory)
        figure.savefig(config.output / f"timeline-{directory.name}.png", dpi=150)
    result = {"runs": runs, "passed": bool(runs) and all(r["passed"] for r in runs.values())}
    (config.output / "checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _both(swarm: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {"passed": swarm["passed"] and baseline["passed"], "swarm": swarm, "baseline": baseline}
