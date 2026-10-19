import json
from pathlib import Path

import pytest

from syncytium.runs import Kind, State, inventory, table
from syncytium.runs.status import kind_of, purpose

REPOSITORY = Path(__file__).parents[2]

SWEEP = """# A tiny sweep. It has two sentences.
name = "tiny"
output = "results/tiny"
[model]
path = "m"
[grid]
bits = [4]
group_sizes = [32]
children = [2]
max_missing = 0
"""

TASKS = """# Tasks
name = "agents"
output = "results/agents"
[model]
path = "m"
[setting]
bits = 3
group_size = 64
children = 2
drop = false
[gsm8k]
"""


@pytest.fixture
def root(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "tiny.toml").write_text(SWEEP, encoding="utf-8")
    (tmp_path / "configs" / "agents.toml").write_text(TASKS, encoding="utf-8")
    return tmp_path


def log(root, name, ids, job="12345"):
    out = root / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in ids), "utf-8")
    run = {"started": "2026-10-18T00:00:00", "git_commit": "abcdef123", "slurm_job": job}
    (out / "runs.jsonl").write_text(json.dumps(run) + "\n", encoding="utf-8")


def test_runs_are_missing_partial_or_complete(root):
    runs, strays = inventory(root)
    by_name = {r.name: r for r in runs}
    assert by_name["tiny"].expected == 4  # bf16, two children, the rebuild
    assert by_name["agents"].expected == 4  # bf16, two children, the group
    assert all(r.state == State.MISSING for r in runs)
    log(root, "tiny", ["a", "b"])
    log(root, "agents", ["a", "b", "c", "d"])
    log(root, "copied-somewhere", ["a"])
    runs, strays = inventory(root)
    by_name = {r.name: r for r in runs}
    assert by_name["tiny"].state == State.PARTIAL
    assert by_name["agents"].state == State.COMPLETE
    assert (by_name["agents"].commit, by_name["agents"].job) == ("abcdef1", "12345")
    assert strays == ["copied-somewhere"]
    text = table(runs, strays)
    assert "Results with no config: copied-somewhere" in text
    assert "A tiny sweep" in text


def test_purpose_is_the_first_sentence_of_the_opening_comment(root):
    assert purpose(root / "configs" / "tiny.toml") == "A tiny sweep"


def test_every_repository_config_is_recognized():
    configs = sorted((REPOSITORY / "configs").glob("*.toml"))
    kinds = {kind_of(path) for path in configs}
    assert kinds == set(Kind)
    runs, _ = inventory(REPOSITORY)
    assert len(runs) == len(configs)
    assert all(run.expected > 0 and run.purpose for run in runs)


LIVE = """# A live run. Scenarios on two networks.
name = "demo"
output = "results/demo"
[model]
path = "m"
[live]
children = "shards/c"
baseline = "shards/b"
storage = "/tmp/s"
"""


def test_a_live_run_counts_scenarios_whose_baseline_finished(root):
    (root / "configs" / "demo.toml").write_text(LIVE, encoding="utf-8")
    for name, events in (("A-lan", ["scenario", "stopped"]), ("B-lan", ["scenario"])):
        log = root / "results" / "demo" / name / "pipeline" / "events.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text("".join(json.dumps({"event": e}) + "\n" for e in events), "utf-8")
    run = {r.name: r for r in inventory(root)[0]}["demo"]
    assert (run.kind, run.expected, run.logged, run.state) == (Kind.LIVE, 6, 1, State.PARTIAL)
