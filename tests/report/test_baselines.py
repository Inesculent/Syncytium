import json
import math

import pytest

from syncytium.report import all_figures, load_results, write_report
from syncytium.report.baselines import by_survivors, compare, quantized_share, to_markdown

# Bits per weight of a 4-bit child, and how much each extra code bit adds.
CHILD_BPW, SHARE = 4.5, 0.9


def row(kind, perplexity, indices, n=4, bits=4, phase=0.0, rotation_seed=None, bpw=CHILD_BPW):
    offsets = [f"{k}/{n}" if phase == 0 else f"{2 * k + 1}/{2 * n}" for k in indices]
    return {
        "id": f"{kind}/b{bits}-n{n}-p{phase}-r{rotation_seed}/k{indices}",
        "kind": kind,
        "bits": bits,
        "group_size": 64,
        "num_children": n,
        "phase": phase,
        "rotation_seed": rotation_seed,
        "indices": list(indices),
        "offsets": offsets,
        "perplexity": perplexity,
        "bits_per_weight": bpw,
    }


def rtn(bits, perplexity):
    return row(
        "rtn", perplexity, (0,), n=1, bits=bits, phase=0.5, bpw=CHILD_BPW + SHARE * (bits - 4)
    )


@pytest.fixture
def results(tmp_path):
    rows = [
        {"id": "bf16", "kind": "bf16", "indices": [], "perplexity": 10.0, "bits_per_weight": 16},
        row("child", 16.0, (0,)),
        row("child", 14.0, (1,)),
        row("child", 13.0, (2,)),  # offset ½: round-to-nearest, the coded scheme's base
        row("child", 15.0, (3,)),
        row("rebuild", 10.5, (0, 1, 2, 3), bpw=6.5),
        row("partial", 11.0, (0, 1, 2), bpw=6.5),
        row("partial", 11.5, (1, 2, 3), bpw=6.5),
        rtn(4, 13.0),
        rtn(5, 11.8),
        rtn(6, 10.6),
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return load_results(path)


def pick(frame, scheme, survivors, k=None):
    rows = frame[(frame["scheme"] == scheme) & (frame["survivors"] == survivors)]
    if k is not None:
        rows = rows[rows["k"] == k]
    (value,) = rows["perplexity"]
    return value


def test_quantized_share_is_the_memory_of_one_code_bit(results):
    assert quantized_share(results) == pytest.approx(SHARE)


def test_quantized_share_needs_two_bit_widths(results):
    with pytest.raises(ValueError, match="two bit widths"):
        quantized_share(results[results["bits"] != 5].query("bits != 6"))


def test_staggered_takes_the_worst_case_at_each_survivor_count(results):
    comparison = compare(results)
    assert pick(comparison, "staggered", 1) == 16.0
    assert math.isnan(pick(comparison, "staggered", 2))  # not measured
    assert pick(comparison, "staggered", 3) == 11.5
    assert pick(comparison, "staggered", 4) == 10.5


def test_replicated_ignores_survivors(results):
    comparison = compare(results)
    assert {pick(comparison, "replicated", s) for s in range(1, 5)} == {13.0}


def test_coded_serves_its_base_until_k_nodes_survive(results):
    comparison = compare(results)
    assert [pick(comparison, "coded", s, k=3) for s in range(1, 5)] == [13.0, 13.0, 10.5, 10.5]
    coded = comparison[comparison["scheme"] == "coded"].drop_duplicates("k").set_index("k")
    for k in range(1, 5):
        assert coded.loc[k, "node_bpw"] == pytest.approx(CHILD_BPW + SHARE * 2 / k)


def test_base_child_is_the_one_closest_to_one_half(tmp_path):
    # With φ = ½ and N = 4 the offsets are 1/8, 3/8, 5/8 and 7/8; 3/8 and 5/8 tie and the
    # lower index wins.
    rows = [
        {"id": "bf16", "kind": "bf16", "indices": [], "perplexity": 10.0, "bits_per_weight": 16},
        *(row("child", 20.0 + k, (k,), phase=0.5) for k in range(4)),
        row("rebuild", 10.5, (0, 1, 2, 3), phase=0.5, bpw=6.5),
        rtn(4, 13.0),
        rtn(5, 11.8),
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    comparison = compare(load_results(path))
    assert pick(comparison, "coded", 1, k=4) == 21.0


def test_survivor_table_compares_against_the_cheapest_rivals(results):
    table = by_survivors(results).set_index("survivors")
    assert table.loc[3, "staggered"] == 11.5
    assert table.loc[3, "replicated"] == 13.0
    # Round-to-nearest first matches 11.5 at 6 bits: two more bits per node.
    assert table.loc[3, "replicated_match_bpw"] == pytest.approx(2 * SHARE)
    assert table.loc[1, "replicated_match_bpw"] == 0  # the same model beats the worst child
    assert table.loc[3, "coded"] == 10.5
    assert table.loc[3, "coded_extra_bpw"] == pytest.approx(SHARE * 2 / 3)
    assert math.isnan(table.loc[2, "replicated_match_bpw"])  # staggered not measured


def test_markdown_explains_the_columns(results):
    text = to_markdown(by_survivors(results), quantized_share(results))
    assert text.startswith("Worst-case perplexity")
    assert "coded_extra_bpw" in text and "90.0%" in text


def test_real_sweep_log_compares_and_reports(results_dir, tmp_path):
    results = load_results(results_dir / "results.jsonl")
    table = by_survivors(results)
    for n, rows in table.groupby("num_children"):
        measured = rows[rows["survivors"].isin({1, n - 1, n})]
        assert measured["staggered"].notna().all()
    figure = all_figures(results)["baselines-p0.5-rot-on"]
    assert [axis.get_title() for axis in figure.axes] == ["3-bit children", "4-bit children"]
    write_report(results_dir, tmp_path)
    assert (tmp_path / "baselines.md").exists()
    assert (tmp_path / "baselines-p0-rot-off.svg").exists()
