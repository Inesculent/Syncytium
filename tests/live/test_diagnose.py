import torch

from syncytium.live.diagnose import logits_report, nonfinite, scale_report, weight_gaps


def test_nonfinite_names_only_the_tensors_with_bad_values():
    tensors = {"a": torch.ones(3), "b": torch.tensor([1.0, float("nan"), float("inf")])}
    assert nonfinite(tensors) == {"b": 2}


def test_scale_report_flags_zero_and_nonfinite_scales():
    assert scale_report({}) == {"scaled": False}
    report = scale_report({"a": torch.tensor([0.5, 2.0]), "b": torch.tensor([0.0, float("inf")])})
    assert report["min"] == 0.0 and report["max"] == 2.0
    assert report["nonfinite"] == {"b": 1} and report["zero"] == ["b"]


def test_weight_gaps_put_nan_first():
    parent = {"a": torch.zeros(2), "b": torch.zeros(2), "c": torch.zeros(2)}
    rebuilt = {"a": torch.tensor([0.1, 0.0]), "b": torch.tensor([float("nan"), 0.0])}
    rebuilt["c"] = parent["c"]
    assert [name for name, _ in weight_gaps(rebuilt, parent)] == ["b", "a", "c"]


def test_logits_report_on_the_tiny_model(tiny_model):
    report = logits_report(tiny_model, [5, 6, 7])
    assert report["finite"] and report["positions_affected"] == 0
