"""Evaluation: perplexity and weight memory."""

from syncytium.eval.corpus import load_wikitext2
from syncytium.eval.likelihood import perplexity, windows
from syncytium.eval.memory import (
    Footprint,
    ParameterCounts,
    child_footprint,
    count_parameters,
    dense_footprint,
    parent_footprint,
)

__all__ = [
    "Footprint",
    "ParameterCounts",
    "child_footprint",
    "count_parameters",
    "dense_footprint",
    "load_wikitext2",
    "parent_footprint",
    "perplexity",
    "windows",
]
