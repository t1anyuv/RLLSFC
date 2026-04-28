"""Shared foundational types and evaluators."""

from typing import Any

__all__ = ["SpatialBoundingBox", "TraversalCostEvaluator"]


def __getattr__(name: str) -> Any:
    if name == "SpatialBoundingBox":
        from .spatial import SpatialBoundingBox

        return SpatialBoundingBox
    if name == "TraversalCostEvaluator":
        from .costs import TraversalCostEvaluator

        return TraversalCostEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
