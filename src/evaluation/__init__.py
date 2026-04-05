"""Evaluation helpers comparing traversal orders."""

__all__ = ["TraversalPerformanceEvaluator"]


def __getattr__(name):
    if name == "TraversalPerformanceEvaluator":
        from .traversal_evaluator import TraversalPerformanceEvaluator

        return TraversalPerformanceEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
