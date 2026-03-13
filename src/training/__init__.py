"""Training orchestration for the traversal learning pipeline."""

# 延迟导入以避免循环依赖
# from .traversal_trainer import TraversalTrainer

__all__ = ["TraversalTrainer"]


def __getattr__(name):
    """延迟导入以避免循环依赖"""
    if name == "TraversalTrainer":
        from .traversal_trainer import TraversalTrainer
        return TraversalTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

