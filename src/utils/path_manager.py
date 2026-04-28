"""Project path utilities.

Simple utilities for path resolution. For path management, use PathConfig from src.config.
"""

from pathlib import Path
from typing import Union


def resolve_path(path: Union[str, Path], base: Path) -> Path:
    """Resolve a path relative to a base directory.

    Args:
        path: The path to resolve
        base: The base directory for relative paths

    Returns:
        Resolved absolute path
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Resolved Path object
    """
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_path(path: Union[str, Path], must_exist: bool = False) -> bool:
    """Validate a path.

    Args:
        path: Path to validate
        must_exist: Whether the path must exist

    Returns:
        True if valid, False otherwise
    """
    if path is None:
        return False

    candidate = Path(path)
    if must_exist:
        return candidate.exists()

    try:
        candidate.resolve()
        return True
    except (OSError, RuntimeError):
        return False
