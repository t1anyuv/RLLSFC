"""Centralized project path management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


class PathManager:
    """Singleton path manager for project-local resources and experiment outputs."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._project_root: Optional[Path] = None
            self._resource_base: Optional[Path] = None
            self._experiment_name: Optional[str] = None
            PathManager._initialized = True

    @property
    def project_root(self) -> Path:
        if self._project_root is None:
            self._project_root = self._detect_project_root()
        return self._project_root

    @staticmethod
    def _detect_project_root() -> Path:
        env_root = os.environ.get("PROJECT_ROOT")
        if env_root:
            root = Path(env_root)
            if root.exists() and (root / "src").is_dir():
                return root.resolve()

        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "src").is_dir():
                return parent

        cwd = Path.cwd()
        if (cwd / "src").is_dir():
            return cwd
        return cwd

    def set_project_root(self, path: Union[str, Path]) -> None:
        self._project_root = Path(path).resolve()

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.project_root / candidate).resolve()

    @property
    def resource_base(self) -> Path:
        if self._resource_base is None:
            base = os.environ.get("RESOURCE_BASE_DIR", "resource")
            self._resource_base = self._resolve_path(base)
        return self._resource_base

    def set_resource_base(self, path: Union[str, Path]) -> None:
        self._resource_base = self._resolve_path(path)

    @property
    def experiment_name(self) -> Optional[str]:
        return self._experiment_name

    def set_experiment_name(self, name: str) -> None:
        self._experiment_name = name

    def get_experiment_dir(self, experiment_name: Optional[str] = None) -> Path:
        name = experiment_name or self._experiment_name or "default"
        exp_dir = self.resource_base / "experiments" / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def get_model_dir(self, experiment_name: Optional[str] = None) -> Path:
        name = experiment_name or self._experiment_name
        base = self.get_experiment_dir(name) / "models" if name else self.resource_base / "models"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def get_order_dir(self, experiment_name: Optional[str] = None) -> Path:
        name = experiment_name or self._experiment_name
        base = self.get_experiment_dir(name) / "orders" if name else self.resource_base / "orders"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def get_log_dir(self, experiment_name: Optional[str] = None) -> Path:
        name = experiment_name or self._experiment_name
        base = self.get_experiment_dir(name) / "logs" if name else self.resource_base / "logs"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def get_curves_dir(self, experiment_name: Optional[str] = None) -> Path:
        name = experiment_name or self._experiment_name
        base = self.get_experiment_dir(name) / "curves" if name else self.resource_base / "curves"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def get_queries_dir(self, experiment_name: Optional[str] = None) -> Path:
        name = experiment_name or self._experiment_name
        base = self.get_experiment_dir(name) / "queries" if name else self.resource_base / "queries"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def get_similarity_dir(self) -> Path:
        base = self.resource_base / "shared" / "similarity"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def get_similarity_matrix_path(self, filename: str) -> Path:
        path = Path(filename)
        if path.is_absolute():
            return path
        if "/" in filename or "\\" in filename:
            return self._resolve_path(filename)
        return self.get_similarity_dir() / filename

    def ensure_dir(self, path: Union[str, Path]) -> Path:
        resolved = Path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def validate_path(self, path: Union[str, Path], must_exist: bool = False) -> bool:
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


_path_manager = PathManager()


def get_path_manager() -> PathManager:
    return _path_manager


def get_project_root() -> Path:
    return _path_manager.project_root
