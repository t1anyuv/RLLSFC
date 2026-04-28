"""Adaptive partition export helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import TShapeConfig
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex
from src.utils.logger import setup_logging


def _build_parent_descriptor(cell: QuadTreeCell, max_level: int) -> Dict[str, Any]:
    return {
        "alpha": int(cell.alpha),
        "beta": int(cell.beta),
        "level": int(cell.level),
        "element_code": int(cell.get_quadrant_code(max_level)),
        "xmin": float(cell.bbox.min_x),
        "ymin": float(cell.bbox.min_y),
        "xmax": float(cell.bbox.max_x),
        "ymax": float(cell.bbox.max_y),
    }


class AdaptivePartitionFormatter:
    """Export signature-optimized partition metadata without traversal order."""

    CONFIG_VERSION = "1.4"

    def __init__(
        self,
        output_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[TShapeConfig] = None,
    ):
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = config.experiment.get_results_dir(config.paths)
        self.logger = logger or setup_logging("AdaptivePartitionFormatter")
        self.config = config

    def _build_entries(self, quadtree: QuadTreeIndex) -> List[Dict[str, Any]]:
        max_level = quadtree.max_level
        active_cells = [cell for cell in quadtree.get_all_cells() if not cell.muted]
        active_cells.sort(key=lambda cell: cell.get_quadrant_code(max_level))

        return [
            {
                "quad_code": int(cell.get_quadrant_code(max_level)),
                "parent": _build_parent_descriptor(cell, max_level),
            }
            for cell in active_cells
        ]

    def _build_metadata(self, quadtree: QuadTreeIndex, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_cells = list(quadtree.all_cells.values())
        max_shape_count = 0
        for cell in all_cells:
            if not cell.signatures:
                continue
            max_shape_count = max(max_shape_count, len(set(cell.signatures.values())))

        max_partition_alpha = max((int(cell.alpha) for cell in all_cells), default=int(quadtree.alpha))
        max_partition_beta = max((int(cell.beta) for cell in all_cells), default=int(quadtree.beta))
        max_partition = max(
            (int(cell.alpha) * int(cell.beta) for cell in all_cells),
            default=int(quadtree.alpha) * int(quadtree.beta),
        )
        min_trajs = int(self.config.index.min_cell_trajs) if self.config is not None else None

        return {
            "total_cells": len(all_cells),
            "active_cells": len(entries),
            "muted_cells": len([cell for cell in all_cells if cell.muted]),
            "spatial_boundary": {
                "xmin": float(quadtree.bbox.min_x),
                "ymin": float(quadtree.bbox.min_y),
                "xmax": float(quadtree.bbox.max_x),
                "ymax": float(quadtree.bbox.max_y),
            },
            "quadtree_max_level": int(quadtree.max_level),
            "global_alpha": int(quadtree.alpha),
            "global_beta": int(quadtree.beta),
            "max_partition_alpha": max_partition_alpha,
            "max_partition_beta": max_partition_beta,
            "max_partition": max_partition,
            "max_shape_count": max_shape_count,
            "min_trajs": min_trajs,
            "generation_timestamp": datetime.now().isoformat(),
            "version": self.CONFIG_VERSION,
            "export_type": "adaptive_partitions",
        }

    def export(self, quadtree: QuadTreeIndex) -> Dict[str, Any]:
        entries = self._build_entries(quadtree)
        metadata = self._build_metadata(quadtree, entries)
        return {"partitions": entries, "metadata": metadata}

    def save(self, payload: Dict[str, Any], filename: str) -> str:
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        self.logger.info("Saved adaptive partition config to %s", filepath)
        return str(filepath)
