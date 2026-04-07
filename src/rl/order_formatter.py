import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import TShapeConfig
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex
from src.rl.traversal_environment import TraversalEnvironment
from src.utils.logger import setup_logging


def _get_unique_active_nodes(order: List[QuadTreeCell], quadtree: QuadTreeIndex) -> List[QuadTreeCell]:
    seen_codes = set()
    active_nodes: List[QuadTreeCell] = []

    root = quadtree.root
    if not root.muted:
        active_nodes.append(root)
        seen_codes.add(root.get_quadrant_code(quadtree.max_level))

    for cell in order:
        if cell.muted:
            continue
        quadrant_code = cell.get_quadrant_code(quadtree.max_level)
        if quadrant_code in seen_codes:
            continue
        active_nodes.append(cell)
        seen_codes.add(quadrant_code)

    return active_nodes


def _build_active_map(
    active_nodes: List[QuadTreeCell],
    max_level: int,
) -> Tuple[Dict[QuadTreeCell, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    active_map: Dict[QuadTreeCell, Dict[str, Any]] = {}
    qc_to_info: Dict[int, Dict[str, Any]] = {}

    for idx, cell in enumerate(active_nodes):
        quadrant_code = cell.get_quadrant_code(max_level)
        info = {
            "order": idx,
            "cell": cell,
            "active_qc": quadrant_code,
            "muted_set": set(),
        }
        active_map[cell] = info
        qc_to_info[quadrant_code] = info

    return active_map, qc_to_info


def _build_parent_descriptor(cell: QuadTreeCell, max_level: int) -> Dict[str, Any]:
    return {
        "alpha": int(cell.alpha),
        "beta": int(cell.beta),
        "level": int(cell.level),
        "elementCode": int(cell.get_quadrant_code(max_level)),
        "xmin": float(cell.bbox.min_x),
        "ymin": float(cell.bbox.min_y),
        "xmax": float(cell.bbox.max_x),
        "ymax": float(cell.bbox.max_y),
    }


def _find_active_parent(cell: QuadTreeCell) -> Optional[QuadTreeCell]:
    current = cell.parent
    while current is not None:
        if not current.muted:
            return current
        current = current.parent
    return None


def _assemble_ordering(qc_to_info: Dict[int, Dict[str, Any]], max_level: int) -> List[Dict[str, Any]]:
    sorted_infos = sorted(qc_to_info.values(), key=lambda info: info["order"])
    ordering: List[Dict[str, Any]] = []

    for info in sorted_infos:
        cell = info["cell"]
        combined_codes = [info["active_qc"]] + sorted(info["muted_set"])
        ordering.append(
            {
                "quad_code": combined_codes,
                "order": info["order"],
                "parent": _build_parent_descriptor(cell, max_level),
            }
        )

    return ordering


class TrajectoryOrderFormatter:
    """Generate the exported traversal order JSON."""

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
            self.output_dir = config.experiment.get_orders_dir()
        self.logger = logger or setup_logging("TrajectoryOrderFormatter")
        self.config = config
        self.active_ordering: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def process_order(self, env: TraversalEnvironment, quadtree: QuadTreeIndex) -> Dict[str, Any]:
        self.logger.info("Start formatting quadorder output.")

        order = env.quadorder()
        if not order:
            self.logger.warning("No quadorder found.")
            return {}

        max_level = quadtree.max_level
        active_nodes = _get_unique_active_nodes(order, quadtree)
        active_map, qc_to_info = _build_active_map(active_nodes, max_level)
        self._assign_muted_nodes(quadtree, active_map, qc_to_info, max_level)
        self.active_ordering = _assemble_ordering(qc_to_info, max_level)
        self.metadata = self._generate_metadata(env, quadtree, active_nodes)

        self.logger.info("Generated %s ordering groups.", len(self.active_ordering))
        return {"ordering": self.active_ordering, "metadata": self.metadata}

    def _assign_muted_nodes(
        self,
        quadtree: QuadTreeIndex,
        active_map: Dict[QuadTreeCell, Dict[str, Any]],
        qc_to_info: Dict[int, Dict[str, Any]],
        max_level: int,
    ) -> None:
        active_qcs = set(qc_to_info.keys())

        for cell in quadtree.all_cells.values():
            if not cell.muted:
                continue

            muted_quad_code = cell.get_quadrant_code(max_level)
            if muted_quad_code in active_qcs:
                continue

            parent = _find_active_parent(cell)
            if parent in active_map:
                active_map[parent]["muted_set"].add(muted_quad_code)

    def _generate_metadata(
        self,
        env: TraversalEnvironment,
        quadtree: QuadTreeIndex,
        active_nodes: List[QuadTreeCell],
    ) -> Dict[str, Any]:
        all_cells = list(quadtree.all_cells.values())
        max_shape_count = 0
        max_partition_alpha = max((int(cell.alpha) for cell in all_cells), default=int(env.alpha))
        max_partition_beta = max((int(cell.beta) for cell in all_cells), default=int(env.beta))
        max_partition = max((int(cell.alpha) * int(cell.beta) for cell in all_cells), default=int(env.alpha) * int(env.beta))
        min_trajs: Optional[int] = None
        if self.config is not None:
            min_trajs = int(self.config.index.min_cell_trajs)

        for cell in all_cells:
            if not cell.signatures:
                continue
            max_shape_count = max(max_shape_count, len(set(cell.signatures.values())))

        return {
            "total_cells": len(all_cells),
            "active_cells": len(active_nodes),
            "muted_cells": len([cell for cell in all_cells if cell.muted]),
            "spatial_boundary": {
                "xmin": float(quadtree.bbox.min_x),
                "ymin": float(quadtree.bbox.min_y),
                "xmax": float(quadtree.bbox.max_x),
                "ymax": float(quadtree.bbox.max_y),
            },
            "quadtree_max_level": int(quadtree.max_level),
            "global_alpha": int(env.alpha),
            "global_beta": int(env.beta),
            "max_partition_alpha": max_partition_alpha,
            "max_partition_beta": max_partition_beta,
            "max_partition": max_partition,
            "max_shape_count": max_shape_count,
            "min_trajs": min_trajs,
            "generation_timestamp": datetime.now().isoformat(),
            "version": self.CONFIG_VERSION,
        }

    def save_json_config(self, data: Dict[str, Any], filename: str = "quadorder.json") -> str:
        filepath = self.output_dir / filename
        export_data = {"ordering": data["ordering"], "metadata": data["metadata"]}

        with open(filepath, "w", encoding="utf-8") as file_obj:
            json.dump(export_data, file_obj, indent=2, ensure_ascii=False)

        self.logger.info("Saved JSON config to %s", filepath)
        return str(filepath)

    def generate_config_file(
        self,
        env: TraversalEnvironment,
        quadtree: QuadTreeIndex,
        filename: str = "quadorder.json",
    ) -> Tuple[Dict[str, Any], str]:
        data = self.process_order(env, quadtree)
        if not data:
            return {}, ""

        json_path = self.save_json_config(data, filename)
        self.logger.info("Generated config file %s", filename)
        return data, json_path


def create_rl_postprocessor(
    output_dir: Optional[str] = None,
    config: Optional[TShapeConfig] = None,
) -> TrajectoryOrderFormatter:
    return TrajectoryOrderFormatter(output_dir=output_dir, config=config)

