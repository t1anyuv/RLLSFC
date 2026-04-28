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
) -> Dict[int, Dict[str, Any]]:
    qc_to_info: Dict[int, Dict[str, Any]] = {}

    for idx, cell in enumerate(active_nodes):
        quadrant_code = cell.get_quadrant_code(max_level)
        info = {
            "order": idx,
            "cell": cell,
        }
        qc_to_info[quadrant_code] = info

    return qc_to_info


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


def _assemble_ordering(qc_to_info: Dict[int, Dict[str, Any]], max_level: int) -> List[Dict[str, Any]]:
    sorted_infos = sorted(qc_to_info.values(), key=lambda info: info["order"])
    ordering: List[Dict[str, Any]] = []

    for info in sorted_infos:
        cell = info["cell"]
        quad_code = int(cell.get_quadrant_code(max_level))
        ordering.append(
            {
                "quad_code": quad_code,
                "order": info["order"],
                "parent": _build_parent_descriptor(cell, max_level),
            }
        )

    return ordering


def _build_effective_tree(active_nodes: List[QuadTreeCell]) -> Dict[QuadTreeCell, List[QuadTreeCell]]:
    active_set = set(active_nodes)
    children_map: Dict[QuadTreeCell, List[QuadTreeCell]] = {cell: [] for cell in active_nodes}

    for cell in active_nodes:
        if cell.parent is None:
            continue
        current = cell.parent
        while current is not None and current not in active_set:
            current = current.parent
        if current is not None:
            children_map[current].append(cell)

    return children_map


def _annotate_effective_subtree(
        ordering: List[Dict[str, Any]],
        active_nodes: List[QuadTreeCell],
        order_source: str,
) -> bool:
    if not ordering or not active_nodes:
        return "xz" in order_source.lower()

    order_by_cell = {cell: idx for idx, cell in enumerate(active_nodes)}
    children_map = _build_effective_tree(active_nodes)
    is_xz_order = "xz" in order_source.lower()

    def collect_descendant_orders(cell: QuadTreeCell) -> List[int]:
        collected: List[int] = []
        stack = list(children_map.get(cell, []))
        while stack:
            current = stack.pop()
            collected.append(order_by_cell[current])
            stack.extend(children_map.get(current, []))
        collected.sort()
        return collected

    for entry, cell in zip(ordering, active_nodes):
        descendant_orders = collect_descendant_orders(cell)
        coverage: Dict[str, Any] = {
            "effective_subtree_count": len(descendant_orders),
        }
        if not is_xz_order:
            coverage["effective_subtree_orders"] = descendant_orders
        entry["coverage"] = coverage

    return is_xz_order


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
            self.output_dir = config.experiment.get_results_dir(config.paths)
        self.logger = logger or setup_logging("TrajectoryOrderFormatter")
        self.config = config
        self.active_ordering: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def _build_payload(
            self,
            order: List[QuadTreeCell],
            quadtree: QuadTreeIndex,
            global_alpha: int,
            global_beta: int,
            order_source: str,
    ) -> Dict[str, Any]:
        if not order:
            self.logger.warning("No quadorder found.")
            return {}

        max_level = quadtree.max_level
        active_nodes = _get_unique_active_nodes(order, quadtree)
        qc_to_info = _build_active_map(active_nodes, max_level)
        self.active_ordering = _assemble_ordering(qc_to_info, max_level)
        effective_subtree_contiguous = _annotate_effective_subtree(
            self.active_ordering, active_nodes, order_source
        )
        self.metadata = self._generate_metadata(
            quadtree,
            active_nodes,
            global_alpha=global_alpha,
            global_beta=global_beta,
            order_source=order_source,
            effective_subtree_contiguous=effective_subtree_contiguous,
        )

        self.logger.info("Generated %s ordering groups.", len(self.active_ordering))
        return {"ordering": self.active_ordering, "metadata": self.metadata}

    def process_order(self, env: TraversalEnvironment, quadtree: QuadTreeIndex) -> Dict[str, Any]:
        self.logger.info("Start formatting quadorder output.")
        return self._build_payload(
            order=env.quadorder(),
            quadtree=quadtree,
            global_alpha=int(env.alpha),
            global_beta=int(env.beta),
            order_source="rl_quadorder",
        )

    def process_order_list(
            self,
            order: List[QuadTreeCell],
            quadtree: QuadTreeIndex,
            global_alpha: Optional[int] = None,
            global_beta: Optional[int] = None,
            order_source: str = "default_xz_order",
    ) -> Dict[str, Any]:
        self.logger.info("Start formatting explicit traversal order output.")
        return self._build_payload(
            order=order,
            quadtree=quadtree,
            global_alpha=int(global_alpha if global_alpha is not None else quadtree.alpha),
            global_beta=int(global_beta if global_beta is not None else quadtree.beta),
            order_source=order_source,
        )

    def _generate_metadata(
            self,
            quadtree: QuadTreeIndex,
            active_nodes: List[QuadTreeCell],
            global_alpha: int,
            global_beta: int,
            order_source: str,
            effective_subtree_contiguous: bool,
    ) -> Dict[str, Any]:
        all_cells = list(quadtree.all_cells.values())
        max_shape_count = 0
        max_partition = max(
            (int(cell.alpha) * int(cell.beta) for cell in all_cells),
            default=int(global_alpha) * int(global_beta),
        )
        min_trajs: Optional[int] = None
        partition_search = [[2, 2], [8, 8]]
        if self.config is not None:
            min_trajs = int(self.config.index.min_cell_trajs)

        for cell in all_cells:
            if not cell.signatures:
                continue
            max_shape_count = max(max_shape_count, len(set(cell.signatures.values())))

        return {
            "total_cells": len(all_cells),
            "active_cells": len(active_nodes),
            "spatial_boundary": {
                "xmin": float(quadtree.bbox.min_x),
                "ymin": float(quadtree.bbox.min_y),
                "xmax": float(quadtree.bbox.max_x),
                "ymax": float(quadtree.bbox.max_y),
            },
            "quadtree_max_level": int(quadtree.max_level),
            "global_alpha": int(global_alpha),
            "global_beta": int(global_beta),
            "generation_timestamp": datetime.now().isoformat(),
            "version": self.CONFIG_VERSION,
            "order_source": order_source,
            "effective_subtree_contiguous": effective_subtree_contiguous,
            "partition_search": partition_search,
            "max_partition": max_partition,
            "max_shape_count": max_shape_count,
            "min_trajs": min_trajs,
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

    def generate_config_file_from_order(
            self,
            order: List[QuadTreeCell],
            quadtree: QuadTreeIndex,
            filename: str = "quadorder.json",
            global_alpha: Optional[int] = None,
            global_beta: Optional[int] = None,
            order_source: str = "default_xz_order",
    ) -> Tuple[Dict[str, Any], str]:
        data = self.process_order_list(
            order=order,
            quadtree=quadtree,
            global_alpha=global_alpha,
            global_beta=global_beta,
            order_source=order_source,
        )
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
