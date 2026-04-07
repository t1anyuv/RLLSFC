"""Merge adaptive partition metadata into an exported order file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将自适应划分信息更新到顺序文件")
    parser.add_argument("--order-file", type=str, required=True, help="顺序文件路径")
    parser.add_argument("--partitions-file", type=str, required=True, help="自适应划分文件路径")
    parser.add_argument("--output-file", type=str, default=None, help="输出路径；默认原地覆盖 order-file")
    return parser.parse_args()


def build_partition_lookup(partitions_payload: Dict) -> Dict[int, Tuple[int, int]]:
    lookup: Dict[int, Tuple[int, int]] = {}
    for item in partitions_payload.get("partitions", []):
        parent = item.get("parent", {})
        element_code = parent.get("elementCode")
        alpha = parent.get("alpha")
        beta = parent.get("beta")
        if element_code is None or alpha is None or beta is None:
            continue
        lookup[int(element_code)] = (int(alpha), int(beta))
    return lookup


def merge_payloads(order_payload: Dict, partitions_payload: Dict) -> Dict:
    partition_lookup = build_partition_lookup(partitions_payload)
    for item in order_payload.get("ordering", []):
        parent = item.get("parent", {})
        element_code = parent.get("elementCode")
        if element_code is None:
            continue
        alpha_beta = partition_lookup.get(int(element_code))
        if alpha_beta is None:
            continue
        parent["alpha"], parent["beta"] = alpha_beta

    order_metadata = order_payload.setdefault("metadata", {})
    partition_metadata = partitions_payload.get("metadata", {})
    for key in (
        "max_partition_alpha",
        "max_partition_beta",
        "max_partition",
        "max_shape_count",
        "min_trajs",
        "global_alpha",
        "global_beta",
    ):
        if key in partition_metadata:
            order_metadata[key] = partition_metadata[key]
    return order_payload


def main() -> None:
    args = parse_args()
    order_path = Path(args.order_file)
    partitions_path = Path(args.partitions_file)
    output_path = Path(args.output_file) if args.output_file else order_path

    with open(order_path, "r", encoding="utf-8") as file_obj:
        order_payload = json.load(file_obj)
    with open(partitions_path, "r", encoding="utf-8") as file_obj:
        partitions_payload = json.load(file_obj)

    merged = merge_payloads(order_payload, partitions_payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(merged, file_obj, indent=2, ensure_ascii=False)

    print(f"updated_order_file: {output_path}")


if __name__ == "__main__":
    main()
