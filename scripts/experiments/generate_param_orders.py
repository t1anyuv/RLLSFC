"""Batch-generate order files for resolution/minTrajs sweeps.

This script is intended for reproducing the parameter settings shown in the
latency table:
1. Sweep `resolution` while keeping `minTrajs` fixed.
2. Sweep `minTrajs` while keeping `resolution` fixed.

Defaults:
- distribution: skewed
- resolution: 8
- minTrajs: 4

Example:
    python -m scripts.experiments.generate_param_orders
    python -m scripts.experiments.generate_param_orders --distribution uniform
    python -m scripts.experiments.generate_param_orders --resolutions 6 7 8 9 10 --min-trajs 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import NetworkConfig, TShapeConfig
from src.rl.pipeline import LSFCPipeLine
from src.utils.logger import setup_logging
from src.utils.path_manager import get_path_manager


DEFAULT_DISTRIBUTION = "skewed"
DEFAULT_RESOLUTION = 8
DEFAULT_MIN_TRAJS = 4
DEFAULT_RESOLUTION_SWEEP = [6, 7, 8, 9, 10]
DEFAULT_MIN_TRAJS_SWEEP = [1, 2, 3, 4, 5]
VALID_DISTRIBUTIONS = {"skewed", "uniform", "gaussian"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量训练并导出不同 resolution / minTrajs 参数下的顺序文件"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default=DEFAULT_DISTRIBUTION,
        choices=sorted(VALID_DISTRIBUTIONS),
        help=f"查询分布类型，默认 {DEFAULT_DISTRIBUTION}",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="基础配置文件路径；未指定时自动使用 resource/experiments/{distribution}/config.yaml",
    )
    parser.add_argument(
        "--default-resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"未指定时使用的 resolution，默认 {DEFAULT_RESOLUTION}",
    )
    parser.add_argument(
        "--default-min-trajs",
        type=int,
        default=DEFAULT_MIN_TRAJS,
        help=f"未指定时使用的 minTrajs，默认 {DEFAULT_MIN_TRAJS}",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="*",
        default=DEFAULT_RESOLUTION_SWEEP,
        help="resolution 扫描列表，默认 6 7 8 9 10",
    )
    parser.add_argument(
        "--min-trajs",
        type=int,
        nargs="*",
        default=DEFAULT_MIN_TRAJS_SWEEP,
        help="minTrajs 扫描列表，默认 1 2 3 4 5",
    )
    parser.add_argument(
        "--skip-resolution-sweep",
        action="store_true",
        help="跳过 resolution 扫描",
    )
    parser.add_argument(
        "--skip-min-trajs-sweep",
        action="store_true",
        help="跳过 minTrajs 扫描",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs=2,
        default=[256, 256],
        help="网络隐藏层维度，默认 256 256",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="训练设备：auto/cuda/cpu，默认 auto",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="输出汇总目录附加标签，便于区分多次实验",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已存在的实验结果，强制重新运行所有配置",
    )
    return parser.parse_args()


def resolve_base_config(distribution: str, base_config: Optional[str]) -> Path:
    if base_config:
        return Path(base_config).resolve()
    return Path("resource") / "experiments" / distribution / "config.yaml"


def clone_config(base_config: TShapeConfig) -> TShapeConfig:
    return TShapeConfig.from_dict(base_config.to_dict())


def build_run_config(
    base_config: TShapeConfig,
    distribution: str,
    resolution: int,
    min_trajs: int,
    sweep_type: str,
) -> TShapeConfig:
    config = clone_config(base_config)
    config.reward.query_distribution_type = distribution
    config.index.max_level = resolution
    config.index.min_cell_trajs = min_trajs
    config.index.use_prune = min_trajs > 0
    config.experiment.name = f"{distribution}_{sweep_type}_res{resolution}_min{min_trajs}"
    config.experiment.description = (
        f"{sweep_type} sweep | distribution={distribution}, "
        f"resolution={resolution}, minTrajs={min_trajs}"
    )
    # 参数实验不能复用基础配置中绑定的共享相似度矩阵；
    # 每个参数组合都应写入自己的实验目录。
    config.data.similarity_matrix_path = str(config.get_experiment_similarity_matrix_path())
    return config


def ensure_output_dir(distribution: str, tag: Optional[str]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{distribution}_{timestamp}"
    if tag:
        dir_name = f"{dir_name}_{tag}"
    output_dir = Path("resource") / "experiments" / "param_orders" / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_export_prefix(distribution: str, sweep_type: str, resolution: int, min_trajs: int) -> str:
    return f"quadorder_{distribution}_{sweep_type}_res{resolution}_min{min_trajs}"


def get_existing_case_record(
    config: TShapeConfig,
    distribution: str,
    resolution: int,
    min_trajs: int,
    sweep_type: str,
) -> Optional[Dict[str, Any]]:
    export_prefix = make_export_prefix(distribution, sweep_type, resolution, min_trajs)
    order_path = config.experiment.get_orders_dir() / f"{export_prefix}.json"
    metadata_path = config.experiment.get_orders_dir() / f"{export_prefix}_metadata.json"
    latest_model_path = config.experiment.get_models_dir() / "latest.pth"
    best_model_record_path = config.experiment.get_orders_dir() / "best_model_record.json"
    training_summary_path = config.experiment.get_logs_dir() / "training_summary.json"

    has_model = latest_model_path.exists() or best_model_record_path.exists()
    has_outputs = order_path.exists() and metadata_path.exists() and training_summary_path.exists() and has_model
    if not has_outputs:
        return None

    return {
        "status": "skipped",
        "sweep_type": sweep_type,
        "distribution": distribution,
        "resolution": resolution,
        "min_trajs": min_trajs,
        "experiment_name": config.experiment.name,
        "export_prefix": export_prefix,
        "started_at": None,
        "ended_at": datetime.now().isoformat(),
        "train_time_seconds": None,
        "train_time_minutes": None,
        "order_file": str(order_path),
        "model_path": str(latest_model_path) if latest_model_path.exists() else None,
        "quadorder_length": None,
        "improvement_rate": None,
        "summary_report": f"Skipped existing case: {config.experiment.name}",
        "quadtree_stats": None,
        "error": None,
    }


def run_single_case(
    base_config: TShapeConfig,
    network_config: NetworkConfig,
    distribution: str,
    resolution: int,
    min_trajs: int,
    sweep_type: str,
) -> Dict[str, Any]:
    config = build_run_config(base_config, distribution, resolution, min_trajs, sweep_type)
    config.paths.apply_to_path_manager()
    get_path_manager().set_experiment_name(config.experiment.name)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    export_prefix = make_export_prefix(distribution, sweep_type, resolution, min_trajs)
    logger = setup_logging(f"ParamSweep_{distribution}_{sweep_type}_R{resolution}_M{min_trajs}")
    pipeline = LSFCPipeLine(config, network_config, logger=logger)

    started_at = datetime.now().isoformat()
    train_start = time.perf_counter()
    results = pipeline.run_full_pipeline(export_prefix=export_prefix)
    elapsed_seconds = time.perf_counter() - train_start

    order_path = results.get("export_results", {}).get("json")
    return {
        "status": "success",
        "sweep_type": sweep_type,
        "distribution": distribution,
        "resolution": resolution,
        "min_trajs": min_trajs,
        "experiment_name": config.experiment.name,
        "export_prefix": export_prefix,
        "started_at": started_at,
        "ended_at": datetime.now().isoformat(),
        "train_time_seconds": round(elapsed_seconds, 3),
        "train_time_minutes": round(elapsed_seconds / 60.0, 3),
        "order_file": str(order_path) if order_path else None,
        "model_path": results.get("model_path"),
        "quadorder_length": results.get("quadorder_length"),
        "improvement_rate": results.get("improvement_rate"),
        "summary_report": results.get("summary_report"),
        "quadtree_stats": results.get("quadtree_stats"),
    }


def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return

    fieldnames = [
        "status",
        "sweep_type",
        "distribution",
        "resolution",
        "min_trajs",
        "experiment_name",
        "export_prefix",
        "started_at",
        "ended_at",
        "train_time_seconds",
        "train_time_minutes",
        "order_file",
        "model_path",
        "quadorder_length",
        "improvement_rate",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def append_case(
    records: List[Dict[str, Any]],
    output_dir: Path,
    *,
    base_config_path: Path,
    distribution: str,
    sweep_type: str,
    resolution: int,
    min_trajs: int,
    network_config: NetworkConfig,
    base_config: TShapeConfig,
    force: bool,
) -> None:
    run_config = build_run_config(base_config, distribution, resolution, min_trajs, sweep_type)
    if not force:
        existing_record = get_existing_case_record(
            config=run_config,
            distribution=distribution,
            resolution=resolution,
            min_trajs=min_trajs,
            sweep_type=sweep_type,
        )
        if existing_record is not None:
            print(
                f"\n[SKIP] sweep={sweep_type}, distribution={distribution}, "
                f"resolution={resolution}, minTrajs={min_trajs} | "
                f"experiment={run_config.experiment.name}"
            )
            records.append(existing_record)
            save_json(
                output_dir / "summary.json",
                {
                    "base_config": str(base_config_path),
                    "distribution": distribution,
                    "records": records,
                },
            )
            save_csv(output_dir / "summary.csv", records)
            return

    print(
        f"\n[RUN] sweep={sweep_type}, distribution={distribution}, "
        f"resolution={resolution}, minTrajs={min_trajs}"
    )
    try:
        record = run_single_case(
            base_config=base_config,
            network_config=network_config,
            distribution=distribution,
            resolution=resolution,
            min_trajs=min_trajs,
            sweep_type=sweep_type,
        )
        print(
            f"[OK] 用时 {record['train_time_minutes']:.3f} min | "
            f"order={record['order_file']}"
        )
    except Exception as exc:
        traceback.print_exc()
        record = {
            "status": "failed",
            "sweep_type": sweep_type,
            "distribution": distribution,
            "resolution": resolution,
            "min_trajs": min_trajs,
            "experiment_name": f"{distribution}_{sweep_type}_res{resolution}_min{min_trajs}",
            "export_prefix": make_export_prefix(distribution, sweep_type, resolution, min_trajs),
            "started_at": None,
            "ended_at": datetime.now().isoformat(),
            "train_time_seconds": None,
            "train_time_minutes": None,
            "order_file": None,
            "model_path": None,
            "quadorder_length": None,
            "improvement_rate": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[FAIL] {exc}")

    records.append(record)
    save_json(
        output_dir / "summary.json",
        {
            "base_config": str(base_config_path),
            "distribution": distribution,
            "records": records,
        },
    )
    save_csv(output_dir / "summary.csv", records)


def main() -> None:
    args = parse_args()
    base_config_path = resolve_base_config(args.distribution, args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"基础配置文件不存在: {base_config_path}")

    base_config = TShapeConfig.from_yaml(str(base_config_path))
    network_config = NetworkConfig(
        hidden_dims=list(args.hidden_dims),
        device=args.device,
        dropout=base_config.network.dropout,
        state_dim=base_config.network.state_dim,
    )
    output_dir = ensure_output_dir(args.distribution, args.tag)

    records: List[Dict[str, Any]] = []
    manifest = {
        "created_at": datetime.now().isoformat(),
        "base_config": str(base_config_path),
        "distribution": args.distribution,
        "default_resolution": args.default_resolution,
        "default_min_trajs": args.default_min_trajs,
        "resolution_sweep": [] if args.skip_resolution_sweep else list(args.resolutions),
        "min_trajs_sweep": [] if args.skip_min_trajs_sweep else list(args.min_trajs),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "manifest.json", manifest)

    if not args.skip_resolution_sweep:
        for resolution in args.resolutions:
            append_case(
                records,
                output_dir,
                base_config_path=base_config_path,
                distribution=args.distribution,
                sweep_type="resolution",
                resolution=resolution,
                min_trajs=args.default_min_trajs,
                network_config=network_config,
                base_config=base_config,
                force=args.force,
            )

    if not args.skip_min_trajs_sweep:
        for min_trajs in args.min_trajs:
            append_case(
                records,
                output_dir,
                base_config_path=base_config_path,
                distribution=args.distribution,
                sweep_type="min_trajs",
                resolution=args.default_resolution,
                min_trajs=min_trajs,
                network_config=network_config,
                base_config=base_config,
                force=args.force,
            )

    successful = sum(1 for record in records if record["status"] == "success")
    skipped = sum(1 for record in records if record["status"] == "skipped")
    failed = len(records) - successful - skipped
    print("\n" + "=" * 72)
    print(f"输出目录: {output_dir}")
    print(f"成功: {successful} | 跳过: {skipped} | 失败: {failed} | 总计: {len(records)}")
    print(f"汇总 JSON: {output_dir / 'summary.json'}")
    print(f"汇总 CSV : {output_dir / 'summary.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
