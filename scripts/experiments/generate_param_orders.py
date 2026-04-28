"""Batch-generate RL or XZ order files for resolution/minTrajs sweeps."""

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
from src.indexing import TraversalOrderEncoder
from src.rl.order_formatter import TrajectoryOrderFormatter
from src.rl.pipeline import LSFCPipeLine
from src.training.component_factory import TrainingComponentFactory
from src.utils.logger import setup_logging


DEFAULT_DISTRIBUTION = "skewed"
DEFAULT_RESOLUTION = 8
DEFAULT_MIN_TRAJS = 4
DEFAULT_RESOLUTION_SWEEP = [6, 7, 8, 9, 10]
DEFAULT_MIN_TRAJS_SWEEP = [2, 4, 6, 8, 10]
VALID_DISTRIBUTIONS = {"skewed", "uniform", "gaussian"}
VALID_DATASETS = {"tdrive", "cdtaxi", "cd_taxi"}


def make_case_name(distribution: str, resolution: int, min_trajs: int, alpha: int, beta: int) -> str:
    return f"{distribution}_r{resolution}_min{min_trajs}_a{alpha}_b{beta}"


def normalize_dataset_name(dataset: str) -> str:
    return "cdtaxi" if dataset == "cd_taxi" else dataset


def dataset_dir_name(dataset: str) -> str:
    return "cd_taxi" if normalize_dataset_name(dataset) == "cdtaxi" else "tdrive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate order files for resolution/minTrajs sweeps."
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default=DEFAULT_DISTRIBUTION,
        choices=sorted(VALID_DISTRIBUTIONS),
        help=f"query distribution, default: {DEFAULT_DISTRIBUTION}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tdrive",
        choices=sorted(VALID_DATASETS),
        help="dataset key, default: tdrive",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="base config path; defaults to configs/order_sweeps/{dataset}/{distribution}/config.yaml",
    )
    parser.add_argument(
        "--default-resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"default resolution, default: {DEFAULT_RESOLUTION}",
    )
    parser.add_argument(
        "--default-min-trajs",
        type=int,
        default=DEFAULT_MIN_TRAJS,
        help=f"default minTrajs, default: {DEFAULT_MIN_TRAJS}",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="*",
        default=DEFAULT_RESOLUTION_SWEEP,
        help="resolution sweep list",
    )
    parser.add_argument(
        "--min-trajs",
        type=int,
        nargs="*",
        default=DEFAULT_MIN_TRAJS_SWEEP,
        help="minTrajs sweep list",
    )
    parser.add_argument(
        "--skip-resolution-sweep",
        action="store_true",
        help="skip resolution sweep",
    )
    parser.add_argument(
        "--skip-min-trajs-sweep",
        action="store_true",
        help="skip minTrajs sweep",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="generate the full Cartesian product of resolutions x minTrajs",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs=2,
        default=[256, 256],
        help="network hidden dims",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="training device",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="optional output tag",
    )
    parser.add_argument(
        "--order-mode",
        type=str,
        choices=["rl", "xz"],
        default="xz",
        help="order generation mode",
    )
    parser.add_argument(
        "--resource-dir",
        type=str,
        default=None,
        help="optional resource directory for shared inputs and generated order artifacts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even if outputs already exist",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        choices=[2, 3],
        default=None,
        help="global EE alpha size; only 2 or 3 are supported",
    )
    parser.add_argument(
        "--beta",
        type=int,
        choices=[2, 3],
        default=None,
        help="global EE beta size; only 2 or 3 are supported",
    )
    args = parser.parse_args()
    if (args.alpha is None) != (args.beta is None):
        parser.error("--alpha and --beta must be provided together")
    if args.alpha is not None and args.alpha != args.beta:
        parser.error("only 2x2 or 3x3 EE sizes are supported, so --alpha must equal --beta")
    return args


def resolve_base_config(distribution: str, dataset: str, base_config: Optional[str]) -> Path:
    if base_config:
        return Path(base_config).resolve()
    dataset_dir = dataset_dir_name(dataset)
    return Path("configs") / "order_sweeps" / dataset_dir / distribution / "config.yaml"


def clone_config(base_config: TShapeConfig) -> TShapeConfig:
    return TShapeConfig.from_dict(base_config.to_dict())


def build_run_config(
    base_config: TShapeConfig,
    distribution: str,
    resolution: int,
    min_trajs: int,
    sweep_type: str,
    resource_dir: Optional[str] = None,
    alpha: Optional[int] = None,
    beta: Optional[int] = None,
) -> TShapeConfig:
    config = clone_config(base_config)
    config.datasets.active = normalize_dataset_name(config.datasets.active)
    config.reward.query_distribution_type = distribution
    config.datasets.active = normalize_dataset_name(config.datasets.active)
    config.index.max_level = resolution
    config.index.min_cell_trajs = min_trajs
    config.index.use_prune = min_trajs > 0
    if alpha is not None:
        config.index.alpha = alpha
    if beta is not None:
        config.index.beta = beta
    config.experiment.name = make_case_name(
        distribution,
        resolution,
        min_trajs,
        config.index.alpha,
        config.index.beta,
    )
    if sweep_type == "grid":
        config.experiment.description = (
            f"full parameter sweep | distribution={distribution}, "
            f"resolution={resolution}, minTrajs={min_trajs}, "
            f"alpha={config.index.alpha}, beta={config.index.beta}"
        )
    else:
        config.experiment.description = (
            f"{sweep_type} sweep | distribution={distribution}, "
            f"resolution={resolution}, minTrajs={min_trajs}, "
            f"alpha={config.index.alpha}, beta={config.index.beta}"
        )
    if resource_dir:
        config.paths.resource_dir = resource_dir
    config.data.similarity_matrix_path = str(config.get_experiment_similarity_matrix_path())
    return config


def ensure_output_dir(distribution: str, tag: Optional[str], resource_dir: Optional[str] = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{distribution}_{timestamp}"
    if tag:
        dir_name = f"{dir_name}_{tag}"
    del resource_dir
    output_base = Path(os.environ.get("OUTPUT_DIR", "outputs"))
    output_root = output_base / "experiments" / "param_orders"
    output_dir = output_root / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_case_output_dir(config: TShapeConfig, distribution: str, order_mode: str) -> Path:
    if order_mode == "xz":
        dataset_dir = dataset_dir_name(config.datasets.active)
        return config.paths.orders_dir / dataset_dir / distribution
    return config.experiment.get_results_dir(config.paths)


def make_export_prefix(
    distribution: str,
    sweep_type: str,
    resolution: int,
    min_trajs: int,
    alpha: int,
    beta: int,
) -> str:
    _ = sweep_type
    return make_case_name(distribution, resolution, min_trajs, alpha, beta)


def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)


def get_existing_case_record(
    config: TShapeConfig,
    distribution: str,
    resolution: int,
    min_trajs: int,
    sweep_type: str,
    order_mode: str,
) -> Optional[Dict[str, Any]]:
    export_prefix = make_export_prefix(
        distribution,
        sweep_type,
        resolution,
        min_trajs,
        config.index.alpha,
        config.index.beta,
    )
    order_dir = get_case_output_dir(config, distribution, order_mode)
    order_path = order_dir / f"{export_prefix}.json"
    metadata_path = order_dir / f"{export_prefix}_metadata.json"

    if order_mode == "xz":
        has_outputs = order_path.exists() and metadata_path.exists()
        model_path = None
    else:
        latest_model_path = config.experiment.get_checkpoints_dir(config.paths) / "latest.pth"
        best_model_record_path = config.experiment.get_results_dir(config.paths) / "best_model_record.json"
        training_summary_path = config.experiment.get_logs_dir(config.paths) / "training_summary.json"
        has_model = latest_model_path.exists() or best_model_record_path.exists()
        has_outputs = order_path.exists() and metadata_path.exists() and training_summary_path.exists() and has_model
        model_path = str(latest_model_path) if latest_model_path.exists() else None

    if not has_outputs:
        return None

    return {
        "status": "skipped",
        "order_mode": order_mode,
        "sweep_type": sweep_type,
        "distribution": distribution,
        "resolution": resolution,
        "min_trajs": min_trajs,
        "alpha": config.index.alpha,
        "beta": config.index.beta,
        "experiment_name": config.experiment.name,
        "export_prefix": export_prefix,
        "started_at": None,
        "ended_at": datetime.now().isoformat(),
        "train_time_seconds": None,
        "train_time_minutes": None,
        "order_file": str(order_path),
        "model_path": model_path,
        "quadorder_length": None,
        "improvement_rate": None,
        "summary_report": f"Skipped existing case: {config.experiment.name}",
        "quadtree_stats": None,
        "error": None,
    }


def run_single_case_xz(
    config: TShapeConfig,
    export_prefix: str,
) -> Dict[str, Any]:
    factory = TrainingComponentFactory(config)
    quadtree = factory.create_quadtree()
    trajectories = factory.load_trajectories(quadtree)
    for trajectory_id, points in trajectories:
        quadtree.assign_trajectory(trajectory_id, points)

    if config.index.use_prune and config.index.min_cell_trajs is not None:
        quadtree.post_prune_tree(config.index.min_cell_trajs)
    quadtree.compute_signatures(config.index.enable_sig_optimize)

    encoder = TraversalOrderEncoder(
        quadtree,
        alpha=config.index.alpha,
        beta=config.index.beta,
    )
    xz_order = encoder.z_curve_order(include_muted=False)
    orders_dir = get_case_output_dir(config, config.reward.query_distribution_type, "xz")
    orders_dir.mkdir(parents=True, exist_ok=True)
    formatter = TrajectoryOrderFormatter(output_dir=str(orders_dir), config=config)
    _, order_path = formatter.generate_config_file_from_order(
        order=xz_order,
        quadtree=quadtree,
        filename=f"{export_prefix}.json",
        global_alpha=config.index.alpha,
        global_beta=config.index.beta,
        order_source="pruned_default_xz_order",
    )

    metadata_path = orders_dir / f"{export_prefix}_metadata.json"
    save_json(
        metadata_path,
        {
            "timestamp": datetime.now().isoformat(),
            "config": config.to_dict(),
            "results": {
                "order_mode": "xz",
                "quadorder_length": len(xz_order),
                "quadtree_stats": quadtree.get_quadtree_stats(),
            },
        },
    )

    return {
        "export_results": {"json": str(order_path)},
        "model_path": None,
        "quadorder_length": len(xz_order),
        "improvement_rate": None,
        "summary_report": f"Generated XZ order for {config.experiment.name}",
        "quadtree_stats": quadtree.get_quadtree_stats(),
    }


def run_single_case(
    base_config: TShapeConfig,
    network_config: NetworkConfig,
    distribution: str,
    resolution: int,
    min_trajs: int,
    sweep_type: str,
    order_mode: str,
    resource_dir: Optional[str],
    alpha: Optional[int],
    beta: Optional[int],
) -> Dict[str, Any]:
    config = build_run_config(
        base_config,
        distribution,
        resolution,
        min_trajs,
        sweep_type,
        resource_dir=resource_dir,
        alpha=alpha,
        beta=beta,
    )
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    export_prefix = make_export_prefix(
        distribution,
        sweep_type,
        resolution,
        min_trajs,
        config.index.alpha,
        config.index.beta,
    )
    started_at = datetime.now().isoformat()
    train_start = time.perf_counter()

    if order_mode == "xz":
        results = run_single_case_xz(config, export_prefix)
    else:
        logger = setup_logging(f"ParamSweep_{distribution}_{sweep_type}_R{resolution}_M{min_trajs}")
        pipeline = LSFCPipeLine(config, network_config, logger=logger)
        results = pipeline.run_full_pipeline(export_prefix=export_prefix)

    elapsed_seconds = time.perf_counter() - train_start
    order_path = results.get("export_results", {}).get("json")
    return {
        "status": "success",
        "order_mode": order_mode,
        "sweep_type": sweep_type,
        "distribution": distribution,
        "resolution": resolution,
        "min_trajs": min_trajs,
        "alpha": config.index.alpha,
        "beta": config.index.beta,
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
        "error": None,
    }


def save_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return

    fieldnames = [
        "status",
        "order_mode",
        "sweep_type",
        "distribution",
        "resolution",
        "min_trajs",
        "alpha",
        "beta",
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
    order_mode: str,
    resource_dir: Optional[str],
    alpha: Optional[int],
    beta: Optional[int],
) -> None:
    run_config = build_run_config(
        base_config,
        distribution,
        resolution,
        min_trajs,
        sweep_type,
        resource_dir=resource_dir,
        alpha=alpha,
        beta=beta,
    )
    if not force:
        existing_record = get_existing_case_record(
            config=run_config,
            distribution=distribution,
            resolution=resolution,
            min_trajs=min_trajs,
            sweep_type=sweep_type,
            order_mode=order_mode,
        )
        if existing_record is not None:
            print(
                f"\n[SKIP] mode={order_mode}, sweep={sweep_type}, distribution={distribution}, "
                f"resolution={resolution}, minTrajs={min_trajs}, "
                f"alpha={run_config.index.alpha}, beta={run_config.index.beta} | "
                f"experiment={run_config.experiment.name}"
            )
            records.append(existing_record)
            save_json(
                output_dir / "summary.json",
                {
                    "base_config": str(base_config_path),
                    "distribution": distribution,
                    "order_mode": order_mode,
                    "records": records,
                },
            )
            save_csv(output_dir / "summary.csv", records)
            return

    print(
        f"\n[RUN] mode={order_mode}, sweep={sweep_type}, distribution={distribution}, "
        f"resolution={resolution}, minTrajs={min_trajs}, alpha={run_config.index.alpha}, beta={run_config.index.beta}"
    )
    try:
        record = run_single_case(
            base_config=base_config,
            network_config=network_config,
            distribution=distribution,
            resolution=resolution,
            min_trajs=min_trajs,
            sweep_type=sweep_type,
            order_mode=order_mode,
            resource_dir=resource_dir,
            alpha=alpha,
            beta=beta,
        )
        print(
            f"[OK] used {record['train_time_minutes']:.3f} min | "
            f"order={record['order_file']}"
        )
    except Exception as exc:
        traceback.print_exc()
        record = {
            "status": "failed",
            "order_mode": order_mode,
            "sweep_type": sweep_type,
            "distribution": distribution,
            "resolution": resolution,
            "min_trajs": min_trajs,
            "alpha": run_config.index.alpha,
            "beta": run_config.index.beta,
            "experiment_name": (
                make_case_name(
                    distribution,
                    resolution,
                    min_trajs,
                    run_config.index.alpha,
                    run_config.index.beta,
                )
            ),
            "export_prefix": make_export_prefix(
                distribution,
                sweep_type,
                resolution,
                min_trajs,
                run_config.index.alpha,
                run_config.index.beta,
            ),
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
            "order_mode": order_mode,
            "records": records,
        },
    )
    save_csv(output_dir / "summary.csv", records)


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    base_config_path = resolve_base_config(args.distribution, args.dataset, args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    base_config = TShapeConfig.from_yaml(str(base_config_path))
    network_config = NetworkConfig(
        hidden_dims=list(args.hidden_dims),
        device=args.device,
        dropout=base_config.network.dropout,
        state_dim=base_config.network.state_dim,
    )
    output_dir = ensure_output_dir(args.distribution, args.tag, args.resource_dir)

    records: List[Dict[str, Any]] = []
    manifest = {
        "created_at": datetime.now().isoformat(),
        "base_config": str(base_config_path),
        "distribution": args.distribution,
        "order_mode": args.order_mode,
        "full_grid": args.full_grid,
        "default_resolution": args.default_resolution,
        "default_min_trajs": args.default_min_trajs,
        "alpha": args.alpha if args.alpha is not None else base_config.index.alpha,
        "beta": args.beta if args.beta is not None else base_config.index.beta,
        "resolution_sweep": [] if args.skip_resolution_sweep else list(args.resolutions),
        "min_trajs_sweep": [] if args.skip_min_trajs_sweep else list(args.min_trajs),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "manifest.json", manifest)

    if args.full_grid:
        for resolution in args.resolutions:
            for min_trajs in args.min_trajs:
                append_case(
                    records,
                    output_dir,
                    base_config_path=base_config_path,
                    distribution=args.distribution,
                    sweep_type="grid",
                    resolution=resolution,
                    min_trajs=min_trajs,
                    network_config=network_config,
                    base_config=base_config,
                    force=args.force,
                    order_mode=args.order_mode,
                    resource_dir=args.resource_dir,
                    alpha=args.alpha,
                    beta=args.beta,
                )
    else:
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
                    order_mode=args.order_mode,
                    resource_dir=args.resource_dir,
                    alpha=args.alpha,
                    beta=args.beta,
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
                    order_mode=args.order_mode,
                    resource_dir=args.resource_dir,
                    alpha=args.alpha,
                    beta=args.beta,
                )

    successful = sum(1 for record in records if record["status"] == "success")
    skipped = sum(1 for record in records if record["status"] == "skipped")
    failed = len(records) - successful - skipped
    print("\n" + "=" * 72)
    print(f"Output dir: {output_dir}")
    print(f"Success: {successful} | Skipped: {skipped} | Failed: {failed} | Total: {len(records)}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    print(f"Summary CSV : {output_dir / 'summary.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
