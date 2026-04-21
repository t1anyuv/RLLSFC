import shutil
import uuid
from pathlib import Path

from scripts.experiments.generate_param_orders import build_run_config, get_existing_case_record
from src.config import TShapeConfig, ExperimentConfig, IndexConfig, DataConfig, PathConfig


def test_build_run_config_uses_experiment_private_similarity_matrix():
    base_config = TShapeConfig(
        index=IndexConfig(max_level=8, alpha=3, beta=3, min_cell_trajs=4),
        data=DataConfig(
            num_trajectories=-1,
            similarity_matrix_path="sim_mtx_L8_A3_B3_T-1.npz",
        ),
    )

    run_config = build_run_config(
        base_config=base_config,
        distribution="skewed",
        resolution=9,
        min_trajs=2,
        sweep_type="resolution",
    )

    assert run_config.data.similarity_matrix_path is not None
    assert "resource/experiments/skewed_r9_min2_a3_b3/similarity/" in (
        run_config.data.similarity_matrix_path.replace("\\", "/")
    )
    assert run_config.data.similarity_matrix_path.endswith("sim_mtx_tdrive_L9_A3_B3_T-1.npz")


def test_build_run_config_overrides_alpha_beta():
    base_config = TShapeConfig(
        index=IndexConfig(max_level=8, alpha=3, beta=3, min_cell_trajs=4),
        data=DataConfig(
            num_trajectories=-1,
            similarity_matrix_path="sim_mtx_L8_A3_B3_T-1.npz",
        ),
    )

    run_config = build_run_config(
        base_config=base_config,
        distribution="skewed",
        resolution=8,
        min_trajs=4,
        sweep_type="grid",
        alpha=2,
        beta=2,
    )

    assert run_config.index.alpha == 2
    assert run_config.index.beta == 2
    assert run_config.experiment.name == "skewed_r8_min4_a2_b2"
    assert run_config.data.similarity_matrix_path.endswith("sim_mtx_tdrive_L8_A2_B2_T-1.npz")


def test_get_existing_case_record_skips_completed_case():
    test_root = Path("tests/test_resources") / f"param_orders_{uuid.uuid4().hex}"
    try:
        config = TShapeConfig(
            experiment=ExperimentConfig(name="demo_case"),
            index=IndexConfig(max_level=8, alpha=3, beta=3, min_cell_trajs=4),
            data=DataConfig(num_trajectories=-1),
            paths=PathConfig(resource_base_dir=str(test_root)),
        )
        run_config = build_run_config(
            base_config=config,
            distribution="skewed",
            resolution=8,
            min_trajs=4,
            sweep_type="resolution",
        )
        export_prefix = "skewed_r8_min4_a3_b3"
        orders_dir = run_config.experiment.get_orders_dir()
        models_dir = run_config.experiment.get_models_dir()
        logs_dir = run_config.experiment.get_logs_dir()
        orders_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (orders_dir / f"{export_prefix}.json").write_text("{}", encoding="utf-8")
        (orders_dir / f"{export_prefix}_metadata.json").write_text("{}", encoding="utf-8")
        (models_dir / "latest.pth").write_text("stub", encoding="utf-8")
        (logs_dir / "training_summary.json").write_text("{}", encoding="utf-8")

        record = get_existing_case_record(
            config=run_config,
            distribution="skewed",
            resolution=8,
            min_trajs=4,
            sweep_type="resolution",
            order_mode="rl",
        )

        assert record is not None
        assert record["status"] == "skipped"
        assert record["order_file"].endswith(f"{export_prefix}.json")
    finally:
        if test_root.exists():
            shutil.rmtree(test_root)
