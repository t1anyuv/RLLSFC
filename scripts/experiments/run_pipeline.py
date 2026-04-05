"""统一实验运行脚本：使用LSFCPipeLine进行模型训练与顺序导出

Usage:
    python -m scripts.experiments.run_pipeline --config resource/experiments/test/config.yaml --name test
    python -m scripts.experiments.run_pipeline --config resource/experiments/formal/config.yaml --name formal
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

from src.utils.path_manager import get_path_manager
from src.config import NetworkConfig, TShapeConfig
from src.rl.pipeline import LSFCPipeLine
from src.utils.logger import setup_logging

# 获取路径管理器
path_manager = get_path_manager()


def run_rl_indexing_experiment(config_path: str, exp_name: str, nt_config: NetworkConfig):
    """执行完整的RL索引优化实验

    Args:
        config_path: YAML配置文件路径
        exp_name: 实验名称（test/formal等）
        nt_config: 神经网络配置对象
    """
    # 1. 从YAML加载配置
    ts_config = TShapeConfig.from_yaml(config_path)

    # 2. 基础环境配置
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(f"Experiment_{exp_name.capitalize()}_{timestamp}")
    logger.info(f"=== 启动{exp_name}实验 ===")

    # 应用路径配置
    ts_config.paths.apply_to_path_manager()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # 3. 验证神经网络配置
    if nt_config is None:
        nt_config = NetworkConfig(
            hidden_dims=[256, 256],
            device="auto"
        )

    # 4. 初始化流水线
    export_prefix = f"quadorder_{exp_name}_{timestamp}"

    pipeline = LSFCPipeLine(
        ts_config,
        nt_config,
        logger=logger
    )

    # 5. 执行流水线
    try:
        logger.info("步骤 1: 开始完整训练与导出流程...")
        start_time = time.time()

        results = pipeline.run_full_pipeline(export_prefix=export_prefix)

        end_time = time.time()
        duration = (end_time - start_time) / 60
        logger.info(f"流水线执行成功，耗时: {duration:.2f} 分钟")

        # 6. 验证结果并输出摘要
        logger.info("步骤 2: 验证输出结果...")
        orders_dir = Path(pipeline.resource_paths["orders"])
        models_dir = Path(pipeline.resource_paths["models"])

        # 检查关键导出文件（只检查JSON）
        check_files = [
            f"{export_prefix}.json"
        ]

        print("\n" + "=" * 50)
        print(f" 实验摘要 - {timestamp} ")
        print("-" * 50)
        print(f"模型保存路径: {models_dir}")
        print(f"学习节点总数: {results['quadorder_length']}")
        improvement_rate = results.get('improvement_rate', None)
        if improvement_rate is not None:
            print(f"改进率 (vs QuadCode): {improvement_rate:.2f}%")
        else:
            print(f"改进率 (vs QuadCode): N/A")

        print("\n导出文件状态:")
        for fname in check_files:
            fpath = orders_dir / fname
            status = "[OK]" if fpath.exists() else "[MISSING]"
            size = f"{fpath.stat().st_size / 1024:.1f} KB" if fpath.exists() else "0 KB"
            print(f"  {status} {fname} ({size})")

        # 将配置保存为JSON，方便日后回溯实验条件
        meta_path = orders_dir / f"{export_prefix}_metadata.json"
        with open(meta_path, 'w') as f:
            meta_data = {
                "timestamp": timestamp,
                "duration_min": duration,
                "config": ts_config.to_dict(),
                "results": {
                    k: (str(v) if not isinstance(v, dict) else v) for k, v in results.items()
                    if k != 'export_results'
                }
            }
            json.dump(meta_data, f, indent=4)
        print(f"\n实验元数据已保存至: {meta_path}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"实验执行过程中发生错误: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行RL索引优化实验")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML配置文件路径（如：resource/experiments/test/config.yaml）"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="实验名称（如：test/formal），用于组织输出目录"
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs=2,
        default=[256, 256],
        help="神经网络隐藏层维度（默认：256 256）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备（auto/cuda/cpu，默认：auto）"
    )

    args = parser.parse_args()

    # 神经网络配置
    network_config = NetworkConfig(
        hidden_dims=list(args.hidden_dims),
        device=args.device
    )

    # 运行实验
    run_rl_indexing_experiment(args.config, args.name, network_config)

