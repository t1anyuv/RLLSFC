import argparse
from pathlib import Path

from src.config import TShapeConfig
from src.resource import get_similarity_matrix_path
from src.training import TraversalTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="预计算相似度矩阵")

    # 核心参数：直接透传给 TShapeConfig
    parser.add_argument("--max-level", type=int, default=9, help="四叉树最大层级")
    parser.add_argument("--min-cell-trajs", type=int, default=3, help="剪枝阈值")
    parser.add_argument("--num-trajectories", type=int, default=-1, help="轨迹数量 (-1 为全部)")
    parser.add_argument("--use-tdrive-data", action="store_true", default=True, help="是否使用 TDrive 数据")

    # 计算资源参数
    parser.add_argument("--num-workers", type=int, default=None, help="并行工作进程数")
    parser.add_argument("--output-file", type=str, default=None, help="手动指定输出路径")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 构造配置对象
    from src.config import IndexConfig, DataConfig
    config = TShapeConfig(
        index=IndexConfig(
            max_level=args.max_level,
            min_cell_trajs=args.min_cell_trajs,
            use_prune=(args.min_cell_trajs is not None)
        ),
        data=DataConfig(
            num_trajectories=args.num_trajectories,
            use_tdrive_data=args.use_tdrive_data
        )
    )

    print("=" * 60)
    print(f"🚀 启动预计算流程 | 目标层级: L{config.index.max_level} | 剪枝阈值: M{config.index.min_cell_trajs}")
    print("=" * 60)

    # 2. 实例化并调用现成的流程
    trainer = TraversalTrainer(config)

    # 调用setup()自动完成：创建四叉树 -> 分配轨迹 -> 剪枝 -> 组件配置 -> 环境构建
    env, quadtree = trainer.setup()

    # 3. 执行相似度矩阵计算
    print(f"\n💡 环境构建完成。有效单元格数量: {len(env.all_cells)}")
    print("--- 开始计算相似度矩阵 ---")

    sim_matrix = trainer.similarity_matrix
    if sim_matrix is None:
        from src.utils.similarity_matrix import SimilarityMatrix
        sim_matrix = SimilarityMatrix(quadtree, trainer.cost_evaluator)

    sim_matrix.compute(
        env.all_cells,
        use_symmetric=True,
        show_progress=True,
        num_workers=args.num_workers
    )

    # 4. 保存与统计
    save_path = args.output_file or str(get_similarity_matrix_path())
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    sim_matrix.save(save_path)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " 相似度矩阵预计算完成统计 ".center(50) + "║")
    print("╚" + "═" * 58 + "╝")

    stats = sim_matrix.get_statistics()
    print(f"  - 计算单元格总数: {stats.get('matrix_size', len(env.all_cells))}")
    print(f"  - 平均相似度分数: {stats.get('mean_similarity', 0.0):.4f}")
    print(f"  - 保存路径: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
