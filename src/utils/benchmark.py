"""性能基准测试工具

用于测量和比较不同加速策略的性能表现。
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    elapsed_time: float
    throughput: float  # items/second
    memory_peak_mb: float = 0.0
    device: str = "cpu"
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return (f"{self.name}: {self.elapsed_time:.3f}s, "
                f"{self.throughput:.1f} items/s, "
                f"device={self.device}")


class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def measure(
        self,
        name: str,
        num_items: int,
        device: str = "cpu"
    ):
        """上下文管理器测量代码块执行时间。
        
        使用示例:
            benchmark = PerformanceBenchmark()
            with benchmark.measure("矩阵计算", n=10000):
                result = compute_large_matrix()
        """
        start_time = time.perf_counter()
        
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            throughput = num_items / elapsed if elapsed > 0 else 0
            
            memory_peak = 0.0
            if device == "cuda" and torch.cuda.is_available():
                memory_peak = torch.cuda.max_memory_allocated() / 1024**2
            
            result = BenchmarkResult(
                name=name,
                elapsed_time=elapsed,
                throughput=throughput,
                memory_peak_mb=memory_peak,
                device=device
            )
            self.results.append(result)
            logger.info(f"[Benchmark] {result}")
    
    def compare_gpu_cpu(
        self,
        func: Callable,
        args_cpu: tuple,
        args_gpu: tuple,
        num_items: int,
        warmup: int = 3,
        repeats: int = 5
    ) -> Dict[str, float]:
        """比较GPU和CPU执行性能。
        
        参数:
            func: 测试函数，接收设备参数
            args_cpu: CPU执行参数
            args_gpu: GPU执行参数
            num_items: 处理的数据量
            warmup: 预热次数
            repeats: 正式测试次数
            
        返回:
            包含加速比等信息的字典
        """
        # 预热
        for _ in range(warmup):
            func(*args_cpu, device="cpu")
            if torch.cuda.is_available():
                func(*args_gpu, device="cuda")
        
        # CPU测试
        cpu_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            func(*args_cpu, device="cpu")
            cpu_times.append(time.perf_counter() - start)
        
        cpu_time = np.median(cpu_times)
        
        # GPU测试
        gpu_time = None
        if torch.cuda.is_available():
            gpu_times = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                func(*args_gpu, device="cuda")
                torch.cuda.synchronize()
                gpu_times.append(time.perf_counter() - start)
            
            gpu_time = np.median(gpu_times)
        
        # 记录结果
        self.results.append(BenchmarkResult(
            name="CPU",
            elapsed_time=cpu_time,
            throughput=num_items / cpu_time,
            device="cpu"
        ))
        
        speedup = None
        if gpu_time:
            self.results.append(BenchmarkResult(
                name="GPU",
                elapsed_time=gpu_time,
                throughput=num_items / gpu_time,
                device="cuda"
            ))
            speedup = cpu_time / gpu_time
        
        return {
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": speedup,
            "num_items": num_items
        }
    
    def print_summary(self):
        """打印所有基准测试结果摘要。"""
        print("\n" + "="*60)
        print("性能基准测试摘要")
        print("="*60)
        
        for result in self.results:
            print(f"\n{result.name}:")
            print(f"  执行时间: {result.elapsed_time:.4f} s")
            print(f"  吞吐量: {result.throughput:.2f} items/s")
            print(f"  设备: {result.device}")
            if result.memory_peak_mb > 0:
                print(f"  峰值显存: {result.memory_peak_mb:.1f} MB")
        
        # 自动检测GPU加速场景
        cpu_results = {r.name: r for r in self.results if r.device == "cpu"}
        gpu_results = {r.name: r for r in self.results if r.device == "cuda"}
        
        if cpu_results and gpu_results:
            print("\n" + "-"*60)
            print("GPU加速对比:")
            for name in set(cpu_results.keys()) & set(gpu_results.keys()):
                cpu_t = cpu_results[name].elapsed_time
                gpu_t = gpu_results[name].elapsed_time
                speedup = cpu_t / gpu_t
                print(f"  {name}: {speedup:.2f}x 加速")
        
        print("="*60 + "\n")


def benchmark_similarity_matrix(
    num_cells: int = 1000,
    use_gpu: bool = True
) -> Dict[str, float]:
    """基准测试相似度矩阵计算性能。
    
    参数:
        num_cells: 测试用的单元格数量
        use_gpu: 是否测试GPU版本
        
    返回:
        性能统计字典
    """
    from src.utils.gpu_accelerator import GPUSimilarityCalculator
    
    print(f"\n相似度矩阵计算基准测试 (cells={num_cells})")
    print("-" * 50)
    
    # 创建模拟数据
    class MockCell:
        def __init__(self):
            self.signatures = {
                i: np.random.rand(100) > 0.5  # 模拟签名
                for i in range(50)
            }
    
    cells = [MockCell() for _ in range(num_cells)]
    
    class MockCostEvaluator:
        def jaccard_similarity(self, a, b):
            # 简化的Jaccard计算
            return np.random.rand()
    
    evaluator = MockCostEvaluator()
    
    benchmark = PerformanceBenchmark()
    
    # CPU测试
    with benchmark.measure("相似度矩阵-CPU", num_items=num_cells*num_cells):
        calculator_cpu = GPUSimilarityCalculator(device=torch.device('cpu'))
        matrix_cpu = calculator_cpu._compute_cpu(cells, evaluator, symmetric=True)
    
    # GPU测试
    if use_gpu and torch.cuda.is_available():
        with benchmark.measure("相似度矩阵-GPU", num_items=num_cells*num_cells, device="cuda"):
            calculator_gpu = GPUSimilarityCalculator(device=torch.device('cuda'))
            matrix_gpu = calculator_gpu.compute_similarity_matrix(
                cells, evaluator, batch_size=256, symmetric=True
            )
    
    benchmark.print_summary()
    
    return {
        "num_cells": num_cells,
        "matrix_size": num_cells * num_cells
    }


def benchmark_trajectory_assignment(
    num_trajectories: int = 10000,
    num_workers_list: List[int] = [1, 2, 4, 8]
) -> Dict[str, float]:
    """基准测试轨迹分配并行性能。
    
    参数:
        num_trajectories: 测试轨迹数量
        num_workers_list: 测试的工作进程数列表
        
    返回:
        性能统计字典
    """
    print(f"\n轨迹分配并行基准测试 (trajectories={num_trajectories})")
    print("-" * 50)
    
    # 创建模拟轨迹数据
    trajectories = [
        (i, [(np.random.rand(), np.random.rand()) for _ in range(20)])
        for i in range(num_trajectories)
    ]
    
    benchmark = PerformanceBenchmark()
    
    # 模拟处理函数
    def process_trajectory(traj):
        tid, points = traj
        return sum(p[0] + p[1] for p in points)
    
    for workers in num_workers_list:
        with benchmark.measure(f"轨迹处理-{workers}workers", num_items=num_trajectories):
            if workers == 1:
                # 单线程
                results = [process_trajectory(t) for t in trajectories]
            else:
                # 多线程
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    results = list(executor.map(process_trajectory, trajectories))
    
    benchmark.print_summary()
    
    return {
        "num_trajectories": num_trajectories,
        "optimal_workers": num_workers_list[
            np.argmin([r.elapsed_time for r in benchmark.results])
        ] if benchmark.results else 1
    }


def run_full_benchmark_suite():
    """运行完整基准测试套件。"""
    print("\n" + "="*70)
    print("TShape训练系统性能基准测试套件")
    print("="*70)
    
    # 检测硬件
    print("\n硬件检测:")
    print(f"  CPU核心数: {torch.multiprocessing.cpu_count()}")
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}")
        print(f"  GPU显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"  CUDA版本: {torch.version.cuda}")
    else:
        print("  GPU: 不可用")
    
    # 运行各项基准测试
    similarity_stats = benchmark_similarity_matrix(num_cells=500)
    assignment_stats = benchmark_trajectory_assignment(num_trajectories=5000)
    
    print("\n" + "="*70)
    print("基准测试完成!")
    print(f"  - 相似度矩阵规模: {similarity_stats['matrix_size']} 元素")
    print(f"  - 轨迹处理规模: {assignment_stats['num_trajectories']} 条轨迹")
    print(f"  - 推荐并行度: {assignment_stats['optimal_workers']} workers")
    print("="*70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_benchmark_suite()
