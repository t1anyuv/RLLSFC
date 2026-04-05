"""
查询数据集划分脚本。

将 resource/queries/ 下的原始查询文件划分为训练集、验证集和测试集，
保存到 resource/queries/<category>/ 目录下。

用法:
    python scripts/split_query_datasets.py --categories gus ske uni --train-ratio 0.6 --val-ratio 0.2
    
或者划分所有类型:
    python scripts/split_query_datasets.py --all
"""
import argparse
import random
from pathlib import Path
from typing import List, Tuple


def load_queries_from_files(category: str, queries_dir: Path) -> List[Tuple[float, float, float, float]]:
    """从原始文件加载某类别的所有查询。"""
    all_queries = []

    # 将类别代码映射到文件名前缀
    category_map = {
        'gus': 'gaussian',
        'ske': 'skewed',
        'uni': 'uniform',
        'gaussian': 'gaussian',
        'skewed': 'skewed',
        'uniform': 'uniform',
    }

    dist_type = category_map.get(category, category)

    # 加载所有范围的查询文件（新目录结构: range/<dist>/<dist>_<range>m.txt）
    for range_m in [100, 500, 1000, 1500, 2000]:
        query_file = queries_dir / "range" / dist_type / f"{dist_type}_{range_m}m.txt"

        if not query_file.exists():
            print(f"警告: 文件不存在 {query_file}")
            continue

        with open(query_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                continue

            # 兼容按换行和按分号两种格式
            queries_str = content.replace('\n', ';').split(';')
            for q_str in queries_str:
                q_str = q_str.strip()
                if not q_str:
                    continue

                try:
                    coords = [float(x.strip()) for x in q_str.split(',')]
                    if len(coords) >= 4:
                        all_queries.append((coords[0], coords[1], coords[2], coords[3]))
                except (ValueError, IndexError) as e:
                    print(f"解析查询失败 '{q_str}': {e}")
                    continue

    return all_queries


def save_queries(queries: List[Tuple[float, float, float, float]], output_path: Path) -> None:
    """保存查询到文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        content = ';'.join(f"{x1},{y1},{x2},{y2}" for x1, y1, x2, y2 in queries)
        f.write(content)

    print(f"  已保存 {len(queries)} 个查询到 {output_path}")


def split_dataset(queries: List, train_ratio: float, val_ratio: float,
                  seed: int = 42) -> Tuple[List, List, List]:
    """划分数据集。"""
    random.seed(seed)
    shuffled = queries.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_set = shuffled[:train_size]
    val_set = shuffled[train_size:train_size + val_size]
    test_set = shuffled[train_size + val_size:]

    return train_set, val_set, test_set


def split_category(category: str, queries_dir: Path, output_base: Path,
                   train_ratio: float, val_ratio: float, seed: int = 42) -> None:
    """划分单个类别的查询集。"""
    print(f"\n处理类别: {category}")

    # 加载查询
    queries = load_queries_from_files(category, queries_dir)
    print(f"  总共加载 {len(queries)} 个查询")

    if len(queries) == 0:
        print(f"  警告: 没有找到 {category} 的查询数据")
        return

    # 划分数据集
    train_set, val_set, test_set = split_dataset(queries, train_ratio, val_ratio, seed)

    print(f"  划分结果: 训练集={len(train_set)}, 验证集={len(val_set)}, 测试集={len(test_set)}")

    # 映射类别代码到目录名
    category_map = {
        'gus': 'gaussian',
        'ske': 'skewed',
        'uni': 'uniform',
    }
    output_dir = output_base / category_map.get(category, category)

    # 保存到对应目录
    save_queries(train_set, output_dir / "train.txt")
    save_queries(val_set, output_dir / "val.txt")
    save_queries(test_set, output_dir / "test.txt")
    save_queries(queries, output_dir / "all.txt")  # 同时保存完整数据集


def main():
    parser = argparse.ArgumentParser(description="划分查询数据集")
    parser.add_argument('--categories', nargs='+', default=['gus', 'ske', 'uni'],
                        help='要划分的类别代码 (gus=gaussian, ske=skewed, uni=uniform)')
    parser.add_argument('--all', action='store_true',
                        help='划分所有类别')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                        help='训练集比例 (默认: 0.6)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集比例 (默认: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--queries-dir', type=str, default='resource/queries',
                        help='查询文件目录')
    parser.add_argument('--output-dir', type=str, default='resource/queries',
                        help='输出目录')

    args = parser.parse_args()

    # 解析路径
    project_root = Path(__file__).resolve().parent.parent
    queries_dir = project_root / args.queries_dir
    output_dir = project_root / args.output_dir

    print(f"查询文件目录: {queries_dir}")
    print(f"输出目录: {output_dir}")
    print(
        f"划分比例: 训练集={args.train_ratio}, 验证集={args.val_ratio}, 测试集={1 - args.train_ratio - args.val_ratio:.1f}")

    # 确定要处理的类别
    if args.all:
        categories = ['gaussian', 'skewed', 'uniform']
    else:
        categories = args.categories

    # 处理每个类别
    for category in categories:
        split_category(category, queries_dir, output_dir,
                       args.train_ratio, args.val_ratio, args.seed)

    print("\n查询数据集划分完成!")


if __name__ == "__main__":
    main()
