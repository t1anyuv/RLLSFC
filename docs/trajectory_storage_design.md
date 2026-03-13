# 双模式轨迹存储架构设计文档

## 1. 设计目标

支持两种轨迹存储模式，系统可动态切换：
- **内存模式 (In-Memory)**：全量加载，适合小数据集（< 5GB），访问速度快
- **磁盘模式 (Disk-Based)**：磁盘存储 + LRU缓存，适合大数据集（30GB+），内存占用可控

## 2. 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层 (Application)                       │
│                    QuadTreeIndex / Trainer                       │
└─────────────────────────────────────────────────────────────────┘
                              ↕ 统一接口调用
┌─────────────────────────────────────────────────────────────────┐
│                     存储抽象层 (Storage Layer)                     │
│  ┌──────────────────────┐          ┌──────────────────────────┐  │
│  │ TrajectoryStorage    │          │ TrajectoryStorage        │  │
│  │ (Abstract Base)      │◄─────────┤ (Abstract Base)          │  │
│  │                      │          │                          │  │
│  └──────────────────────┘          └──────────────────────────┘  │
│           ↓                                    ↓                  │
│  ┌──────────────────────┐          ┌──────────────────────────┐  │
│  │ InMemoryTrajectory   │          │ DiskTrajectoryStorage    │  │
│  │ Storage              │          │                          │  │
│  │ - Dict[tid, points]  │          │ - Binary data file       │  │
│  │ - 全内存访问           │          │ - Numpy index array      │  │
│  │ - O(1)随机访问         │          │ - LRU memory cache       │  │
│  └──────────────────────┘          │ - Mmap/seek加载          │  │
│                                    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕ 底层存储
┌─────────────────────────────────────────────────────────────────┐
│                      物理存储层 (Physical)                         │
│        RAM                                    SSD/HDD             │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 核心接口设计

### 3.1 抽象基类 `TrajectoryStorage`

```python
class TrajectoryStorage(ABC):
    """轨迹存储抽象基类，定义统一访问接口"""
    
    @abstractmethod
    def get_trajectory(self, tid: int) -> List[Tuple[float, float]]:
        """获取指定轨迹的坐标点序列"""
        pass
    
    @abstractmethod
    def store_trajectory(self, tid: int, points: List[Tuple[float, float]]) -> None:
        """存储轨迹（批量写入阶段调用）"""
        pass
    
    @abstractmethod
    def get_trajectory_mbr(self, tid: int) -> SpatialBoundingBox:
        """获取轨迹MBR（最小边界矩形）"""
        pass
    
    @abstractmethod
    def get_many_trajectories(self, tids: List[int]) -> Dict[int, List[Tuple[float, float]]]:
        """批量获取多条轨迹，用于相似度计算优化"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭存储，释放资源"""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """返回存储的轨迹总数"""
        pass
    
    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """返回当前内存占用（用于监控）"""
        pass
```

### 3.2 内存存储实现 `InMemoryTrajectoryStorage`

沿用现有实现逻辑：
- `trajectory_points: Dict[int, List[Tuple[float, float]]]`
- `trajectory_mbrs: Dict[int, SpatialBoundingBox]`

### 3.3 磁盘存储实现 `DiskTrajectoryStorage`

**存储格式设计：**

```
# 索引文件 (traj_index.npy) - Numpy结构化数组
# tid | data_offset | data_size | mbr_min_x | mbr_min_y | mbr_max_x | mbr_max_y

# 数据文件 (traj_data.bin) - 紧凑二进制格式
[Header: num_points: uint32]
[Points: (x: float32, y: float32) * num_points]
```

**内存缓存策略：**
- LRU Cache，可配置大小（默认2GB）
- 缓存对象：解码后的坐标点列表
- 缓存键：tid

**读取优化：**
- 批量读取时预加载连续区块
- 使用 `mmap` 进行内存映射（可选）

## 4. 与现有系统集成

### 4.1 QuadTreeIndex 改造

**现状：**
```python
class QuadTreeIndex:
    def __init__(self, ...):
        self.trajectory_points: Dict[int, List[Tuple[float, float]]] = {}  # 30GB在这里
        self.trajectory_mbrs: Dict[int, SpatialBoundingBox] = {}
```

**改造后：**
```python
class QuadTreeIndex:
    def __init__(self, ..., storage: Optional[TrajectoryStorage] = None):
        self._storage = storage or InMemoryTrajectoryStorage()  # 默认内存模式
        
    def assign_trajectory(self, traj_id: int, points: List[Tuple[float, float]]) -> None:
        # 统一调用存储接口
        self._storage.store_trajectory(traj_id, points)
        # ... 其他逻辑
```

### 4.2 配置扩展

```yaml
# config.yaml 新增配置项
data:
  storage_mode: "auto"  # "memory" | "disk" | "auto"
  storage_dir: "resource/storage"  # 磁盘模式存储路径
  disk_cache_mb: 2048  # 磁盘模式LRU缓存大小(MB)
  memory_threshold_gb: 8  # auto模式下，预估超过此阈值自动切换磁盘模式
```

## 5. 实施计划

### Phase 1: 基础架构 (2-3天)
1. 创建 `TrajectoryStorage` 抽象基类
2. 实现 `InMemoryTrajectoryStorage`（迁移现有逻辑）
3. 编写基础单元测试

### Phase 2: 磁盘存储实现 (3-4天)
1. 实现 `DiskTrajectoryStorage` 核心读写逻辑
2. 实现 LRU 缓存机制
3. 批量读取优化
4. 磁盘存储单元测试

### Phase 3: 集成改造 (2-3天)
1. 改造 `QuadTreeIndex` 支持注入式存储
2. 配置系统扩展
3. 更新 `component_factory.py` 的工厂方法
4. 集成测试

### Phase 4: 优化与验证 (2天)
1. 大数据集性能测试（模拟30GB）
2. 内存占用监控
3. 读取性能基准测试

## 6. 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| 磁盘I/O成为瓶颈 | 高 | 1) LRU缓存热点数据 2) 批量预读取 3) 考虑SSD必需 |
| 二进制格式兼容性 | 中 | 版本号头+向后兼容读取逻辑 |
| 内存/磁盘模式切换复杂性 | 低 | 统一接口，工厂模式创建实例 |

## 7. 验收标准

- [ ] 内存模式与磁盘模式API完全一致
- [ ] 磁盘模式支持30GB数据集，内存占用< 4GB
- [ ] 磁盘模式随机读取延迟 < 10ms（缓存命中时< 1ms）
- [ ] 批量读取1000条轨迹 < 500ms
- [ ] 现有小数据集测试用例100%通过
