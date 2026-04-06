"""轨迹存储模块单元测试"""
import os
import tempfile
from pathlib import Path

import pytest

from src.core.bounding_box import SpatialBoundingBox
from src.indexing.quadtree_index import QuadTreeIndex
from src.storage import create_storage, InMemoryTrajectoryStorage, DiskTrajectoryStorage


class TestInMemoryTrajectoryStorage:
    """内存存储测试"""
    
    def test_basic_operations(self):
        """测试基本CRUD操作"""
        storage = InMemoryTrajectoryStorage()
        
        # 存储轨迹
        points = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
        storage.store_trajectory(1, points)
        
        # 读取轨迹
        retrieved = storage.get_trajectory(1)
        assert retrieved == points
        
        # 检查存在性
        assert storage.has_trajectory(1)
        assert not storage.has_trajectory(999)
        
        # 获取MBR
        mbr = storage.get_trajectory_mbr(1)
        assert mbr.min_x == 0.1
        assert mbr.min_y == 0.2
        assert mbr.max_x == 0.5
        assert mbr.max_y == 0.6
        
        # 获取所有ID
        assert storage.get_all_tids() == [1]
        
        # 长度
        assert len(storage) == 1
    
    def test_batch_operations(self):
        """测试批量操作"""
        storage = InMemoryTrajectoryStorage()
        
        # 存储多条轨迹
        for i in range(10):
            points = [(i * 0.1, i * 0.1), ((i + 1) * 0.1, (i + 1) * 0.1)]
            storage.store_trajectory(i, points)
        
        # 批量获取
        tids = [0, 2, 4, 6, 8]
        result = storage.get_many_trajectories(tids)
        assert len(result) == 5
        assert all(tid in result for tid in tids)
    
    def test_nonexistent_trajectory(self):
        """测试访问不存在的轨迹"""
        storage = InMemoryTrajectoryStorage()
        
        with pytest.raises(KeyError):
            storage.get_trajectory(999)
        
        with pytest.raises(KeyError):
            storage.get_trajectory_mbr(999)
    
    def test_memory_usage(self):
        """测试内存占用估算"""
        storage = InMemoryTrajectoryStorage()
        
        initial = storage.memory_usage_bytes
        
        # 添加轨迹
        points = [(0.1, 0.2)] * 1000  # 1000个点
        storage.store_trajectory(1, points)
        
        after = storage.memory_usage_bytes
        assert after > initial


class TestDiskTrajectoryStorage:
    """磁盘存储测试"""
    
    def test_basic_operations(self):
        """测试基本CRUD操作"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskTrajectoryStorage(Path(tmpdir), cache_mb=100)
            
            # 存储轨迹（批量模式）
            points = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
            storage.store_trajectory(1, points)
            
            # 完成写入
            storage.finalize_writes()
            
            # 读取轨迹
            retrieved = storage.get_trajectory(1)
            assert len(retrieved) == 3
            assert abs(retrieved[0][0] - 0.1) < 1e-6  # float32精度
            
            # 检查存在性
            assert storage.has_trajectory(1)
            assert not storage.has_trajectory(999)
            
            # 获取MBR（从索引直接读取）
            mbr = storage.get_trajectory_mbr(1)
            assert abs(mbr.min_x - 0.1) < 1e-6
            
            storage.close()
    
    def test_persistence(self):
        """测试数据持久化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 第一次会话：写入数据
            storage1 = DiskTrajectoryStorage(Path(tmpdir), cache_mb=100)
            for i in range(5):
                points = [(i * 0.1, i * 0.1), ((i + 1) * 0.1, (i + 1) * 0.1)]
                storage1.store_trajectory(i, points)
            storage1.finalize_writes()
            storage1.close()
            
            # 第二次会话：读取数据
            storage2 = DiskTrajectoryStorage(Path(tmpdir), cache_mb=100)
            assert len(storage2) == 5
            assert storage2.has_trajectory(3)
            
            points = storage2.get_trajectory(3)
            assert len(points) == 2
            storage2.close()
    
    def test_batch_read(self):
        """测试批量读取优化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskTrajectoryStorage(Path(tmpdir), cache_mb=100)
            
            # 写入
            for i in range(10):
                points = [(i * 0.1, i * 0.1)] * 50
                storage.store_trajectory(i, points)
            storage.finalize_writes()
            
            # 批量读取
            tids = [0, 2, 4, 6, 8]
            result = storage.get_many_trajectories(tids)
            assert len(result) == 5
            
            storage.close()


class TestStorageFactory:
    """存储工厂测试"""
    
    def test_create_memory_storage(self):
        """测试创建内存存储"""
        storage = create_storage("memory")
        assert isinstance(storage, InMemoryTrajectoryStorage)
    
    def test_create_disk_storage(self):
        """测试创建磁盘存储"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = create_storage("disk", storage_dir=tmpdir, cache_mb=100)
            assert isinstance(storage, DiskTrajectoryStorage)
    
    def test_auto_mode_small_data(self):
        """测试auto模式选择内存（小数据）"""
        storage = create_storage("auto", estimated_data_gb=1.0)
        assert isinstance(storage, InMemoryTrajectoryStorage)
    
    def test_auto_mode_large_data(self):
        """测试auto模式选择磁盘（大数据）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = create_storage("auto", storage_dir=tmpdir, estimated_data_gb=30.0)
            assert isinstance(storage, DiskTrajectoryStorage)


class TestQuadTreeWithStorage:
    """QuadTree与存储集成测试"""
    
    def test_quadtree_with_memory_storage(self):
        """测试QuadTree使用内存存储"""
        storage = InMemoryTrajectoryStorage()
        bbox = SpatialBoundingBox(0, 0, 1, 1)
        quadtree = QuadTreeIndex(bbox, max_level=4, storage=storage)
        
        # 分配轨迹
        points = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
        quadtree.assign_trajectory(1, points)
        
        # 验证存储
        assert storage.has_trajectory(1)
        assert quadtree.trajectory_to_cells[1] is not None
        
        # 验证MBR计算
        mbr = storage.get_trajectory_mbr(1)
        assert mbr.min_x == 0.1
        assert mbr.max_x == 0.3
    
    def test_quadtree_with_disk_storage(self):
        """测试QuadTree使用磁盘存储"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskTrajectoryStorage(Path(tmpdir), cache_mb=100)
            bbox = SpatialBoundingBox(0, 0, 1, 1)
            quadtree = QuadTreeIndex(bbox, max_level=4, storage=storage)
            
            # 分配轨迹
            points = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
            quadtree.assign_trajectory(1, points)
            
            # 在剪枝前完成写入（剪枝需要读取轨迹点计算签名）
            storage.finalize_writes()
            
            # 验证轨迹可访问
            assert storage.has_trajectory(1)
            
            # 测试相交查询
            cell = quadtree.trajectory_to_cells[1]
            assert quadtree.trajectory_intersects_cell(1, cell)
            
            storage.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
