# 测试说明

## 测试结构

```
tests/
├── conftest.py           # pytest配置和共享fixtures
├── test_config.py        # 配置管理测试
├── generate_queries.py   # 查询生成测试
├── mapping_load.py       # 映射加载测试
├── quad_index.py         # 四叉树索引测试
├── search.py            # 搜索功能测试
└── state_build.py       # 状态构建测试
```

## 运行测试

### 运行所有测试
```bash
pytest
```

### 运行特定测试文件
```bash
pytest tests/test_config.py
```

### 运行特定测试类或函数
```bash
pytest tests/test_config.py::TestConfigCreation
pytest tests/test_config.py::TestConfigCreation::test_default_config_creation
```

### 使用标记运行测试
```bash
# 只运行单元测试
pytest -m unit

# 只运行配置相关测试
pytest -m config

# 排除慢速测试
pytest -m "not slow"
```

### 查看详细输出
```bash
pytest -v
pytest -vv  # 更详细
```

### 生成覆盖率报告
```bash
pytest --cov=src --cov-report=html
# 报告生成在 htmlcov/index.html
```

## 测试分类

### 单元测试 (unit)
测试单个函数或类的功能，不依赖外部资源。

### 集成测试 (integration)
测试多个组件的协作，可能需要文件系统或数据库。

### 慢速测试 (slow)
运行时间较长的测试，通常涉及大量数据或复杂计算。

## 编写测试

### 使用fixtures
```python
def test_with_config(test_config):
    """使用共享的测试配置"""
    assert test_config.index.max_level == 4
```

### 使用临时目录
```python
def test_save_file(temp_config_dir):
    """使用临时配置目录"""
    file_path = temp_config_dir / "test.yaml"
    # 测试代码...
```

### 添加测试标记
```python
import pytest

@pytest.mark.unit
def test_simple_function():
    pass

@pytest.mark.slow
@pytest.mark.integration
def test_complex_workflow():
    pass
```

## 最佳实践

1. **测试命名**: 使用描述性的测试名称，如 `test_config_loads_from_yaml`
2. **独立性**: 每个测试应该独立运行，不依赖其他测试的状态
3. **清理**: 使用fixtures自动清理临时文件和资源
4. **断言**: 使用清晰的断言消息，便于调试
5. **覆盖率**: 保持测试覆盖率在80%以上

## 持续集成

测试应该在每次提交前运行：

```bash
# 快速检查
pytest -m "not slow"

# 完整测试
pytest --cov=src
```

## 调试测试

### 进入调试器
```bash
pytest --pdb  # 失败时进入pdb
```

### 查看打印输出
```bash
pytest -s  # 显示print输出
```

### 只运行失败的测试
```bash
pytest --lf  # last failed
pytest --ff  # failed first
```
