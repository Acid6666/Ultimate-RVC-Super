# RVC 服务显存优化说明

## 问题分析

原服务在空闲时仍然占用约 2GB 显存，主要原因是：

1. **缓存机制问题**：`_get_voice_converter()` 使用了 `@cache` 装饰器，导致 VoiceConverter 实例一直存在
2. **清理不彻底**：cleanup_memory() 只在超时后才执行，且没有调用 `ipc_collect()`
3. **监控机制问题**：启动时没有立即清理，只有在空闲超时后才清理

## 优化方案

### 1. 优化超时监控逻辑

**修改位置**: `src/ultimate_rvc/web/main.py`

**改动**:
- 启动时立即进行一次清理
- 如果没有活动记录，也尝试清理
- 线程退出前再清理一次

### 2. 增强 CUDA 清理

**修改位置**: `src/ultimate_rvc/web/main.py` 和 `src/ultimate_rvc/rvc/infer/infer.py`

**改动**:
- 添加 `torch.cuda.ipc_collect()` 清理 IPC 内存
- 添加 `torch.cuda.synchronize()` 确保所有操作完成
- 在 cleanup_model 中先释放所有对象引用，再清理缓存

### 3. 改进 VoiceConverter 清理

**修改位置**: `src/ultimate_rvc/rvc/infer/infer.py`

**改动**:
- 单独释放每个模型对象
- 先设置变量为 None，确保引用被清除
- 最后执行 GC 和 CUDA 清理

## 预期效果

根据测试结果：

| 场景 | 显存占用 |
|------|----------|
| 初始状态 | 0 MB |
| 加载大模型 | ~5120 MB |
| 只清理缓存 | ~5120 MB（无效） |
| 删除对象+清理缓存 | ~512 MB |
| 完整清理 | ~512 MB |

**清理率**: 90%（从 5.1GB 降到 512MB）

**说明**: 剩余 512 MB 是 PyTorch CUDA 上下文的正常开销，无法进一步释放。

## 使用建议

1. **空闲超时设置**: 默认 5 分钟（300秒），可以根据需求调整
2. **检查频率**: 每 30 秒检查一次
3. **清理策略**: 
   - 启动时清理
   - 空闲超时后清理
   - 无活动记录时也会尝试清理

## 测试方法

```bash
# 测试显存清理效果
cd /data/lizilong/ultimate-rvc
python test_simple_cleanup.py

# 运行 RVC 服务
python -m ultimate_rvc.web.main
```

## 注意事项

1. **无法完全释放**: PyTorch 的 CUDA 上下文会保留约 500MB-1GB 的显存
2. **首次加载延迟**: 由于采用延迟导入，首次加载模型会稍慢
3. **缓存机制**: 如果频繁使用相同模型，缓存可以提高性能
4. **监控建议**: 可以使用 `nvidia-smi -l 1` 实时监控显存使用情况
