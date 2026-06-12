#!/usr/bin/env python3
"""
测试显存清理功能
"""
import os
import sys
import time

# 延迟导入 torch 以避免启动时创建 CUDA 上下文
torch_available = None

def get_torch():
    """延迟导入 torch"""
    global torch_available
    if torch_available is None:
        try:
            import torch
            torch_available = torch
        except ImportError:
            torch_available = False
    return torch_available if torch_available is not False else None

def print_memory_usage():
    """打印当前显存使用情况"""
    torch = get_torch()
    if torch and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(0) / 1024 / 1024
        print(f"显存使用情况:")
        print(f"  已分配: {allocated:.2f} MB")
        print(f"  已保留: {reserved:.2f} MB")
        print(f"  可用: {torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0):.2f} MB")
        return allocated, reserved
    else:
        print("CUDA 不可用")
        return 0, 0

def test_cleanup():
    """测试清理功能"""
    print("=" * 50)
    print("测试显存清理功能")
    print("=" * 50)

    print("\n1. 初始显存状态:")
    initial_alloc, initial_reserved = print_memory_usage()

    print("\n2. 模拟模型加载（创建一些张量）:")
    torch = get_torch()
    if torch and torch.cuda.is_available():
        # 创建一些张量模拟模型加载
        tensors = []
        for i in range(5):
            t = torch.randn(1000, 1000, 100).cuda()
            tensors.append(t)
        print(f"  已创建 {len(tensors)} 个张量")
        print(f"  张量总大小: {sum(t.numel() for t in tensors) * 4 / 1024 / 1024:.2f} MB")

        after_load_alloc, after_load_reserved = print_memory_usage()

        print("\n3. 清理前后的对比:")
        print(f"  增加的显存: {after_load_alloc - initial_alloc:.2f} MB")

    print("\n4. 开始清理...")
    import gc
    torch = get_torch()
    if torch:
        if torch.cuda.is_available():
            # 清理对象
            del tensors

            # 垃圾回收
            gc.collect()

            # 清理 CUDA 缓存
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

    after_cleanup_alloc, after_cleanup_reserved = print_memory_usage()

    print("\n5. 清理效果:")
    print(f"  清理前后差值: {after_load_alloc - after_cleanup_alloc:.2f} MB")
    print(f"  清理率: {((after_load_alloc - after_cleanup_alloc) / (after_load_alloc - initial_alloc) * 100):.2f}%")

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_cleanup()
