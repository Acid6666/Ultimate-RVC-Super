#!/usr/bin/env python3
"""
简单测试显存清理逻辑
"""
import gc
import time

def get_torch():
    """延迟导入 torch"""
    try:
        import torch
        return torch
    except ImportError:
        return None

def print_memory_usage(prefix=""):
    """打印当前显存使用情况"""
    torch = get_torch()
    if torch and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(0) / 1024 / 1024
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        free = total - reserved
        print(f"{prefix}显存使用:")
        print(f"{prefix}  总显存: {total:.2f} MB")
        print(f"{prefix}  已分配: {allocated:.2f} MB ({allocated/total*100:.1f}%)")
        print(f"{prefix}  已保留: {reserved:.2f} MB ({reserved/total*100:.1f}%)")
        print(f"{prefix}  空闲: {free:.2f} MB ({free/total*100:.1f}%)")
        return allocated, reserved
    else:
        print(f"{prefix}CUDA 不可用")
        return 0, 0

def full_cleanup():
    """完整的清理流程"""
    torch = get_torch()
    if torch:
        # 1. 垃圾回收
        gc.collect()

        # 2. 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

def test_cleanup_levels():
    """测试不同清理级别的效果"""
    print("=" * 60)
    print("测试显存清理级别")
    print("=" * 60)

    print("\n1. 初始状态:")
    initial_alloc, initial_reserved = print_memory_usage("  ")

    torch = get_torch()
    if not torch or not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试")
        return

    print("\n2. 创建大量张量（模拟模型加载）:")
    # 模拟加载大模型
    model_tensors = []
    for i in range(10):
        t = torch.randn(512, 512, 512).cuda()
        model_tensors.append(t)

    after_load_alloc, after_load_reserved = print_memory_usage("  ")
    print(f"  增加的显存: {after_load_alloc - initial_alloc:.2f} MB")

    print("\n3. 测试清理级别 1: 只清理缓存")
    torch.cuda.empty_cache()
    after_level1_alloc, after_level1_reserved = print_memory_usage("  ")
    print(f"  清理效果: {after_load_alloc - after_level1_alloc:.2f} MB")

    print("\n4. 测试清理级别 2: 删除对象 + 清理缓存")
    del model_tensors
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after_level2_alloc, after_level2_reserved = print_memory_usage("  ")
    print(f"  清理效果: {after_load_alloc - after_level2_alloc:.2f} MB")

    print("\n5. 测试清理级别 3: 完整清理（包含 ipc_collect）")
    full_cleanup()
    after_level3_alloc, after_level3_reserved = print_memory_usage("  ")
    print(f"  清理效果: {after_load_alloc - after_level3_alloc:.2f} MB")

    print("\n6. 等待 5 秒再检查:")
    time.sleep(5)
    after_wait_alloc, after_wait_reserved = print_memory_usage("  ")
    print(f"  变化: {after_wait_alloc - after_level3_alloc:.2f} MB")

    print("\n" + "=" * 60)
    print("总结:")
    print(f"  初始显存: {initial_alloc:.2f} MB")
    print(f"  加载后显存: {after_load_alloc:.2f} MB")
    print(f"  清理后显存: {after_wait_alloc:.2f} MB")
    print(f"  净增长: {after_wait_alloc - initial_alloc:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    test_cleanup_levels()
