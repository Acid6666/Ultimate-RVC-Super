#!/usr/bin/env python3
"""
测试 RVC 服务的显存清理功能
"""
import os
import sys
import time
import gc

# 添加项目路径
sys.path.insert(0, '/data/lizilong/ultimate-rvc/src')

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
        print(f"{prefix}显存使用情况:")
        print(f"{prefix}  总显存: {total:.2f} MB")
        print(f"{prefix}  已分配: {allocated:.2f} MB ({allocated/total*100:.1f}%)")
        print(f"{prefix}  已保留: {reserved:.2f} MB ({reserved/total*100:.1f}%)")
        print(f"{prefix}  空闲: {free:.2f} MB ({free/total*100:.1f}%)")
        return allocated, reserved
    else:
        print(f"{prefix}CUDA 不可用")
        return 0, 0

def test_voice_converter_cleanup():
    """测试 VoiceConverter 的清理功能"""
    print("=" * 60)
    print("测试 VoiceConverter 显存清理功能")
    print("=" * 60)

    print("\n1. 初始显存状态:")
    initial_alloc, initial_reserved = print_memory_usage("  ")

    print("\n2. 创建 VoiceConverter 实例:")
    from ultimate_rvc.rvc.infer.infer import VoiceConverter
    vc = VoiceConverter()
    after_create_alloc, after_create_reserved = print_memory_usage("  ")
    print(f"  增加: {after_create_alloc - initial_alloc:.2f} MB")

    print("\n3. 清理 VoiceConverter:")
    vc.cleanup_model()
    del vc
    gc.collect()
    torch = get_torch()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    after_cleanup_alloc, after_cleanup_reserved = print_memory_usage("  ")
    print(f"  清理效果: {after_create_alloc - after_cleanup_alloc:.2f} MB")
    print(f"  清理率: {((after_create_alloc - after_cleanup_alloc) / (after_create_alloc - initial_alloc) * 100) if after_create_alloc > initial_alloc else 0:.1f}%")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

def test_with_cached_converter():
    """测试带缓存的清理功能"""
    print("\n" + "=" * 60)
    print("测试带缓存的 VoiceConverter 清理")
    print("=" * 60)

    print("\n1. 初始显存状态:")
    initial_alloc, initial_reserved = print_memory_usage("  ")

    print("\n2. 使用缓存的 VoiceConverter:")
    from ultimate_rvc.core.generate.common import _get_voice_converter
    vc1 = _get_voice_converter()
    after_first_alloc, _ = print_memory_usage("  ")
    print(f"  第一次加载增加: {after_first_alloc - initial_alloc:.2f} MB")

    print("\n3. 再次获取缓存:")
    vc2 = _get_voice_converter()
    print(f"  是否为同一实例: {vc1 is vc2}")

    print("\n4. 清理缓存:")
    _get_voice_converter.cache_clear()
    vc1.cleanup_model()
    del vc1, vc2
    gc.collect()
    torch = get_torch()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    after_cleanup_alloc, _ = print_memory_usage("  ")
    print(f"  清理效果: {after_first_alloc - after_cleanup_alloc:.2f} MB")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_voice_converter_cleanup()
    test_with_cached_converter()
