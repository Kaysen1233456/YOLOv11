#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 项目清理脚本
自动整理项目目录，归档旧脚本，删除无用文件
"""

import os
import shutil
from pathlib import Path


def cleanup_project():
    """清理和整理 YOLOv11 项目目录"""
    
    # 获取项目根目录（脚本所在目录）
    project_root = Path(__file__).parent
    
    print(f"🔍 项目根目录: {project_root}")
    print("=" * 60)
    
    # 1. 创建 archived_scripts 目录
    archived_dir = project_root / "archived_scripts"
    archived_dir.mkdir(exist_ok=True)
    print(f"✓ 创建归档目录: {archived_dir}")
    
    # 2. 定义需要归档的文件列表
    files_to_archive = [
        "simple_train.py",
        "full_train.py",
        "train_with_yolo11l.py",
        "enhanced_train.py",
    ]
    
    # 3. 添加所有 test_*.py 文件
    test_files = list(project_root.glob("test_*.py"))
    files_to_archive.extend([f.name for f in test_files])
    
    # 4. 添加所有 monitor_*.py 文件
    monitor_files = list(project_root.glob("monitor_*.py"))
    files_to_archive.extend([f.name for f in monitor_files])
    
    # 5. 移动文件到归档目录
    print("\n📦 开始归档旧脚本...")
    archived_count = 0
    for filename in files_to_archive:
        file_path = project_root / filename
        if file_path.exists() and file_path.is_file():
            dest_path = archived_dir / filename
            shutil.move(str(file_path), str(dest_path))
            print(f"  ✓ 已归档: {filename}")
            archived_count += 1
        else:
            print(f"  ⊘ 文件不存在，跳过: {filename}")
    
    print(f"\n✓ 共归档 {archived_count} 个文件")
    
    # 6. 确保权重文件在正确位置
    print("\n🏋️  检查权重文件...")
    weights_dir = project_root / "weights"
    
    # 检查权重文件
    weight_files = ["yolo11n.pt", "yolo11l.pt"]
    for weight_file in weight_files:
        root_weight = project_root / weight_file
        weights_subdir_weight = weights_dir / weight_file
        
        if root_weight.exists():
            print(f"  ✓ 权重文件已存在于根目录: {weight_file}")
        elif weights_dir.exists() and weights_subdir_weight.exists():
            print(f"  ✓ 权重文件已存在于 weights/ 目录: {weight_file}")
        else:
            print(f"  ⊘ 未找到权重文件: {weight_file}")
    
    # 7. 删除脏数据
    print("\n🗑️  清理无用文件...")
    dirty_files = ["image.png"]  # 健身房图片
    
    for dirty_file in dirty_files:
        file_path = project_root / dirty_file
        if file_path.exists() and file_path.is_file():
            os.remove(file_path)
            print(f"  ✓ 已删除: {dirty_file}")
        else:
            print(f"  ⊘ 文件不存在，跳过: {dirty_file}")
    
    # 8. 显示保留的核心文件
    print("\n📌 保留的核心文件:")
    core_files = ["requirements.txt", "dataset.yaml", "app.py"]
    for core_file in core_files:
        file_path = project_root / core_file
        if file_path.exists():
            print(f"  ✓ {core_file}")
        else:
            print(f"  ⚠️  {core_file} (未找到)")
    
    print("\n" + "=" * 60)
    print("✅ 项目清理完成!")
    print(f"\n归档的脚本位于: {archived_dir}")
    print("\n项目根目录现在更加整洁，只保留核心文件。")


if __name__ == "__main__":
    try:
        cleanup_project()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
