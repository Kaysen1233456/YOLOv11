"""
权重管理脚本

此脚本用于帮助用户管理YOLO模型权重文件，包括：
1. 查看已下载的权重文件
2. 将权重文件复制到指定位置
3. 验证权重文件完整性

用法：
python manage_weights.py --action list
python manage_weights.py --action copy --source [源路径] --destination [目标路径]
"""

import os
import shutil
import argparse
from pathlib import Path


def list_weights():
    """列出项目中所有可能的权重文件位置"""
    project_root = Path(__file__).resolve().parent
    print(f"项目根目录: {project_root}")
    
    # 常见权重文件位置
    weight_locations = [
        project_root / 'runs' / 'detect' / 'train' / 'weights',
        project_root / 'weights',
        project_root,
        Path.home() / '.cache' / 'ultralytics',
        Path.home() / 'AppData' / 'Local' / 'ultralytics'
    ]
    
    print("\n搜索权重文件...")
    found_weights = []
    
    # 需要查找的权重文件名模式
    weight_patterns = ["*.pt"]
    
    for location in weight_locations:
        if location.exists():
            print(f"\n检查目录: {location}")
            for pattern in weight_patterns:
                for file in location.rglob(pattern):
                    print(f"  发现权重文件: {file}")
                    found_weights.append(file)
        else:
            print(f"\n目录不存在: {location}")
            
    # 检查项目根目录下的.pt文件
    print(f"\n检查项目根目录下的权重文件:")
    for pattern in weight_patterns:
        for file in project_root.glob(pattern):
            print(f"  发现权重文件: {file}")
            found_weights.append(file)
        
    return found_weights


def copy_weights(source, destination):
    """将权重文件从源路径复制到目标路径"""
    source_path = Path(source)
    dest_path = Path(destination)
    
    if not source_path.exists():
        print(f"错误: 源文件不存在: {source_path}")
        return False
        
    try:
        # 确保目标目录存在
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        shutil.copy2(source_path, dest_path)
        print(f"成功将 {source_path} 复制到 {dest_path}")
        return True
    except Exception as e:
        print(f"复制过程中发生错误: {e}")
        return False


def verify_weights(weights_path):
    """验证权重文件是否存在且非空"""
    weights_file = Path(weights_path)
    
    if not weights_file.exists():
        print(f"错误: 权重文件不存在: {weights_file}")
        # 尝试查找相似命名的文件
        similar_files = list(weights_file.parent.glob(weights_file.stem + "*.pt"))
        if similar_files:
            print(f"提示: 在同一目录下找到相似文件:")
            for f in similar_files:
                print(f"  - {f}")
        return False
        
    if weights_file.stat().st_size == 0:
        print(f"错误: 权重文件为空: {weights_file}")
        return False
        
    print(f"权重文件验证通过: {weights_file} (大小: {weights_file.stat().st_size} 字节)")
    return True


def fix_weights():
    """修复权重文件问题，主要是解决yolov11n.pt和yolo11n.pt命名不一致问题"""
    project_root = Path(__file__).resolve().parent
    target_dir = project_root / 'runs' / 'detect' / 'train' / 'weights'
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义常见的权重文件名
    possible_names = [
        "yolov11n.pt",
        "yolo11n.pt",
        "yolov8n.pt",
        "yolo8n.pt"
    ]
    
    # 在项目各处寻找这些权重文件
    search_paths = [
        project_root,
        target_dir,
        project_root / 'weights',
        Path.home() / '.cache' / 'ultralytics',
        Path.home() / 'AppData' / 'Local' / 'ultralytics'
    ]
    
    found_weights = {}
    for path in search_paths:
        if path.exists():
            for name in possible_names:
                weight_file = path / name
                if weight_file.exists():
                    found_weights[name] = weight_file
    
    if not found_weights:
        print("未找到任何权重文件，请先运行 download_yolov11n.py 脚本下载权重文件")
        return False
    
    print("找到以下权重文件:")
    for name, path in found_weights.items():
        print(f"  {name}: {path} ({path.stat().st_size} bytes)")
    
    # 确保目标位置有一个正确的权重文件
    standard_name = "yolo11n.pt"
    target_file = target_dir / standard_name
    
    if standard_name in found_weights:
        if found_weights[standard_name].resolve() != target_file.resolve():
            # 需要复制到标准位置
            if copy_weights(found_weights[standard_name], target_file):
                print(f"已将权重文件复制到标准位置: {target_file}")
        else:
            print(f"权重文件已在标准位置: {target_file}")
    else:
        # 使用找到的第一个权重文件作为替代
        name, path = next(iter(found_weights.items()))
        if copy_weights(path, target_file):
            print(f"已将 {name} 复制到标准位置并重命名为 {standard_name}: {target_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='权重管理工具')
    parser.add_argument('--action', type=str, choices=['list', 'copy', 'verify', 'fix'], 
                       default='list', help='要执行的操作')
    parser.add_argument('--source', type=str, help='源文件路径（用于复制操作）')
    parser.add_argument('--destination', type=str, help='目标文件路径（用于复制操作）')
    parser.add_argument('--weights', type=str, help='权重文件路径（用于验证操作）')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_weights()
    elif args.action == 'copy':
        if not args.source or not args.destination:
            print("错误: 复制操作需要指定 --source 和 --destination 参数")
            return
        copy_weights(args.source, args.destination)
    elif args.action == 'verify':
        if not args.weights:
            print("错误: 验证操作需要指定 --weights 参数")
            return
        verify_weights(args.weights)
    elif args.action == 'fix':
        fix_weights()


if __name__ == '__main__':
    main()