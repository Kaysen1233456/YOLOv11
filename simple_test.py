"""
简化版测试脚本
"""

import os
import sys


def check_environment():
    """检查环境"""
    print("=== 检查环境 ===")
    print(f"Python版本: {sys.version}")
    
    packages = ['ultralytics', 'streamlit', 'cv2', 'numpy', 'PIL']
    for package in packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ OpenCV已安装")
            elif package == 'PIL':
                from PIL import Image
                print(f"✅ Pillow已安装")
            else:
                __import__(package)
                print(f"✅ {package}已安装")
        except ImportError:
            print(f"❌ 缺少{package}")


def check_files():
    """检查关键文件"""
    print("\n=== 检查文件 ===")
    
    # 检查配置文件
    files_to_check = [
        'dataset.yaml',
        'app.py',
        'train.py',
        os.path.join('runs', 'detect', 'train', 'weights', 'yolo11l.pt'),
        os.path.join('datasets', 'power_safety', 'train', 'images'),
        os.path.join('datasets', 'power_safety', 'train', 'labels')
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            if os.path.isfile(file):
                size = os.path.getsize(file)
                print(f"✅ {file} (大小: {size} 字节)")
            else:
                # 是目录
                items = len(os.listdir(file)) if os.path.exists(file) else 0
                print(f"✅ {file} (包含 {items} 个项目)")
        else:
            print(f"❌ {file} 不存在")


def check_dataset_config():
    """检查数据集配置"""
    print("\n=== 检查数据集配置 ===")
    
    if os.path.exists('dataset.yaml'):
        try:
            import yaml
            with open('dataset.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"路径: {config.get('path', '未设置')}")
            print(f"类别数: {config.get('nc', '未设置')}")
            print(f"类别名称: {config.get('names', '未设置')}")
        except Exception as e:
            print(f"读取配置文件出错: {e}")
    else:
        print("找不到dataset.yaml")


if __name__ == "__main__":
    check_environment()
    check_files()
    check_dataset_config()
    print("\n✅ 简单检查完成")