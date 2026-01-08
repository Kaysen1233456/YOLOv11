"""
验证脚本，用于检查项目设置是否正确

此脚本将检查：
1. 数据集结构是否正确
2. 权重文件是否存在
3. 模型是否能正常加载
4. 环境依赖是否满足
"""

import os
import yaml
from pathlib import Path


def check_dataset():
    """检查数据集结构和完整性"""
    print("=== 检查数据集 ===")
    
    # 检查配置文件
    config_file = "dataset.yaml"
    if not os.path.exists(config_file):
        print(f"❌ 未找到配置文件: {config_file}")
        return False
        
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ 配置文件存在")
    print(f"  路径: {config['path']}")
    print(f"  类别数: {config['nc']}")
    print(f"  类别名称: {config['names']}")
    
    # 检查数据集路径
    dataset_path = config['path']
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return False
        
    # 检查训练集和验证集
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    
    if not os.path.exists(train_path):
        print(f"❌ 训练集路径不存在: {train_path}")
        return False
        
    if not os.path.exists(val_path):
        print(f"❌ 验证集路径不存在: {val_path}")
        return False
    
    # 检查训练集中的图像和标签
    train_images = os.path.join(train_path, "images")
    train_labels = os.path.join(train_path, "labels")
    
    if not os.path.exists(train_images):
        print(f"❌ 训练图像目录不存在: {train_images}")
        return False
        
    if not os.path.exists(train_labels):
        print(f"❌ 训练标签目录不存在: {train_labels}")
        return False
    
    # 统计训练集文件数
    train_image_count = len([f for f in os.listdir(train_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    train_label_count = len([f for f in os.listdir(train_labels) if f.lower().endswith('.txt')])
    
    print(f"✅ 训练集:")
    print(f"  图像文件数: {train_image_count}")
    print(f"  标签文件数: {train_label_count}")
    
    if train_image_count == 0:
        print("⚠️  训练集没有图像文件")
        
    if train_label_count == 0:
        print("⚠️  训练集没有标签文件")
    
    # 检查验证集中的图像和标签
    val_images = os.path.join(val_path, "images")
    val_labels = os.path.join(val_path, "labels")
    
    if not os.path.exists(val_images):
        print(f"❌ 验证图像目录不存在: {val_images}")
        return False
        
    if not os.path.exists(val_labels):
        print(f"❌ 验证标签目录不存在: {val_labels}")
        return False
    
    # 统计验证集文件数
    val_image_count = len([f for f in os.listdir(val_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_label_count = len([f for f in os.listdir(val_labels) if f.lower().endswith('.txt')])
    
    print(f"✅ 验证集:")
    print(f"  图像文件数: {val_image_count}")
    print(f"  标签文件数: {val_label_count}")
    
    if val_image_count == 0:
        print("⚠️  验证集没有图像文件")
        
    if val_label_count == 0:
        print("⚠️  验证集没有标签文件")
    
    return True


def check_weights():
    """检查权重文件"""
    print("\n=== 检查权重文件 ===")
    
    # 检查默认权重路径
    default_weight_path = os.path.join("runs", "detect", "train", "weights", "yolo11n.pt")
    if os.path.exists(default_weight_path):
        size = os.path.getsize(default_weight_path) / (1024*1024)  # MB
        print(f"✅ 默认权重文件存在: {default_weight_path} ({size:.1f} MB)")
        return True
    else:
        print(f"⚠️  默认权重文件不存在: {default_weight_path}")
        return False


def check_model_loading():
    """检查模型加载"""
    print("\n=== 检查模型加载 ===")
    
    try:
        from ultralytics import YOLO
        print("✅ ultralytics 包导入成功")
    except ImportError as e:
        print(f"❌ 无法导入 ultralytics: {e}")
        return False
    
    # 尝试加载模型
    weight_path = os.path.join("runs", "detect", "train", "weights", "yolo11n.pt")
    if os.path.exists(weight_path):
        try:
            model = YOLO(weight_path)
            print(f"✅ 模型加载成功: {weight_path}")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    else:
        # 尝试加载默认模型
        try:
            model = YOLO("yolo11n.pt")
            print("✅ 默认模型加载成功")
            return True
        except Exception as e:
            print(f"❌ 默认模型加载失败: {e}")
            return False


def check_dependencies():
    """检查依赖包"""
    print("\n=== 检查依赖包 ===")
    
    required_packages = [
        "ultralytics",
        "streamlit", 
        "cv2",
        "PIL",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
                print(f"✅ {package} (OpenCV) 导入成功")
            elif package == "PIL":
                from PIL import Image
                print(f"✅ {package} (Pillow) 导入成功")
            else:
                __import__(package)
                print(f"✅ {package} 导入成功")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"⚠️  缺少依赖包: {missing_packages}")
        return False
    else:
        print("✅ 所有依赖包均已安装")
        return True


def main():
    """主函数"""
    print("开始验证项目设置...\n")
    
    checks = [
        ("依赖包检查", check_dependencies),
        ("数据集检查", check_dataset),
        ("权重文件检查", check_weights),
        ("模型加载检查", check_model_loading)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name}执行出错: {e}")
            results.append((check_name, False))
    
    print("\n=== 验证结果汇总 ===")
    all_passed = True
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有检查通过！项目设置正确。")
        return True
    else:
        print("\n⚠️  有些检查未通过，请查看上面的错误信息。")
        return False


if __name__ == "__main__":
    main()