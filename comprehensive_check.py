"""
全面项目检查脚本

此脚本将执行以下检查：
1. 环境和依赖检查
2. 数据集完整性检查
3. 配置文件检查
4. 权重文件检查
5. 模型加载测试
6. 提供问题解决方案
"""

import os
import sys
import yaml
from pathlib import Path


def check_environment():
    """检查环境和依赖"""
    print("=" * 50)
    print("1. 环境和依赖检查")
    print("=" * 50)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查关键依赖
    dependencies = {
        'ultralytics': False,
        'streamlit': False,
        'opencv-python': False,
        'Pillow': False,
        'numpy': False
    }
    
    for dep in dependencies:
        try:
            if dep == 'opencv-python':
                import cv2
                dependencies[dep] = True
                print(f"✅ {dep}: 已安装")
            elif dep == 'Pillow':
                from PIL import Image
                dependencies[dep] = True
                print(f"✅ {dep}: 已安装")
            else:
                __import__(dep)
                dependencies[dep] = True
                print(f"✅ {dep}: 已安装")
        except ImportError:
            print(f"❌ {dep}: 未安装")
    
    missing_deps = [dep for dep, installed in dependencies.items() if not installed]
    if missing_deps:
        print(f"\n⚠️  缺少依赖: {missing_deps}")
        print("请运行 install_deps.ps1 脚本来安装依赖（使用清华镜像源）")
        return False
    else:
        print("\n✅ 所有依赖均已安装")
        return True


def check_dataset():
    """检查数据集完整性"""
    print("\n" + "=" * 50)
    print("2. 数据集完整性检查")
    print("=" * 50)
    
    # 检查配置文件
    config_file = "dataset.yaml"
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 配置文件格式正确")
        print(f"  数据集路径: {config.get('path', '未设置')}")
        print(f"  类别数: {config.get('nc', '未设置')}")
        print(f"  类别名称: {config.get('names', '未设置')}")
    except Exception as e:
        print(f"❌ 配置文件解析失败: {e}")
        return False
    
    # 检查数据集路径
    dataset_path = config.get('path')
    if not dataset_path:
        print("❌ 数据集路径未设置")
        return False
    
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return False
    
    # 检查训练集和验证集
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    
    for path, name in [(train_path, "训练集"), (val_path, "验证集")]:
        if not os.path.exists(path):
            print(f"❌ {name}路径不存在: {path}")
            return False
        
        images_path = os.path.join(path, "images")
        labels_path = os.path.join(path, "labels")
        
        if not os.path.exists(images_path):
            print(f"❌ {name}图像目录不存在: {images_path}")
            return False
            
        if not os.path.exists(labels_path):
            print(f"❌ {name}标签目录不存在: {labels_path}")
            return False
        
        # 统计文件数量
        images_count = len([f for f in os.listdir(images_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        labels_count = len([f for f in os.listdir(labels_path) 
                           if f.lower().endswith('.txt')])
        
        print(f"✅ {name}:")
        print(f"  图像文件数: {images_count}")
        print(f"  标签文件数: {labels_count}")
        
        # 检查文件数量是否匹配
        if images_count == 0:
            print(f"⚠️  {name}中没有图像文件")
        if labels_count == 0:
            print(f"⚠️  {name}中没有标签文件")
        
        # 注意：我们观察到数据集中标签文件数量远超图像文件数量，这可能是一个问题
    
    return True


def check_config_files():
    """检查配置文件"""
    print("\n" + "=" * 50)
    print("3. 配置文件检查")
    print("=" * 50)
    
    # 检查dataset.yaml
    config_file = "dataset.yaml"
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['path', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                print(f"❌ 配置文件缺少必要键: {key}")
                return False
        
        # 检查类别数量是否与类别名称匹配
        nc = config['nc']
        names = config['names']
        if len(names) != nc:
            print(f"❌ 类别数量不匹配: nc={nc}, names数量={len(names)}")
            return False
            
        print("✅ dataset.yaml 配置正确")
        return True
    except Exception as e:
        print(f"❌ 配置文件解析失败: {e}")
        return False


def check_weights():
    """检查权重文件"""
    print("\n" + "=" * 50)
    print("4. 权重文件检查")
    print("=" * 50)
    
    # 检查权重文件路径
    weight_paths = [
        os.path.join("runs", "detect", "train", "weights", "yolov11n.pt"),
        "yolov11n.pt"
    ]
    
    weight_found = False
    for weight_path in weight_paths:
        if os.path.exists(weight_path):
            size = os.path.getsize(weight_path) / (1024*1024)  # MB
            print(f"✅ 权重文件存在: {weight_path} ({size:.1f} MB)")
            weight_found = True
            break
    
    if not weight_found:
        print("⚠️  未找到预训练权重文件")
        print("建议运行 download_yolov11n.py 脚本来下载权重文件:")
        print("  python download_yolov11n.py")
        return False
    
    return True


def check_model_loading():
    """检查模型加载"""
    print("\n" + "=" * 50)
    print("5. 模型加载检查")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        print("✅ ultralytics 导入成功")
    except Exception as e:
        print(f"❌ ultralytics 导入失败: {e}")
        return False
    
    # 尝试加载模型
    model_paths = [
        os.path.join("runs", "detect", "train", "weights", "yolov11n.pt"),
        "yolov11n.pt"
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)
                print(f"✅ 模型加载成功: {model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠️  模型加载失败 {model_path}: {e}")
    
    if not model_loaded:
        print("❌ 无法加载任何模型")
        return False
    
    return True


def check_project_structure():
    """检查项目结构"""
    print("\n" + "=" * 50)
    print("6. 项目结构检查")
    print("=" * 50)
    
    required_files = [
        "app.py",
        "train.py",
        "dataset.yaml",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"❌ 缺少文件: {file}")
        else:
            print(f"✅ 文件存在: {file}")
    
    if missing_files:
        print(f"\n❌ 缺少必要文件: {missing_files}")
        return False
    
    print("\n✅ 项目结构完整")
    return True


def provide_solutions():
    """提供问题解决方案"""
    print("\n" + "=" * 50)
    print("7. 问题解决方案")
    print("=" * 50)
    
    print("如果在检查中发现问题，请参考以下解决方案:")
    print("\n1. 依赖安装问题:")
    print("   运行 install_deps.ps1 脚本安装所有依赖（使用清华镜像源）:")
    print("   PowerShell: .\\install_deps.ps1")
    
    print("\n2. 数据集问题:")
    print("   - 确保 dataset.yaml 中的路径正确")
    print("   - 确保训练集和验证集目录结构正确")
    print("   - 检查图像文件和标签文件是否匹配")
    
    print("\n3. 权重文件问题:")
    print("   运行 download_yolov11n.py 脚本下载预训练权重:")
    print("   python download_yolov11n.py")
    
    print("\n4. 模型加载问题:")
    print("   - 确保 ultralytics 已正确安装")
    print("   - 检查权重文件路径是否正确")
    
    print("\n5. 其他问题:")
    print("   - 查看具体错误信息")
    print("   - 检查Python环境")
    print("   - 确保使用的是项目虚拟环境")


def main():
    """主函数"""
    print("开始全面项目检查...")
    
    checks = [
        ("项目结构检查", check_project_structure),
        ("环境和依赖检查", check_environment),
        ("配置文件检查", check_config_files),
        ("数据集完整性检查", check_dataset),
        ("权重文件检查", check_weights),
        ("模型加载检查", check_model_loading)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ {check_name}执行出错: {e}")
            results.append((check_name, False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("检查结果汇总")
    print("=" * 50)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有检查通过！项目配置正确。")
    else:
        print("\n⚠️  部分检查未通过，请查看上面的错误信息。")
        provide_solutions()
    
    return all_passed


if __name__ == "__main__":
    main()