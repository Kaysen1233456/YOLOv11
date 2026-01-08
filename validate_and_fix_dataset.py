"""
数据集验证和修复脚本

此脚本用于：
1. 验证数据集中图像和标签文件是否一一对应
2. 修复不匹配的问题
3. 提供详细的统计信息
"""

import os
import yaml
from pathlib import Path
import argparse


def check_dataset(images_dir, labels_dir):
    """
    检查数据集中的图像和标签文件是否一一对应
    
    Args:
        images_dir: 图像目录路径
        labels_dir: 标签目录路径
    
    Returns:
        dict: 检查结果统计信息
    """
    # 获取所有图像文件
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(Path(images_dir).glob(ext))
    
    # 获取所有标签文件
    labels = set(os.listdir(labels_dir))
    
    # 统计匹配和不匹配的数量
    matched = 0
    unmatched = 0
    unmatched_files = []
    
    # 检查每个图像文件是否有对应的标签文件
    for img in images:
        label_file = f"{img.stem}.txt"
        if label_file in labels:
            matched += 1
        else:
            unmatched += 1
            unmatched_files.append(str(img))
    
    # 计算不匹配率
    total_images = len(images)
    unmatched_rate = (unmatched / total_images * 100) if total_images > 0 else 0
    
    # 输出统计信息
    print(f"数据集检查结果:")
    print(f"  总图像文件数: {total_images}")
    print(f"  匹配的文件对: {matched}")
    print(f"  不匹配的文件数: {unmatched}")
    print(f"  不匹配率: {unmatched_rate:.2f}%")
    
    if unmatched_files:
        print(f"  不匹配的文件列表:")
        for file in unmatched_files[:10]:  # 只显示前10个
            print(f"    {file}")
        if len(unmatched_files) > 10:
            print(f"    ... 还有 {len(unmatched_files) - 10} 个文件")
    
    return {
        'total_images': total_images,
        'matched': matched,
        'unmatched': unmatched,
        'unmatched_rate': unmatched_rate,
        'unmatched_files': unmatched_files
    }


def get_file_pairs(images_dir, labels_dir):
    """
    获取图像和标签文件对
    
    Args:
        images_dir: 图像目录
        labels_dir: 标签目录
    
    Returns:
        tuple: (匹配的文件对列表, 只有图像没有标签的文件, 只有标签没有图像的文件)
    """
    # 获取所有图像文件
    image_files = {}
    for ext in ['.jpg', '.jpeg', '.png']:
        for file in Path(images_dir).glob(f'*{ext}'):
            stem = file.stem
            image_files[stem] = file
    
    # 获取所有标签文件
    label_files = {}
    for file in Path(labels_dir).glob('*.txt'):
        stem = file.stem
        label_files[stem] = file
    
    # 找到匹配的文件对
    matched_pairs = []
    image_only = []
    label_only = []
    
    # 检查图像有但标签没有的文件
    for stem, image_path in image_files.items():
        if stem in label_files:
            matched_pairs.append((image_path, label_files[stem]))
        else:
            image_only.append(image_path)
    
    # 检查标签有但图像没有的文件
    for stem, label_path in label_files.items():
        if stem not in image_files:
            label_only.append(label_path)
    
    return matched_pairs, image_only, label_only


def validate_dataset_split(split_path, split_name):
    """
    验证单个数据集分割（训练集或验证集）
    
    Args:
        split_path: 数据集分割路径
        split_name: 数据集分割名称（'train' 或 'val'）
    
    Returns:
        dict: 验证结果
    """
    print(f"\n验证{split_name}集...")
    
    images_dir = os.path.join(split_path, 'images')
    labels_dir = os.path.join(split_path, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"❌ {split_name}集图像目录不存在: {images_dir}")
        return None
        
    if not os.path.exists(labels_dir):
        print(f"❌ {split_name}集标签目录不存在: {labels_dir}")
        return None
    
    # 使用新的check_dataset函数进行检查
    check_result = check_dataset(images_dir, labels_dir)
    
    # 获取文件对信息（为了保持向后兼容）
    matched_pairs, image_only, label_only = get_file_pairs(images_dir, labels_dir)
    
    result = {
        'total_images': check_result['total_images'],
        'total_labels': len(os.listdir(labels_dir)),
        'matched_pairs': check_result['matched'],
        'image_only': check_result['unmatched'],
        'label_only': len(label_only),
        'image_only_files': [Path(images_dir) / Path(f).name for f in check_result['unmatched_files']],
        'label_only_files': label_only
    }
    
    print(f"  总图像文件数: {result['total_images']}")
    print(f"  总标签文件数: {result['total_labels']}")
    print(f"  匹配的文件对: {result['matched_pairs']}")
    print(f"  只有图像没有标签: {result['image_only']}")
    print(f"  只有标签没有图像: {result['label_only']}")
    
    if result['image_only'] > 0:
        print(f"  警告: 发现 {result['image_only']} 个只有图像没有标签的文件")
        
    if result['label_only'] > 0:
        print(f"  警告: 发现 {result['label_only']} 个只有标签没有图像的文件")
    
    return result


def fix_dataset_mismatch(split_path, split_name, action='report'):
    """
    修复数据集不匹配问题
    
    Args:
        split_path: 数据集分割路径
        split_name: 数据集分割名称
        action: 操作类型 ('report' 仅报告, 'remove_image_only' 删除只有图像的文件, 
               'remove_label_only' 删除只有标签的文件)
    """
    print(f"\n处理{split_name}集不匹配问题...")
    
    images_dir = os.path.join(split_path, 'images')
    labels_dir = os.path.join(split_path, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        return
    
    matched_pairs, image_only, label_only = get_file_pairs(images_dir, labels_dir)
    
    if action == 'remove_image_only' and image_only:
        print(f"  删除 {len(image_only)} 个只有图像没有标签的文件...")
        for image_file in image_only:
            try:
                os.remove(image_file)
                print(f"    已删除: {image_file}")
            except Exception as e:
                print(f"    删除失败 {image_file}: {e}")
    
    if action == 'remove_label_only' and label_only:
        print(f"  删除 {len(label_only)} 个只有标签没有图像的文件...")
        for label_file in label_only:
            try:
                os.remove(label_file)
                print(f"    已删除: {label_file}")
            except Exception as e:
                print(f"    删除失败 {label_file}: {e}")


def validate_dataset_config():
    """
    验证数据集配置文件
    
    Returns:
        tuple: (配置是否有效, 数据集根路径)
    """
    config_file = 'dataset.yaml'
    if not os.path.exists(config_file):
        print("❌ 找不到数据集配置文件 dataset.yaml")
        return False, None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        dataset_path = config.get('path')
        if not dataset_path:
            print("❌ 配置文件中未指定数据集路径")
            return False, None
            
        if not os.path.exists(dataset_path):
            print(f"❌ 数据集路径不存在: {dataset_path}")
            return False, None
            
        return True, dataset_path
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return False, None


def check_yolo_weights_compatibility():
    """
    检查YOLO权重文件兼容性
    
    Returns:
        bool: 是否兼容
    """
    print("\n检查YOLO权重文件兼容性...")
    
    # 检查权重文件
    weight_paths = [
        os.path.join("runs", "detect", "train", "weights", "yolov11n.pt"),
        "yolov11n.pt"
    ]
    
    weight_found = False
    for weight_path in weight_paths:
        if os.path.exists(weight_path):
            size = os.path.getsize(weight_path) / (1024*1024)  # MB
            print(f"✅ 找到权重文件: {weight_path} ({size:.1f} MB)")
            weight_found = True
            
            # 检查是否可以加载权重
            try:
                from ultralytics import YOLO
                model = YOLO(weight_path)
                print(f"✅ 权重文件加载成功")
                print(f"  模型任务: {model.task if hasattr(model, 'task') else 'unknown'}")
                print(f"  模型类别数: {len(model.names) if hasattr(model, 'names') else 'unknown'}")
                return True
            except Exception as e:
                print(f"❌ 权重文件加载失败: {e}")
                return False
    
    if not weight_found:
        print("⚠️  未找到预训练权重文件，将使用默认模型")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov11n.pt')
            print("✅ 默认模型加载成功")
            return True
        except Exception as e:
            print(f"❌ 默认模型加载失败: {e}")
            return False
    
    return True


def run_comprehensive_check():
    """
    运行综合检查
    
    Returns:
        bool: 检查是否通过
    """
    print("=" * 60)
    print("数据集和权重文件综合检查")
    print("=" * 60)
    
    # 1. 验证数据集配置
    is_valid, dataset_path = validate_dataset_config()
    if not is_valid:
        return False
    
    # 2. 验证训练集和验证集
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    
    train_result = validate_dataset_split(train_path, '训练')
    val_result = validate_dataset_split(val_path, '验证')
    
    if train_result is None or val_result is None:
        return False
    
    # 3. 检查权重文件兼容性
    weights_compatible = check_yolo_weights_compatibility()
    
    # 4. 汇总结果
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    total_issues = (train_result['image_only'] + train_result['label_only'] + 
                   val_result['image_only'] + val_result['label_only'])
    
    if total_issues > 0:
        print(f"⚠️  发现 {total_issues} 个数据集匹配问题:")
        print(f"  训练集:")
        print(f"    只有图像没有标签: {train_result['image_only']} 个")
        print(f"    只有标签没有图像: {train_result['label_only']} 个")
        print(f"  验证集:")
        print(f"    只有图像没有标签: {val_result['image_only']} 个")
        print(f"    只有标签没有图像: {val_result['label_only']} 个")
    else:
        print("✅ 数据集文件匹配良好")
    
    if weights_compatible:
        print("✅ 权重文件兼容性检查通过")
    else:
        print("❌ 权重文件兼容性检查失败")
    
    overall_success = (total_issues == 0) and weights_compatible
    if overall_success:
        print("\n🎉 所有检查通过！数据集和权重文件配置正确。")
    else:
        print("\n⚠️  存在问题需要处理。")
    
    return overall_success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据集验证和修复工具')
    parser.add_argument('--action', choices=['check', 'fix-image-only', 'fix-label-only', 'fix-all'], 
                       default='check', help='执行操作类型')
    parser.add_argument('--split', choices=['train', 'val', 'both'], 
                       default='both', help='要处理的数据集分割')
    
    args = parser.parse_args()
    
    if args.action == 'check':
        run_comprehensive_check()
        return
    
    # 验证配置
    is_valid, dataset_path = validate_dataset_config()
    if not is_valid:
        return
    
    # 根据参数执行修复操作
    splits_to_process = []
    if args.split == 'both':
        splits_to_process = ['train', 'val']
    else:
        splits_to_process = [args.split]
    
    for split_name in splits_to_process:
        split_path = os.path.join(dataset_path, split_name)
        if os.path.exists(split_path):
            if args.action == 'fix-image-only':
                fix_dataset_mismatch(split_path, split_name, 'remove_image_only')
            elif args.action == 'fix-label-only':
                fix_dataset_mismatch(split_path, split_name, 'remove_label_only')
            elif args.action == 'fix-all':
                fix_dataset_mismatch(split_path, split_name, 'remove_image_only')
                fix_dataset_mismatch(split_path, split_name, 'remove_label_only')


if __name__ == "__main__":
    main()


