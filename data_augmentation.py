"""
数据增强脚本
功能：
1. 自动检测数据集中的样本分布
2. 对样本较少的类别进行数据增强
3. 生成增强后的数据集
4. 更新标签文件

使用方法：python data_augmentation.py
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
import shutil
import yaml
from collections import defaultdict
import argparse


class DataAugmenter:
    def __init__(self, data_path='datasets/power_safety', output_path='datasets/augmented_power_safety'):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.train_images_path = self.data_path / 'train' / 'images'
        self.train_labels_path = self.data_path / 'train' / 'labels'
        self.val_images_path = self.data_path / 'val' / 'images'
        self.val_labels_path = self.data_path / 'val' / 'labels'
        
        # 类别名称
        self.class_names = {
            0: 'person',
            1: 'helmet_person',
            2: 'insulated_gloves',
            3: 'safety_belt',
            4: 'power_pole',
            5: 'voltage_tester',
            6: 'work_clothes'
        }
        
        # 目标样本数量
        self.target_count = 500
        
        # 创建输出目录
        self.create_output_dirs()
    
    def create_output_dirs(self):
        """创建输出目录结构"""
        dirs = [
            self.output_path / 'train' / 'images',
            self.output_path / 'train' / 'labels',
            self.output_path / 'val' / 'images',
            self.output_path / 'val' / 'labels'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def analyze_dataset(self):
        """分析数据集中的类别分布"""
        print("分析数据集类别分布...")
        
        class_counts = defaultdict(int)
        image_class_mapping = {}
        
        # 统计训练集
        for label_file in self.train_labels_path.glob('*.txt'):
            image_name = label_file.stem
            image_class_mapping[image_name] = []
            
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
                    image_class_mapping[image_name].append(class_id)
        
        # 统计验证集
        for label_file in self.val_labels_path.glob('*.txt'):
            image_name = label_file.stem
            image_class_mapping[image_name] = []
            
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
                    image_class_mapping[image_name].append(class_id)
        
        print("类别分布统计：")
        for class_id, count in sorted(class_counts.items()):
            class_name = self.class_names.get(class_id, f'unknown_{class_id}')
            print(f"  {class_id}: {class_name} - {count} 个样本")
        
        return class_counts, image_class_mapping
    
    def calculate_augmentation_ratios(self, class_counts):
        """计算每个类别的增强比例"""
        augmentation_ratios = {}
        
        for class_id, count in class_counts.items():
            if count < self.target_count:
                ratio = min(self.target_count // count, 5)  # 最多增强5倍
                augmentation_ratios[class_id] = ratio
                print(f"类别 {class_id} ({self.class_names[class_id]}) 需要增强 {ratio} 倍")
        
        return augmentation_ratios
    
    def augment_image(self, image, bboxes, augmentation_type='auto'):
        """对图像进行增强"""
        augmented_image = image.copy()
        augmented_bboxes = bboxes.copy()
        
        height, width = image.shape[:2]
        
        if augmentation_type == 'flip_horizontal':
            # 水平翻转
            augmented_image = cv2.flip(image, 1)
            for i, bbox in enumerate(augmented_bboxes):
                x_center, y_center, w, h = bbox[1:5]
                augmented_bboxes[i][1] = 1 - x_center  # 翻转x坐标
        
        elif augmentation_type == 'rotation':
            # 随机旋转
            angle = random.uniform(-15, 15)
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            # 更新边界框（简化版本，实际应该更精确）
            # 这里只做简单处理，实际应用中应该使用更精确的边界框变换
        
        elif augmentation_type == 'scale':
            # 随机缩放
            scale = random.uniform(0.8, 1.2)
            new_width = int(width * scale)
            new_height = int(height * scale)
            augmented_image = cv2.resize(image, (new_width, new_height))
            
            # 缩放回原始尺寸
            augmented_image = cv2.resize(augmented_image, (width, height))
        
        elif augmentation_type == 'brightness':
            # 调整亮度
            brightness = random.uniform(0.8, 1.2)
            augmented_image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        elif augmentation_type == 'noise':
            # 添加噪声
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            augmented_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        elif augmentation_type == 'auto':
            # 自动组合增强
            transforms = ['flip_horizontal', 'rotation', 'brightness']
            selected_transform = random.choice(transforms)
            return self.augment_image(image, bboxes, selected_transform)
        
        return augmented_image, augmented_bboxes
    
    def process_single_image(self, image_path, label_path, output_dir, image_suffix):
        """处理单张图像和标签"""
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"警告: 无法读取图像 {image_path}")
            return
        
        # 读取标签
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                bbox = [class_id] + [float(x) for x in parts[1:]]
                bboxes.append(bbox)
        
        # 应用增强
        augmentation_types = ['flip_horizontal', 'rotation', 'scale', 'brightness', 'noise']
        for i, aug_type in enumerate(augmentation_types):
            try:
                augmented_image, augmented_bboxes = self.augment_image(image, bboxes, aug_type)
                
                # 保存增强后的图像
                new_image_name = f"{image_path.stem}_{aug_type}_{i}{image_path.suffix}"
                new_image_path = output_dir / 'images' / new_image_name
                cv2.imwrite(str(new_image_path), augmented_image)
                
                # 保存增强后的标签
                new_label_name = f"{image_path.stem}_{aug_type}_{i}.txt"
                new_label_path = output_dir / 'labels' / new_label_name
                
                with open(new_label_path, 'w') as f:
                    for bbox in augmented_bboxes:
                        f.write(f"{' '.join(map(str, bbox))}\n")
                
            except Exception as e:
                print(f"增强 {aug_type} 处理失败: {e}")
    
    def augment_dataset(self, augmentation_ratios):
        """增强整个数据集"""
        print("开始数据增强...")
        
        # 复制原始数据
        self.copy_original_data()
        
        # 对需要增强的类别进行处理
        for class_id, ratio in augmentation_ratios.items():
            print(f"\n增强类别 {class_id} ({self.class_names[class_id]})，目标比例: {ratio}")
            
            # 找到包含该类别的图像
            images_to_augment = []
            
            for label_file in self.train_labels_path.glob('*.txt'):
                with open(label_file, 'r') as f:
                    labels = f.readlines()
                    # 检查是否包含目标类别
                    if any(int(line.strip().split()[0]) == class_id for line in labels):
                        image_file = self.train_images_path / f"{label_file.stem}{self.get_image_extension(label_file)}"
                        if image_file.exists():
                            images_to_augment.append((image_file, label_file))
            
            # 对每个图像应用增强
            for image_file, label_file in images_to_augment:
                for _ in range(ratio):
                    try:
                        self.process_single_image(
                            image_file, 
                            label_file, 
                            self.output_path / 'train',
                            f"_aug_{class_id}"
                        )
                    except Exception as e:
                        print(f"处理 {image_file.name} 时出错: {e}")
    
    def copy_original_data(self):
        """复制原始数据到新目录"""
        print("复制原始数据...")
        
        # 复制训练数据
        for image_file in self.train_images_path.glob('*.jpg'):
            shutil.copy2(image_file, self.output_path / 'train' / 'images')
        
        for label_file in self.train_labels_path.glob('*.txt'):
            shutil.copy2(label_file, self.output_path / 'train' / 'labels')
        
        # 复制验证数据
        for image_file in self.val_images_path.glob('*.jpg'):
            shutil.copy2(image_file, self.output_path / 'val' / 'images')
        
        for label_file in self.val_labels_path.glob('*.txt'):
            shutil.copy2(label_file, self.output_path / 'val' / 'labels')
    
    def get_image_extension(self, label_file):
        """获取对应图像文件的扩展名"""
        for ext in ['.jpg', '.jpeg', '.png']:
            image_file = self.train_images_path / f"{label_file.stem}{ext}"
            if image_file.exists():
                return ext
        return '.jpg'  # 默认扩展名
    
    def create_augmented_config(self):
        """创建增强数据集的配置文件"""
        config = {
            'path': str(self.output_path),
            'nc': 7,
            'names': self.class_names,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'val/images'
        }
        
        config_path = self.output_path / 'augmented_dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"增强数据集配置文件已保存至: {config_path}")
        return config_path
    
    def main(self):
        """主函数"""
        print("=" * 60)
        print("YOLOv11 电力安全检测 - 数据增强")
        print("=" * 60)
        
        # 分析数据集
        class_counts, image_class_mapping = self.analyze_dataset()
        
        # 计算增强比例
        augmentation_ratios = self.calculate_augmentation_ratios(class_counts)
        
        if not augmentation_ratios:
            print("所有类别的样本数量已满足要求，无需增强")
            return
        
        # 执行数据增强
        self.augment_dataset(augmentation_ratios)
        
        # 创建配置文件
        config_path = self.create_augmented_config()
        
        # 统计增强结果
        self.print_augmentation_summary()
        
        print("=" * 60)
        print("数据增强完成！")
        print(f"增强数据集位置: {self.output_path}")
        print(f"配置文件: {config_path}")
        print("=" * 60)
    
    def print_augmentation_summary(self):
        """打印增强结果总结"""
        print("\n增强结果总结：")
        
        # 统计原始数据
        original_counts = defaultdict(int)
        for label_file in self.train_labels_path.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    original_counts[class_id] += 1
        
        for label_file in self.val_labels_path.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    original_counts[class_id] += 1
        
        # 统计增强后数据
        augmented_counts = defaultdict(int)
        for label_file in (self.output_path / 'train' / 'labels').glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    augmented_counts[class_id] += 1
        
        for label_file in (self.output_path / 'val' / 'labels').glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    augmented_counts[class_id] += 1
        
        print("类别样本数量对比：")
        for class_id in sorted(set(original_counts.keys()) | set(augmented_counts.keys())):
            class_name = self.class_names.get(class_id, f'unknown_{class_id}')
            original = original_counts.get(class_id, 0)
            augmented = augmented_counts.get(class_id, 0)
            increase = augmented - original
            print(f"  {class_id}: {class_name:15} {original:4d} -> {augmented:4d} (+{increase:4d})")


def main():
    parser = argparse.ArgumentParser(description='YOLOv11电力安全检测数据增强')
    parser.add_argument('--data_path', type=str, default='datasets/power_safety', help='原始数据集路径')
    parser.add_argument('--output_path', type=str, default='datasets/augmented_power_safety', help='增强数据集输出路径')
    parser.add_argument('--target_count', type=int, default=500, help='每个类别的目标样本数量')
    
    args = parser.parse_args()
    
    augmenter = DataAugmenter(args.data_path, args.output_path)
    augmenter.target_count = args.target_count
    augmenter.main()


if __name__ == '__main__':
    main()
