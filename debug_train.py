import argparse
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_train():
    """
    调试训练过程，打印关键信息
    """
    print("开始调试训练过程...")
    
    # 检查数据集配置文件
    dataset_yaml = 'dataset.yaml'
    if os.path.exists(dataset_yaml):
        print(f"[INFO] 找到数据集配置文件: {dataset_yaml}")
        with open(dataset_yaml, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"[INFO] 数据集配置内容预览:\n{content[:200]}...")
    else:
        print(f"[ERROR] 未找到数据集配置文件: {dataset_yaml}")
        return False
        
    # 检查数据集目录
    dataset_path = 'datasets/power_safety'
    if os.path.exists(dataset_path):
        print(f"[INFO] 找到数据集目录: {dataset_path}")
        for subdir in ['train', 'val']:
            sub_path = os.path.join(dataset_path, subdir)
            if os.path.exists(sub_path):
                print(f"[INFO] 找到子目录: {sub_path}")
                for subsubdir in ['images', 'labels']:
                    subsub_path = os.path.join(sub_path, subsubdir)
                    if os.path.exists(subsub_path):
                        file_count = len([f for f in os.listdir(subsub_path) if f.endswith(('.jpg', '.jpeg', '.png', '.txt'))])
                        print(f"[INFO] 找到子目录: {subsub_path} (包含 {file_count} 个文件)")
                    else:
                        print(f"[WARNING] 未找到子目录: {subsub_path}")
            else:
                print(f"[ERROR] 未找到子目录: {sub_path}")
                return False
    else:
        print(f"[ERROR] 未找到数据集目录: {dataset_path}")
        return False
        
    # 检查模型文件
    model_file = 'yolo11n.pt'
    if os.path.exists(model_file):
        size = os.path.getsize(model_file) / (1024*1024)  # MB
        print(f"[INFO] 找到模型文件: {model_file} ({size:.1f} MB)")
    else:
        print(f"[WARNING] 未找到预训练模型: {model_file}，将从头开始训练")
        
    print("[INFO] 所有检查通过，准备开始训练...")
    return True

if __name__ == '__main__':
    success = debug_train()
    if success:
        print("[SUCCESS] 调试检查完成，可以开始训练")
    else:
        print("[FAILURE] 调试检查失败，请检查上述错误")
        sys.exit(1)