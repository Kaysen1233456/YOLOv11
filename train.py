"""
训练脚本（基于 ultralytics YOLO API，适用于 YOLOv11）
可直接运行：python train.py

说明：
- 使用模型: yolo11l.pt
- 数据: dataset.yaml
- epochs=100, imgsz=640, batch=16
训练日志会输出到 runs/ 下（ultralytics 默认），最好的权重位于 runs/detect/train/weights/best.pt
"""

import argparse
from ultralytics import YOLO
import os
import torch
import matplotlib.pyplot as plt
import shutil
from pathlib import Path


def train_model(args):
    """
    训练YOLO模型
    
    Args:
        args: 命令行参数
        
    Returns:
        model: 训练后的模型
        results: 训练结果
    """
    # 确保输出目录存在
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'train'))
    os.makedirs(project_dir, exist_ok=True)

    # 加载模型（如果本地没有，会自动从 ultralytics 下载）
    model_name = args.model
    print(f"使用模型: {model_name}")
    
    # 检查权重文件是否存在
    if os.path.exists(model_name):
        print(f"从本地路径加载模型: {model_name}")
    else:
        # 尝试在项目目录中查找权重文件
        project_model_path = os.path.join(os.path.dirname(__file__), model_name)
        if os.path.exists(project_model_path):
            model_name = project_model_path
            print(f"从项目路径加载模型: {model_name}")
        else:
            # 再次检查是否有yolo11l.pt文件可以使用
            yolo11l_path = os.path.join(os.path.dirname(__file__), 'yolo11l.pt')
            if os.path.exists(yolo11l_path):
                model_name = yolo11l_path
                print(f"使用替代权重文件: {model_name}")
            else:
                print(f"警告: 未找到指定的模型文件: {model_name}，将尝试从Ultralytics下载")

    model = YOLO(model_name)

    # 调用 ultralytics 的 train 接口
    print("开始训练... 参数：", args)
    results = model.train(data=args.data,
                         epochs=args.epochs,
                         imgsz=args.imgsz,
                         batch=args.batch,
                         project=os.path.join(os.path.dirname(__file__), 'runs', 'detect'),
                         name='train',
                         exist_ok=True)
                         
    print("训练完成。")

    return model, results


def validate_model(model, data_cfg='dataset.yaml'):
    """
    验证训练后的模型性能
    
    Args:
        model: 训练后的模型
        data_cfg: 数据集配置文件路径
    """
    print("开始验证模型...")
    metrics = model.val(data=data_cfg)
    print("验证完成。")
    return metrics


def plot_training_results():
    """
    绘制训练过程中的损失和指标曲线
    """
    print("绘制训练结果图表...")
    
    # 查找训练结果目录
    results_dir = os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'train')
    results_file = os.path.join(results_dir, 'results.csv')
    
    if not os.path.exists(results_file):
        print("警告: 未找到训练结果文件 results.csv")
        return
    
    # 读取训练结果
    import pandas as pd
    try:
        results = pd.read_csv(results_file)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练过程指标变化', fontsize=16)
        
        # 损失曲线
        if 'train/box_loss' in results.columns:
            axes[0, 0].plot(results['train/box_loss'], label='训练框损失')
            axes[0, 0].set_title('框损失')
            axes[0, 0].legend()
            
        if 'train/cls_loss' in results.columns:
            axes[0, 1].plot(results['train/cls_loss'], label='训练分类损失')
            axes[0, 1].set_title('分类损失')
            axes[0, 1].legend()
            
        if 'train/dfl_loss' in results.columns:
            axes[1, 0].plot(results['train/dfl_loss'], label='训练分布焦点损失')
            axes[1, 0].set_title('分布焦点损失')
            axes[1, 0].legend()
            
        # mAP曲线
        if 'metrics/mAP50(B)' in results.columns:
            axes[1, 1].plot(results['metrics/mAP50(B)'], label='mAP50')
            axes[1, 1].set_title('mAP50')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, 'training_curves.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"训练曲线图已保存至: {plot_path}")
    except Exception as e:
        print(f"绘制图表时出错: {e}")


def copy_best_model():
    """
    将训练得到的最佳权重复制到指定位置
    """
    source_path = os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'train', 'weights', 'best.pt')
    target_dir = os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'train', 'weights')
    
    if os.path.exists(source_path):
        print(f"最佳模型权重已保存至: {source_path}")
    else:
        print("警告: 未找到最佳权重文件 best.pt")
        # 查看weights目录下的所有文件
        weights_dir = os.path.dirname(source_path)
        if os.path.exists(weights_dir):
            files = os.listdir(weights_dir)
            print(f"weights目录下可用文件: {files}")


def detect_device():
    """
    自动检测可用设备(CUDA或CPU)
    
    Returns:
        device: 可用设备 ('cuda' 或 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"使用GPU进行训练: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("使用CPU进行训练")
    return device


def main(args):
    # 自动检测设备
    device = detect_device()
    
    # 训练模型
    model, results = train_model(args)
    
    # 验证模型
    metrics = validate_model(model, args.data)
    print(f"验证结果 - mAP50: {metrics.box.map50}, mAP50-95: {metrics.box.map}")
    
    # 绘制训练结果
    plot_training_results()
    
    # 复制最佳模型
    copy_best_model()
    
    print("训练流程全部完成。最佳权重位于 runs/detect/train/weights/best.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv11 model for power safety detection')
    parser.add_argument('--model', type=str, default='yolo11l.pt', help='模型文件或型号（默认 yolo11l.pt）')
    parser.add_argument('--data', type=str, default='dataset.yaml', help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图片尺寸')
    parser.add_argument('--batch', type=int, default=16, help='batch 大小')

    args = parser.parse_args()
    main(args)