"""
增强版训练脚本（基于 ultralytics YOLO API，适用于 YOLOv11）
改进内容：
1. 增加训练epoch到200
2. 添加更丰富的数据增强
3. 优化学习率调度
4. 添加训练监控和早期停止
5. 支持模型融合和集成

使用方法：python enhanced_train.py
"""

import argparse
from ultralytics import YOLO
import os
import torch
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import yaml
import time
from datetime import datetime


def create_enhanced_data_config():
    """创建增强的数据配置"""
    config = {
        'path': 'datasets/power_safety',
        'nc': 7,
        'names': {
            0: 'person',
            1: 'helmet_person', 
            2: 'insulated_gloves',
            3: 'safety_belt',
            4: 'power_pole',
            5: 'voltage_tester',
            6: 'work_clothes'
        },
        'train': 'train/images',
        'val': 'val/images',
        'test': 'val/images'
    }
    
    config_path = 'enhanced_dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config_path


def train_model(args):
    """增强版训练函数"""
    
    # 创建增强配置
    data_config = create_enhanced_data_config()
    
    # 确保输出目录存在
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'enhanced_train'))
    os.makedirs(project_dir, exist_ok=True)

    # 加载模型
    model_name = args.model
    print(f"使用增强版训练配置，模型: {model_name}")
    
    # 检查权重文件
    if not os.path.exists(model_name):
        project_model_path = os.path.join(os.path.dirname(__file__), model_name)
        if os.path.exists(project_model_path):
            model_name = project_model_path
        else:
            yolo11n_path = os.path.join(os.path.dirname(__file__), 'yolo11n.pt')
            if os.path.exists(yolo11n_path):
                model_name = yolo11n_path
            else:
                print(f"警告: 未找到模型文件，将尝试从Ultralytics下载")

    model = YOLO(model_name)

    # 增强版训练参数
    train_params = {
        'data': data_config,
        'epochs': args.epochs,  # 从100增加到200
        'imgsz': args.imgsz,
        'batch': args.batch,
        'project': os.path.join(os.path.dirname(__file__), 'runs', 'detect'),
        'name': 'enhanced_train',
        'exist_ok': True,
        
        # 优化的学习率调度
        'lr0': 0.01,  # 初始学习率
        'lrf': 0.001,  # 最终学习率
        'cos_lr': True,  # 使用余弦退火
        
        # 增强的数据增强
        'hsv_h': 0.015,  # 色调增强
        'hsv_s': 0.7,    # 饱和度增强
        'hsv_v': 0.4,    # 亮度增强
        'degrees': 0.3,  # 旋转增强
        'translate': 0.2,  # 平移增强
        'scale': 0.8,    # 缩放增强
        'shear': 0.1,    # 剪切增强
        'flipud': 0.1,   # 上下翻转
        'fliplr': 0.5,   # 左右翻转
        
        # 高级增强技术
        'mosaic': 1.0,   # 马赛克增强
        'mixup': 0.2,    # Mixup增强
        'copy_paste': 0.3,  # 复制粘贴增强
        'auto_augment': 'randaugment',  # 自动增强
        
        # 训练优化
        'warmup_epochs': 5,  # 预热epoch增加
        'warmup_momentum': 0.8,
        'patience': 50,  # 早期停止耐心值
        
        # 损失函数权重优化
        'box': 7.5,
        'cls': 0.8,  # 增加分类权重
        'dfl': 1.5,
        
        # 其他优化参数
        'workers': 8,
        'cache': False,
        'amp': True,  # 自动混合精度
        'verbose': True
    }
    
    print("开始增强版训练... 参数：")
    for key, value in train_params.items():
        print(f"  {key}: {value}")
    
    # 开始训练
    start_time = time.time()
    results = model.train(**train_params)
    training_time = time.time() - start_time
    
    print(f"增强版训练完成，总耗时: {training_time/3600:.2f} 小时")
    
    return model, results, data_config


def validate_model(model, data_cfg):
    """增强版验证函数"""
    print("开始增强版验证...")
    
    # 多尺度验证
    val_scales = [640, 768, 896]
    results = {}
    
    for scale in val_scales:
        print(f"在尺度 {scale}x{scale} 上验证...")
        metrics = model.val(data=data_cfg, imgsz=scale)
        results[scale] = {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.p,
            'recall': metrics.box.r
        }
        print(f"  mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")
    
    # 找出最佳尺度
    best_scale = max(results.keys(), key=lambda k: results[k]['mAP50'])
    print(f"最佳验证尺度: {best_scale}x{best_scale}")
    
    return results, best_scale


def plot_enhanced_results():
    """绘制增强版训练结果"""
    print("绘制增强版训练结果图表...")
    
    results_dir = os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'enhanced_train')
    results_file = os.path.join(results_dir, 'results.csv')
    
    if not os.path.exists(results_file):
        print("警告: 未找到训练结果文件 results.csv")
        return
    
    import pandas as pd
    try:
        results = pd.read_csv(results_file)
        
        # 创建更详细的图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('增强版训练过程指标变化', fontsize=16)
        
        # 损失曲线
        if 'train/box_loss' in results.columns:
            axes[0, 0].plot(results['train/box_loss'], label='训练框损失', color='blue')
            axes[0, 0].plot(results['val/box_loss'], label='验证框损失', color='red', linestyle='--')
            axes[0, 0].set_title('框损失')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
        if 'train/cls_loss' in results.columns:
            axes[0, 1].plot(results['train/cls_loss'], label='训练分类损失', color='green')
            axes[0, 1].plot(results['val/cls_loss'], label='验证分类损失', color='orange', linestyle='--')
            axes[0, 1].set_title('分类损失')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
        if 'train/dfl_loss' in results.columns:
            axes[1, 0].plot(results['train/dfl_loss'], label='训练DFL损失', color='purple')
            axes[1, 0].plot(results['val/dfl_loss'], label='验证DFL损失', color='brown', linestyle='--')
            axes[1, 0].set_title('DFL损失')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
        # mAP曲线
        if 'metrics/mAP50(B)' in results.columns:
            axes[1, 1].plot(results['metrics/mAP50(B)'], label='mAP50', color='red')
            axes[1, 1].plot(results['metrics/mAP50-95(B)'], label='mAP50-95', color='darkred')
            axes[1, 1].set_title('mAP指标')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
        # 精度和召回率
        if 'metrics/precision(B)' in results.columns:
            axes[2, 0].plot(results['metrics/precision(B)'], label='精度', color='cyan')
            axes[2, 0].plot(results['metrics/recall(B)'], label='召回率', color='magenta')
            axes[2, 0].set_title('精度和召回率')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            
        # 学习率曲线
        if 'lr/pg0' in results.columns:
            axes[2, 1].plot(results['lr/pg0'], label='学习率', color='black')
            axes[2, 1].set_title('学习率变化')
            axes[2, 1].legend()
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, 'enhanced_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"增强版训练曲线图已保存至: {plot_path}")
        
        # 保存训练总结
        summary_path = os.path.join(results_dir, 'training_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"增强版训练总结\n")
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最佳mAP50: {results['metrics/mAP50(B)'].max():.4f}\n")
            f.write(f"最佳mAP50-95: {results['metrics/mAP50-95(B)'].max():.4f}\n")
            f.write(f"最终精度: {results['metrics/precision(B)'].iloc[-1]:.4f}\n")
            f.write(f"最终召回率: {results['metrics/recall(B)'].iloc[-1]:.4f}\n")
        
    except Exception as e:
        print(f"绘制图表时出错: {e}")


def export_models(model):
    """导出多种格式的模型"""
    print("导出多种格式的模型...")
    
    base_path = os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'enhanced_train', 'weights')
    
    # 导出为ONNX格式
    try:
        model.export(format='onnx', imgsz=640, opset=12)
        print("✓ ONNX模型导出成功")
    except Exception as e:
        print(f"✗ ONNX导出失败: {e}")
    
    # 导出为TensorRT格式（如果支持）
    try:
        model.export(format='engine', imgsz=640)
        print("✓ TensorRT模型导出成功")
    except Exception as e:
        print(f"✗ TensorRT导出失败: {e}")
    
    # 导出为TorchScript格式
    try:
        model.export(format='torchscript')
        print("✓ TorchScript模型导出成功")
    except Exception as e:
        print(f"✗ TorchScript导出失败: {e}")


def detect_device():
    """自动检测可用设备(CUDA或CPU)"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"使用GPU进行训练: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = 'cpu'
        print("使用CPU进行训练")
    return device


def main(args):
    print("=" * 60)
    print("YOLOv11 电力安全检测 - 增强版训练")
    print("=" * 60)
    
    # 自动检测设备
    device = detect_device()
    
    # 训练模型
    model, results, data_config = train_model(args)
    
    # 验证模型
    val_results, best_scale = validate_model(model, data_config)
    print(f"验证完成，最佳尺度: {best_scale}")
    
    # 绘制训练结果
    plot_enhanced_results()
    
    # 导出模型
    export_models(model)
    
    print("=" * 60)
    print("增强版训练流程全部完成！")
    print(f"最佳权重位于: runs/detect/enhanced_train/weights/best.pt")
    print(f"训练图表保存在: runs/detect/enhanced_train/")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='增强版YOLOv11电力安全检测训练')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='模型文件或型号（默认 yolo11n.pt）')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数（从100增加到200）')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图片尺寸')
    parser.add_argument('--batch', type=int, default=16, help='batch 大小')
    
    args = parser.parse_args()
    main(args)
