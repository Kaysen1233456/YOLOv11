#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 电力安全检测 - 标准化训练脚本 (毕设项目唯一训练入口)

功能特性:
1. 默认使用 YOLOv11-Large 模型 (yolov11l.pt)
2. 针对 RTX 3060 优化的默认参数 (batch=4, workers=4, epochs=100)
3. 集成增强训练策略 (Mosaic, Mixup, 余弦退火学习率等)
4. 支持 debug 模式快速验证代码
5. 自动设备检测并显示显卡信息
6. 训练结束后自动验证并导出 ONNX 模型

作者: 毕设学生
日期: 2025-12-21
使用方法:
    - 本地训练 (RTX 3060): python main_train.py
    - 云端训练 (更大batch): python main_train.py --batch 16 --workers 8
    - 快速验证代码: python main_train.py --debug
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO
import yaml


def print_banner():
    """打印训练开始的横幅信息"""
    print("=" * 70)
    print("     YOLOv11 电力安全检测 - 标准化训练系统 (毕业设计专用)")
    print("=" * 70)
    print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def detect_and_print_device():
    """检测可用设备并打印详细的显卡信息"""
    print("\n🔍 正在检测计算设备...")
    print("-" * 70)
    
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_count = torch.cuda.device_count()
        
        print(f"✓ 检测到 {gpu_count} 个可用 GPU")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_properties = torch.cuda.get_device_properties(i)
            total_memory = gpu_properties.total_memory / 1024**3  # 转换为 GB
            
            print(f"\n  GPU {i}: {gpu_name}")
            print(f"    - 总显存: {total_memory:.2f} GB")
            print(f"    - CUDA 计算能力: {gpu_properties.major}.{gpu_properties.minor}")
            print(f"    - 多处理器数量: {gpu_properties.multi_processor_count}")
            
            # 检查是否是 RTX 3060
            if "3060" in gpu_name:
                print(f"    ✓ 已确认使用您的 RTX 3060 显卡!")
        
        # 显示当前 CUDA 版本
        print(f"\n  CUDA 版本: {torch.version.cuda}")
        print(f"  PyTorch 版本: {torch.__version__}")
        
    else:
        device = 'cpu'
        print("⚠️  未检测到可用 GPU，将使用 CPU 训练")
        print("   提示: CPU 训练速度会非常慢，建议使用 GPU")
    
    print("-" * 70)
    return device


def check_and_download_model(model_name='yolo11l.pt'):
    """检查模型文件是否存在，如不存在则自动下载
    
    Args:
        model_name: 模型文件名，默认为 yolo11l.pt (Large 模型)
    
    Returns:
        str: 模型文件的完整路径
    """
    print(f"\n📦 正在检查模型文件: {model_name}...")
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    model_path = script_dir / model_name
    
    # 也检查 weights 目录
    weights_dir = script_dir / "weights"
    weights_path = weights_dir / model_name
    
    if model_path.exists():
        print(f"✓ 在根目录找到模型文件: {model_path}")
        return str(model_path)
    elif weights_path.exists():
        print(f"✓ 在 weights/ 目录找到模型文件: {weights_path}")
        return str(weights_path)
    else:
        print(f"❌ 未找到 {model_name} 文件")
        print(f"⬇️  正在通过 ultralytics 自动下载 {model_name}...")
        print("   (首次下载可能需要几分钟，请耐心等待)")
        
        try:
            # ultralytics 会自动下载模型到缓存目录
            # 我们只需要传入模型名称即可
            return model_name
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            sys.exit(1)


def verify_dataset(data_config='dataset.yaml'):
    """验证数据集配置文件是否存在
    
    Args:
        data_config: 数据集配置文件路径
    
    Returns:
        str: 数据集配置文件的完整路径
    """
    print(f"\n📊 正在验证数据集配置: {data_config}...")
    
    script_dir = Path(__file__).parent
    data_path = script_dir / data_config
    
    if not data_path.exists():
        print(f"❌ 错误: 未找到数据集配置文件 {data_config}")
        print("   请确保 dataset.yaml 文件存在于项目根目录")
        sys.exit(1)
    
    # 读取并验证配置
    with open(data_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ 数据集配置验证成功")
    print(f"  - 类别数量: {config.get('nc', 'N/A')}")
    print(f"  - 数据集路径: {config.get('path', 'N/A')}")
    
    return str(data_path)


def train_model(args):
    """执行模型训练
    
    Args:
        args: 命令行参数对象
    """
    # 1. 检测设备
    device = detect_and_print_device()
    
    # 2. 检查并加载模型
    model_path = check_and_download_model(args.model)
    print(f"\n🚀 正在加载模型: {model_path}")
    model = YOLO(model_path)
    print(f"✓ 模型加载成功: {args.model}")
    
    # 3. 验证数据集
    data_config = verify_dataset(args.data)
    
    # 4. 设置训练参数
    print("\n⚙️  训练参数配置:")
    print("-" * 70)
    
    # 根据 debug 模式调整参数
    epochs = 1 if args.debug else args.epochs
    project_name = 'debug_test' if args.debug else 'main_train'
    
    # 构建训练参数字典
    train_params = {
        # ===== 基础参数 =====
        'data': data_config,              # 数据集配置文件
        'epochs': epochs,                 # 训练轮数
        'imgsz': args.imgsz,             # 输入图像尺寸
        'batch': args.batch,             # batch size
        'device': device,                # 设备 (cuda/cpu)
        
        # ===== 输出路径 =====
        'project': 'runs/detect',        # 训练结果保存目录
        'name': project_name,            # 实验名称
        'exist_ok': True,                # 允许覆盖已有结果
        
        # ===== 学习率调度策略 (余弦退火) =====
        'lr0': 0.01,                     # 初始学习率
        'lrf': 0.001,                    # 最终学习率 (lr0 * lrf)
        'cos_lr': True,                  # 使用余弦退火学习率调度
        
        # ===== 数据增强参数 =====
        # HSV 颜色空间增强
        'hsv_h': 0.015,                  # 色调抖动范围 (0-1)
        'hsv_s': 0.7,                    # 饱和度抖动范围 (0-1)
        'hsv_v': 0.4,                    # 亮度抖动范围 (0-1)
        
        # 几何变换增强
        'degrees': 0.3,                  # 旋转角度范围 (度)
        'translate': 0.2,                # 平移范围 (图像尺寸的比例)
        'scale': 0.8,                    # 缩放范围 (±)
        'shear': 0.1,                    # 剪切角度 (度)
        'flipud': 0.1,                   # 上下翻转概率
        'fliplr': 0.5,                   # 左右翻转概率
        
        # ===== 高级数据增强策略 =====
        'mosaic': 1.0,                   # Mosaic 增强概率 (图像拼接)
        'mixup': 0.2,                    # Mixup 增强概率 (图像混合)
        'copy_paste': 0.3,               # Copy-Paste 增强概率
        
        # ===== 训练优化参数 =====
        'optimizer': 'AdamW',            # 优化器 (AdamW 通常效果更好)
        'warmup_epochs': 5,              # 学习率预热轮数
        'warmup_momentum': 0.8,          # 预热阶段的动量
        'patience': 50,                  # 早停耐心值 (多少轮无改善则停止)
        
        # ===== 损失函数权重 =====
        'box': 7.5,                      # 边界框损失权重
        'cls': 0.8,                      # 分类损失权重
        'dfl': 1.5,                      # DFL 损失权重
        
        # ===== 其他参数 =====
        'workers': args.workers,         # 数据加载线程数
        'cache': False,                  # 不缓存图像到内存 (节省内存)
        'amp': True,                     # 使用自动混合精度训练 (加速+节省显存)
        'verbose': True,                 # 详细输出
        'save': True,                    # 保存检查点
        'save_period': -1,               # 每隔多少轮保存一次 (-1 表示只保存最佳)
        'plots': True,                   # 生成训练图表
    }
    
    # 打印所有参数
    for key, value in train_params.items():
        print(f"  {key:20s} = {value}")
    
    print("-" * 70)
    
    # Debug 模式提示
    if args.debug:
        print("\n⚠️  DEBUG 模式已启用!")
        print("   - 仅训练 1 个 epoch")
        print("   - 用于快速验证代码是否能正常运行")
        print("-" * 70)
    
    # 5. 开始训练
    print(f"\n🏋️  开始训练... (这可能需要较长时间)")
    print(f"   提示: 您可以在另一个终端运行 TensorBoard 查看实时训练曲线:")
    print(f"   tensorboard --logdir=runs/detect/{project_name}")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        results = model.train(**train_params)
        training_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print(f"✅ 训练完成!")
        print(f"   总耗时: {training_time/3600:.2f} 小时 ({training_time/60:.1f} 分钟)")
        print("=" * 70)
        
        return model, results, project_name
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def validate_model(model, data_config, project_name):
    """验证训练好的模型
    
    Args:
        model: 训练好的 YOLO 模型
        data_config: 数据集配置文件路径
        project_name: 实验名称
    """
    print("\n📊 开始模型验证...")
    print("-" * 70)
    
    try:
        # 在验证集上评估模型
        metrics = model.val(data=data_config, imgsz=640)
        
        print("\n📈 验证结果:")
        print(f"  mAP50      : {metrics.box.map50:.4f}  (IoU=0.5 时的 mAP)")
        print(f"  mAP50-95   : {metrics.box.map:.4f}  (IoU=0.5:0.95 的平均 mAP)")
        print(f"  Precision  : {metrics.box.mp:.4f}  (精确率)")
        print(f"  Recall     : {metrics.box.mr:.4f}  (召回率)")
        
        print("-" * 70)
        print(f"✓ 验证完成")
        
        return metrics
        
    except Exception as e:
        print(f"⚠️  验证过程中出现警告: {e}")
        return None


def export_onnx(model, project_name):
    """导出模型为 ONNX 格式
    
    Args:
        model: 训练好的 YOLO 模型
        project_name: 实验名称
    """
    print("\n📤 开始导出 ONNX 模型...")
    print("-" * 70)
    
    try:
        # 导出为 ONNX 格式
        onnx_path = model.export(
            format='onnx',      # 导出格式
            imgsz=640,          # 输入图像尺寸
            opset=12,           # ONNX opset 版本
            simplify=True       # 简化 ONNX 模型
        )
        
        print(f"✓ ONNX 模型导出成功!")
        print(f"  文件位置: {onnx_path}")
        print(f"  可用于: TensorRT, OpenVINO, ONNX Runtime 等推理引擎")
        print("-" * 70)
        
        return onnx_path
        
    except Exception as e:
        print(f"⚠️  ONNX 导出失败: {e}")
        print("   (这不影响训练结果，您仍然可以使用 .pt 格式的模型)")
        return None


def print_summary(project_name, training_time, metrics, onnx_path):
    """打印训练总结信息
    
    Args:
        project_name: 实验名称
        training_time: 训练耗时 (秒)
        metrics: 验证指标
        onnx_path: ONNX 模型路径
    """
    weights_dir = Path(f"runs/detect/{project_name}/weights")
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    
    print("\n" + "=" * 70)
    print("📋 训练总结")
    print("=" * 70)
    print(f"实验名称: {project_name}")
    print(f"训练耗时: {training_time/3600:.2f} 小时")
    
    if metrics:
        print(f"\n性能指标:")
        print(f"  mAP50      : {metrics.box.map50:.4f}")
        print(f"  mAP50-95   : {metrics.box.map:.4f}")
        print(f"  Precision  : {metrics.box.mp:.4f}")
        print(f"  Recall     : {metrics.box.mr:.4f}")
    
    print(f"\n生成的文件:")
    if best_pt.exists():
        print(f"  ✓ 最佳权重: {best_pt}")
    if last_pt.exists():
        print(f"  ✓ 最后权重: {last_pt}")
    if onnx_path:
        print(f"  ✓ ONNX模型: {onnx_path}")
    
    print(f"\n训练结果目录: runs/detect/{project_name}/")
    print(f"  - 包含训练曲线、混淆矩阵等可视化结果")
    print("=" * 70)
    
    print("\n✨ 全部完成! 祝您的毕业设计顺利! ✨\n")


def main():
    """主函数"""
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description='YOLOv11 电力安全检测 - 标准化训练脚本 (毕设专用)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型参数
    parser.add_argument('--model', type=str, default='yolo11l.pt',
                        help='模型文件名 (默认使用 Large 模型)')
    
    # 数据集参数
    parser.add_argument('--data', type=str, default='dataset.yaml',
                        help='数据集配置文件路径')
    
    # 训练参数 (针对 RTX 3060 优化的默认值)
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (RTX 3060 推荐 100)')
    parser.add_argument('--batch', type=int, default=4,
                        help='批次大小 (RTX 3060 推荐 4，云端可用 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载线程数 (RTX 3060 推荐 4)')
    
    # Debug 模式
    parser.add_argument('--debug', action='store_true',
                        help='Debug 模式: 只训练 1 个 epoch 用于快速验证代码')
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    # 执行训练
    model, results, project_name = train_model(args)
    
    # 记录训练时间
    training_time = results.trainer.epoch_time_sum if hasattr(results, 'trainer') else 0
    
    # 如果不是 debug 模式，执行验证和导出
    if not args.debug:
        # 验证模型
        metrics = validate_model(model, args.data, project_name)
        
        # 导出 ONNX
        onnx_path = export_onnx(model, project_name)
        
        # 打印总结
        print_summary(project_name, training_time, metrics, onnx_path)
    else:
        print("\n⚠️  DEBUG 模式: 跳过验证和导出步骤")
        print("   如需完整训练，请去掉 --debug 参数重新运行")


if __name__ == '__main__':
    main()
