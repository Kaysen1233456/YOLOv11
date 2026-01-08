#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整训练脚本
用于训练电力安全检测模型
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='训练电力安全检测模型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数 (默认: 100)')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸 (默认: 640)')
    parser.add_argument('--batch', type=int, default=16, help='批次大小 (默认: 16)')
    parser.add_argument('--model', type=str, default='yolo11l.pt', help='预训练模型路径')
    parser.add_argument('--data', type=str, default='dataset.yaml', help='数据集配置文件')
    parser.add_argument('--tensorboard', action='store_true', help='启用TensorBoard日志记录')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent
    print(f"项目根目录: {project_root}")
    
    # 设置模型路径
    model_path = args.model
    # 检查相对路径
    if not os.path.isabs(model_path):
        model_path = project_root / model_path
    
    # 检查模型文件是否存在
    if os.path.exists(model_path):
        print(f"✓ 找到模型文件: {model_path}")
    else:
        # 尝试在项目根目录查找
        alt_model_path = project_root / "yolo11l.pt"
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
            print(f"✓ 使用替代模型文件: {model_path}")
        else:
            print(f"✗ 未找到模型文件: {model_path}")
            print("  尝试从Ultralytics自动下载...")
    
    # 设置数据集配置文件路径
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = project_root / data_path
    
    # 检查数据集配置文件
    if os.path.exists(data_path):
        print(f"✓ 找到数据集配置文件: {data_path}")
    else:
        print(f"✗ 未找到数据集配置文件: {data_path}")
        sys.exit(1)
    
    try:
        print(f"\n开始训练...")
        print(f"  模型: {model_path}")
        print(f"  数据集: {data_path}")
        print(f"  训练轮数: {args.epochs}")
        print(f"  图像尺寸: {args.imgsz}")
        print(f"  批次大小: {args.batch}")
        if args.tensorboard:
            print(f"  TensorBoard日志: 启用")
        
        # 加载模型
        print("\n加载模型...")
        model = YOLO(model_path)
        print("✓ 模型加载成功")
        
        # 准备训练参数
        train_args = {
            'data': str(data_path),
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'batch': args.batch,
            'project': str(project_root / 'runs' / 'detect'),
            'name': 'full_train',
            'exist_ok': True,
            'verbose': True
        }
        
        # 如果启用了TensorBoard，则添加相关参数
        if args.tensorboard:
            train_args['project'] = str(project_root / 'runs' / 'detect')
            # Ultralytics YOLO默认支持TensorBoard，只需确保日志目录存在
            log_dir = project_root / 'runs' / 'detect' / 'full_train'
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"  TensorBoard日志目录: {log_dir}")
        
        # 开始训练
        print("\n启动训练过程...")
        results = model.train(**train_args)
        
        print("\n🎉 训练完成!")
        print(f"训练结果保存在: {project_root / 'runs' / 'detect' / 'full_train'}")
        
        # 验证模型
        print("\n验证模型...")
        metrics = model.val()
        print(f"  mAP50: {metrics.box.map50}")
        print(f"  mAP50-95: {metrics.box.map}")
        
        # 提供TensorBoard启动说明
        if args.tensorboard:
            log_dir = project_root / 'runs' / 'detect' / 'full_train'
            print(f"\n📈 要查看TensorBoard日志，请运行:")
            print(f"   tensorboard --logdir {log_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    main()