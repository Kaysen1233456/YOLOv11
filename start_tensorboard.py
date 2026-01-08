#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TensorBoard监控启动脚本
用于启动TensorBoard以监控模型训练进度
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def start_tensorboard(logdir=None):
    """
    启动TensorBoard服务
    
    Args:
        logdir (str): TensorBoard日志目录
    """
    # 如果未指定日志目录，则使用默认路径
    if logdir is None:
        project_root = Path(__file__).resolve().parent
        logdir = project_root / 'runs' / 'detect' / 'full_train'
    
    logdir = Path(logdir)
    
    # 检查日志目录是否存在
    if not logdir.exists():
        print(f"⚠️  日志目录不存在: {logdir}")
        print("  请先开始训练或检查路径是否正确")
        return False
    
    print(f"📈 启动TensorBoard监控...")
    print(f"  日志目录: {logdir}")
    print(f"  访问地址: http://localhost:6006")
    
    try:
        # 启动TensorBoard
        cmd = [
            'tensorboard',
            '--logdir', str(logdir),
            '--host', 'localhost',
            '--port', '6006'
        ]
        
        print(f"  命令: {' '.join(cmd)}")
        print(f"  按 Ctrl+C 停止TensorBoard")
        print("-" * 50)
        
        # 启动TensorBoard进程
        process = subprocess.Popen(cmd)
        
        # 等待进程结束
        process.wait()
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 TensorBoard已停止")
        return True
    except FileNotFoundError:
        print("❌ 未找到TensorBoard命令")
        print("  请确保已安装TensorBoard:")
        print("  pip install tensorboard")
        return False
    except Exception as e:
        print(f"❌ 启动TensorBoard时发生错误: {e}")
        return False

def check_tensorboard_installed():
    """
    检查TensorBoard是否已安装
    """
    try:
        import tensorboard
        print(f"✅ TensorBoard已安装 (版本: {tensorboard.__version__})")
        return True
    except ImportError:
        print("❌ TensorBoard未安装")
        print("  安装命令: pip install tensorboard")
        return False

def main():
    parser = argparse.ArgumentParser(description='启动TensorBoard监控')
    parser.add_argument('--logdir', type=str, help='TensorBoard日志目录')
    parser.add_argument('--check', action='store_true', help='仅检查TensorBoard是否已安装')
    
    args = parser.parse_args()
    
    # 检查TensorBoard是否已安装
    if not check_tensorboard_installed():
        if not args.check:
            response = input("是否现在安装TensorBoard? (y/n): ")
            if response.lower() == 'y':
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorboard'])
                    print("✅ TensorBoard安装完成")
                except Exception as e:
                    print(f"❌ 安装TensorBoard失败: {e}")
                    return False
            else:
                return False
    
    if args.check:
        return True
    
    # 启动TensorBoard
    return start_tensorboard(args.logdir)

if __name__ == '__main__':
    main()