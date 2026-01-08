#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练进度监控脚本
实时监控模型训练进度和状态
"""

import os
import time
import argparse
from pathlib import Path
import json

def monitor_training_progress(log_dir=None):
    """
    监控训练进度
    
    Args:
        log_dir (str): 训练日志目录
    """
    # 如果未指定日志目录，则使用默认路径
    if log_dir is None:
        project_root = Path(__file__).resolve().parent
        log_dir = project_root / 'runs' / 'detect' / 'full_train'
    
    log_dir = Path(log_dir)
    
    print(f"🔬 开始监控训练进度...")
    print(f"   监控目录: {log_dir}")
    print(f"   按 Ctrl+C 停止监控")
    print("-" * 50)
    
    # 记录初始状态
    initial_files = set()
    if log_dir.exists():
        initial_files = set(log_dir.rglob('*'))
        print(f"初始文件数: {len(initial_files)}")
    
    try:
        while True:
            # 检查目录是否存在
            if not log_dir.exists():
                print("⚠️  训练目录不存在，等待训练开始...")
                time.sleep(5)
                continue
            
            # 获取当前所有文件
            current_files = set(log_dir.rglob('*'))
            new_files = current_files - initial_files
            
            # 检查是否有新文件
            if new_files:
                print(f"\n[{time.strftime('%H:%M:%S')}] 发现新文件:")
                for f in sorted(new_files):
                    file_size = f.stat().st_size
                    print(f"  + {f.relative_to(log_dir)} ({file_size} bytes)")
                initial_files = current_files
            
            # 检查results.csv文件（如果存在）
            results_file = log_dir / 'results.csv'
            if results_file.exists():
                lines = sum(1 for _ in open(results_file, 'r', encoding='utf-8'))
                print(f"[{time.strftime('%H:%M:%S')}] 训练进度: 已完成 {lines-1} 轮训练")
            
            # 检查权重文件
            weights_dir = log_dir / 'weights'
            if weights_dir.exists():
                weights_files = list(weights_dir.iterdir())
                if weights_files:
                    print(f"[{time.strftime('%H:%M:%S')}] 权重文件:")
                    for wf in weights_files:
                        size = wf.stat().st_size / (1024*1024)  # MB
                        print(f"  - {wf.name} ({size:.1f} MB)")
            
            time.sleep(5)  # 每5秒检查一次
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 训练监控已停止")
        return True
    except Exception as e:
        print(f"\n❌ 监控过程中发生错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='监控模型训练进度')
    parser.add_argument('--logdir', type=str, help='训练日志目录')
    
    args = parser.parse_args()
    
    # 启动监控
    monitor_training_progress(args.logdir)

if __name__ == '__main__':
    main()