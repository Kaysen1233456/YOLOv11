"""
训练监控脚本，用于监控训练进度并输出详细信息
"""

import os
import time
from pathlib import Path

def monitor_training():
    """
    监控训练过程并输出状态信息
    """
    project_root = Path(__file__).resolve().parent
    train_dir = project_root / 'runs' / 'detect' / 'train'
    weights_dir = train_dir / 'weights'
    
    print("开始监控训练过程...")
    print(f"项目根目录: {project_root}")
    print(f"训练目录: {train_dir}")
    print(f"权重目录: {weights_dir}")
    
    # 显示初始状态
    print("\n初始文件状态:")
    if train_dir.exists():
        print("训练目录文件:")
        for item in train_dir.iterdir():
            print(f"  {item.name}")
    
    if weights_dir.exists():
        print("权重目录文件:")
        for item in weights_dir.iterdir():
            size = item.stat().st_size
            print(f"  {item.name} ({size} bytes)")
    
    # 持续监控文件变化
    print("\n开始持续监控文件变化 (按 Ctrl+C 停止)...")
    try:
        initial_weights = set(weights_dir.iterdir()) if weights_dir.exists() else set()
        
        while True:
            # 检查是否有新文件生成
            if weights_dir.exists():
                current_weights = set(weights_dir.iterdir())
                new_files = current_weights - initial_weights
                
                if new_files:
                    print(f"\n发现新文件: {[f.name for f in new_files]}")
                    initial_weights = current_weights
            
            # 检查训练目录是否有新文件
            if train_dir.exists():
                train_files = list(train_dir.iterdir())
                print(f"\n训练目录当前文件数量: {len(train_files)}")
            
            time.sleep(5)  # 每5秒检查一次
            
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == '__main__':
    monitor_training()