"""
测试训练启动脚本 - 用于验证训练是否能正常启动
"""

import os
from ultralytics import YOLO

def test_training_start():
    """测试训练是否能正常启动"""
    print("测试训练启动...")
    
    # 检查数据集配置文件
    if not os.path.exists('dataset.yaml'):
        print("错误: 找不到 dataset.yaml 文件")
        return False
    
    # 检查数据集目录
    if not os.path.exists('datasets/power_safety'):
        print("错误: 找不到数据集目录 datasets/power_safety")
        return False
        
    print("数据集配置检查通过")
    
    # 尝试加载模型
    try:
        model = YOLO('yolo11n.pt')
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False
    
    # 尝试启动训练（仅1个epoch）
    try:
        print("开始启动训练（1个epoch）...")
        model.train(
            data='dataset.yaml',
            epochs=1,
            imgsz=640,
            batch=16,
            project='runs/detect',
            name='test_train',
            exist_ok=True
        )
        print("训练启动成功")
        return True
    except Exception as e:
        print(f"训练启动失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_training_start()