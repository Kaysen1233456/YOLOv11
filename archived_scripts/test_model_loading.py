import os
from ultralytics import YOLO

def test_model_loading():
    """测试模型权重加载"""
    # 测试1: 加载yolo11l.pt
    try:
        model_path = os.path.join('runs', 'detect', 'train', 'weights', 'yolo11l.pt')
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✓ 成功加载模型: {model_path}")
        else:
            print(f"✗ 模型文件不存在: {model_path}")
    except Exception as e:
        print(f"✗ 加载模型时出错: {e}")
    
    # 测试2: 加载yolo11n.pt
    try:
        model_path = os.path.join('runs', 'detect', 'train', 'weights', 'yolo11n.pt')
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✓ 成功加载模型: {model_path}")
        else:
            print(f"✗ 模型文件不存在: {model_path}")
    except Exception as e:
        print(f"✗ 加载模型时出错: {e}")
        
    # 测试3: 加载yolov11n.pt
    try:
        model_path = os.path.join('runs', 'detect', 'train', 'weights', 'yolov11n.pt')
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✓ 成功加载模型: {model_path}")
        else:
            print(f"✗ 模型文件不存在: {model_path}")
    except Exception as e:
        print(f"✗ 加载模型时出错: {e}")

if __name__ == "__main__":
    test_model_loading()