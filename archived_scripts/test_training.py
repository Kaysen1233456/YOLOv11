"""
测试训练脚本，运行1个epoch来快速验证训练流程
"""

import os
from ultralytics import YOLO

def test_train():
    """
    测试训练函数，只训练1个epoch
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 权重文件路径    model_path = os.path.join(project_root, 'yolo11l.pt')
    
    # 数据集配置文件路径
    data_path = os.path.join(project_root, 'dataset.yaml')
    
    print("测试训练配置:")
    print(f"  模型路径: {model_path}")
    print(f"  数据集配置: {data_path}")
    print(f"  训练轮数: 1")
    print(f"  图像尺寸: 64")
    print(f"  批次大小: 2")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        # 尝试使用yolo11n.pt作为备选
        alt_model_path = os.path.join(project_root, 'yolo11n.pt')
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
            print(f"使用备选权重文件: {model_path}")
        else:
            return
    
    if not os.path.exists(data_path):
        print(f"错误: 数据集配置文件不存在 {data_path}")
        return
    
    try:
        print("\n开始加载模型...")
        model = YOLO(model_path)
        print("模型加载成功!")
        
        print("\n开始训练...")
        # 使用非常小的参数进行测试
        results = model.train(
            data=data_path,
            epochs=1,
            imgsz=64,  # 使用非常小的图像加快速度
            batch=2,   # 使用小批次
            project=os.path.join(project_root, 'runs', 'detect'),
            name='test_train',
            exist_ok=True,
            verbose=True
        )
        
        print("测试训练完成!")
        return True
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_train()