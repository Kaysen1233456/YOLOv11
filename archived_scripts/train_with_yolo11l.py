import argparse
import sys
import os
from ultralytics import YOLO

def train_model(weights_file='runs/detect/train/weights/yolo11l.pt', epochs=5, imgsz=640):
    """
    使用指定的权重文件训练模型
    
    Args:
        weights_file (str): 权重文件路径
        epochs (int): 训练轮数
        imgsz (int): 图像尺寸
    """
    print(f"开始使用 {weights_file} 权重文件进行训练...")
    
    # 检查权重文件是否存在
    if not os.path.exists(weights_file):
        print(f"错误: 权重文件 {weights_file} 不存在")
        return False
    
    try:
        # 加载模型
        print(f"正在加载模型: {weights_file}")
        model = YOLO(weights_file)
        print("模型加载成功")
        
        # 开始训练
        print(f"开始训练 {epochs} 个epochs...")
        model.train(
            data='dataset.yaml',
            epochs=epochs,
            imgsz=imgsz,
            project='runs/detect',
            name='train_yolo11l'
        )
        
        print("训练完成!")
        return True
        
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用YOLO11L权重文件训练模型')
    parser.add_argument('--weights', type=str, default='runs/detect/train/weights/yolo11l.pt', 
                        help='权重文件路径 (default: runs/detect/train/weights/yolo11l.pt)')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='训练轮数 (default: 5)')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='图像尺寸 (default: 640)')
    
    args = parser.parse_args()
    
    success = train_model(args.weights, args.epochs, args.imgsz)
    
    if not success:
        sys.exit(1)