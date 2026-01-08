import os
from ultralytics import YOLO

# 确保权重文件存在
if not os.path.exists('yolo11l.pt'):
    print("错误: 找不到权重文件 yolo11l.pt")
    exit(1)

print("开始训练...")

# 加载模型
model = YOLO('yolo11l.pt')

# 开始训练（只训练很少的epochs以便快速验证）
results = model.train(
    data='dataset.yaml',
    epochs=3,
    imgsz=320,  # 使用较小的图像尺寸加快训练速度
    batch_size=4,  # 较小的批次大小
    project='runs/detect',
    name='simple_train'
)

print("训练完成")
print(f"结果保存在: {results}")