import os
import yaml

# 1. 读取你的 yaml
with open('dataset.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 2. 获取你配置的路径
base_path = config.get('path', '')
train_path = config.get('train', '')

# 3. 模拟 YOLO 的路径拼接
# 假设你在项目根目录 'project' 下运行，或者你在当前目录下运行，这里需要你手动确认
current_working_dir = os.getcwd() 
full_path = os.path.join(current_working_dir, base_path, train_path)

print(f"\n--- 路径诊断 ---")
print(f"当前工作目录: {current_working_dir}")
print(f"YAML配置推算的图片路径: {full_path}")

# 4. 判决时刻
if os.path.exists(full_path):
    print("✅ 状态：路径存在。你可以开始训练。")
    # 进一步检查是否有图片
    files = os.listdir(full_path)
    print(f"✅ 目录下发现 {len(files)} 个文件。")
else:
    print("❌ 状态：路径不存在！训练绝对会报错。")
    print("👉 建议：将 dataset.yaml 中的 'path' 改为图片的【绝对路径】以避免任何歧义。")