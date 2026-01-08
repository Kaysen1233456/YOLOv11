"""
尝试使用 ultralytics 自动下载 yolo11l.pt 并复制到项目的 weights 目录。
用法：在虚拟环境激活后运行：
python download_yolov11n.py

脚本流程：
1. 导入 ultralytics 并调用 YOLO('yolo11l.pt') 触发自动下载。
2. 在常见缓存目录中搜索 yolo11l.pt 文件并复制到 runs/detect/train/weights/ 下，并创建标准名称副本。
3. 如果找不到文件，打印提示并列出搜索过的目录，建议手动下载或提供路径。
"""
import sys
import os
from pathlib import Path
import shutil
import glob

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_DIR = PROJECT_ROOT / 'runs' / 'detect' / 'train' / 'weights'
TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"目标权重目录: {TARGET_DIR}")

# Step 1: try to import ultralytics and create YOLO object (will download if needed)
try:
    from ultralytics import YOLO
except Exception as e:
    print('无法导入 ultralytics，请先安装依赖并激活虚拟环境。错误：', e)
    sys.exit(2)

print('尝试加载 YOLO("yolo11l.pt")，这会触发自动下载（若本地不存在）...')
try:
    _ = YOLO('yolo11l.pt')
    print('YOLO 对象已创建（下载过程可能后台进行或已完成）')
except Exception as e:
    print('构建 YOLO 对象失败（这可能仍会成功下载模型，但出现异常）：', e)

# Step 2: 搜索系统中名为 yolo11l.pt 的文件（在用户目录下搜索以加快定位）
home = Path.home()
search_paths = [
    home / '.cache',
    home / 'AppData' / 'Local' / 'ultralytics',
    PROJECT_ROOT,
    PROJECT_ROOT / 'weights'
]

found = []
print('开始在常见目录中查找 yolo11l.pt（可能需要一些时间）...')
# Limit depth by using glob with pattern
patterns = [
    str(home / '**' / 'yolo11l.pt'),
    str(PROJECT_ROOT / '**' / 'yolo11l.pt'),
    'C:\\**\\yolo11l.pt'
]
for pat in patterns:
    matches = glob.glob(pat, recursive=True)
    for m in matches:
        if os.path.isfile(m):
            found.append(Path(m))

# 同时查找yolo11l.pt以确保兼容性
print('开始在常见目录中查找 yolo11l.pt（可能需要一些时间）...')
patterns_yolo = [
    str(home / '**' / 'yolo11l.pt'),
    str(PROJECT_ROOT / '**' / 'yolo11l.pt'),
    'C:\\**\\yolo11l.pt'
]
for pat in patterns_yolo:
    matches = glob.glob(pat, recursive=True)
    for m in matches:
        if os.path.isfile(m):
            found.append(Path(m))

if not found:
    print('未在常见位置找到 yolo11l.pt。请手动下载并放到下面目录之一：')
    print('  -', TARGET_DIR)
    print('或者在系统上运行：python -c "from ultralytics import YOLO; YOLO(\'yolo11l.pt\')" 然后重试本脚本。')
    sys.exit(1)

# 如果找到多个，优先选择最靠近用户目录的一个
found = sorted(found, key=lambda p: len(str(p)))
chosen = found[0]
print('找到文件：', chosen)

# 确保目标目录中有标准名称的权重文件 (yolo11l.pt)
dst_standard = TARGET_DIR / 'yolo11l.pt'
dst_original = TARGET_DIR / chosen.name

try:
    # 复制为原始名称
    shutil.copy2(chosen, dst_original)
    print(f'已复制到 {dst_original}')
    
    # 如果原始名称不是标准名称，则额外创建一个标准名称的副本
    if chosen.name != 'yolo11l.pt':
        shutil.copy2(chosen, dst_standard)
        print(f'已创建标准名称副本 {dst_standard}')
    else:
        print(f'文件已经是标准名称')
except Exception as e:
    print('复制失败：', e)
    sys.exit(1)

print('完成。现在你可以在 app.py 或 train.py 中使用路径：', dst_standard)