# 阿里云魔搭部署文件清单

## 📦 代码包内容（yolov11_code.zip）

### 必需文件
- `main_train.py` - 主训练脚本
- `dataset.yaml` - 本地数据集配置（参考）
- `dataset_cloud.yaml` - 云端数据集配置（需根据实际路径修改）
- `requirements.txt` - Python 依赖包
- `app.py` - 应用主文件

### 可选文件
- `TRAINING_GUIDE.md` - 训练指南
- `README_IMPROVEMENTS.md` - 项目说明

---

## 🚀 快速打包脚本

在本地项目根目录运行以下命令：

```powershell
# Windows PowerShell

# 1. 创建部署文件夹
New-Item -ItemType Directory -Force -Path deploy_package
cd deploy_package

# 2. 复制必需文件
Copy-Item ..\main_train.py .
Copy-Item ..\dataset.yaml .
Copy-Item ..\requirements.txt .
Copy-Item ..\app.py .
Copy-Item ..\TRAINING_GUIDE.md .

# 3. 创建云端配置文件
@"
path: /mnt/data/power_safety
nc: 7
names:
  0: person
  1: helmet_person
  2: insulated_gloves
  3: safety_belt
  4: power_pole
  5: voltage_tester
  6: work_clothes

train: train/images
val: val/images
"@ | Out-File -FilePath dataset_cloud.yaml -Encoding UTF8

# 4. 打包
Compress-Archive -Path * -DestinationPath ..\yolov11_code.zip -Force

# 5. 返回上级目录
cd ..

Write-Host "✅ 打包完成: yolov11_code.zip"
Write-Host "📦 文件大小: $((Get-Item yolov11_code.zip).Length / 1KB) KB"
```

或者使用简化版（一行命令）：

```powershell
# 快速打包（只包含核心文件）
Compress-Archive -Path main_train.py,dataset.yaml,requirements.txt,app.py -DestinationPath yolov11_code.zip -Force
```

---

## ☁️ 云端部署流程

### 1. 上传文件到魔搭

- 上传 `yolov11_code.zip`（代码包，几 KB）
- 上传 `datasets.zip`（数据集，分开上传）

### 2. 在魔搭 Jupyter 中执行

```bash
# 解压代码包
!unzip -q yolov11_code.zip -d ~/yolov11_project
%cd ~/yolov11_project

# 解压数据集（假设上传到了 ~/datasets.zip）
!mkdir -p /mnt/data
!unzip -q ~/datasets.zip -d /mnt/data

# 检查数据集结构
!ls /mnt/data/power_safety/train/images | head -5
!ls /mnt/data/power_safety/val/images | head -5

# 安装依赖
!pip install -q ultralytics

# 修改 dataset_cloud.yaml 中的 path（如果需要）
# 确保 path 指向 /mnt/data/power_safety 或您的实际路径

# 开始训练（云端高性能配置）
!python main_train.py --data dataset_cloud.yaml --batch 16 --workers 8 --epochs 100
```

### 3. 训练监控

```bash
# 在另一个终端查看训练日志
!tail -f runs/detect/main_train/train.log

# 或使用 TensorBoard
!tensorboard --logdir=runs/detect/main_train --host=0.0.0.0 --port=6006
```

---

## ⚙️ 云端路径配置注意事项

### 数据集路径映射

| 本地 | 云端 | 说明 |
|------|------|------|
| `c:\yolov11\datasets\power_safety` | `/mnt/data/power_safety` | 需修改 `dataset_cloud.yaml` |
| `datasets/power_safety` (相对) | `/home/user/yolov11_project/datasets/power_safety` | 如果代码和数据在同一目录 |

### 推荐配置

**选项 1: 数据和代码分离（推荐）**
```yaml
# dataset_cloud.yaml
path: /mnt/data/power_safety
```

**选项 2: 数据和代码在一起**
```yaml
# dataset_cloud.yaml
path: datasets/power_safety  # 相对路径，数据在代码目录下
```

---

## 📊 文件大小估算

- 代码包 (`yolov11_code.zip`): ~20 KB
- 数据集 (`datasets.zip`): ~6.6 GB
- 总上传量: ~6.6 GB（主要是数据集）

**优化建议**: 
- 代码包很小，可以快速上传
- 数据集大，建议：
  1. 使用魔搭的数据集存储功能
  2. 或使用 rsync/scp 增量上传
  3. 或在云端直接从 OSS/网盘下载

---

## ✅ 部署前检查清单

- [ ] 本地运行 `python main_train.py --debug --epochs 5` 验证通过
- [ ] 创建 `dataset_cloud.yaml` 并设置正确的云端路径
- [ ] 打包代码文件（不包含数据集）
- [ ] 上传代码包到魔搭
- [ ] 上传或挂载数据集到云端
- [ ] 在云端验证数据集路径正确
- [ ] 云端安装依赖 `pip install ultralytics`
- [ ] 开始训练

---

## 🎯 快速命令参考

**本地验证（5轮）:**
```bash
python main_train.py --debug --epochs 5 --model yolo11l.pt
```

**云端训练（高性能）:**
```bash
python main_train.py --data dataset_cloud.yaml --batch 16 --workers 8 --epochs 100
```

**云端训练（超长训练）:**
```bash
python main_train.py --data dataset_cloud.yaml --batch 16 --workers 8 --epochs 200
```
