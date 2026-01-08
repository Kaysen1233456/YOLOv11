# 一键打包脚本 - 准备云端部署文件
# 使用方法: .\pack_for_cloud.ps1

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  YOLOv11 云端部署文件打包工具" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

# 1. 创建临时部署文件夹
$deployFolder = "deploy_package"
Write-Host "`n📁 创建部署文件夹: $deployFolder" -ForegroundColor Yellow

if (Test-Path $deployFolder) {
    Remove-Item $deployFolder -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $deployFolder | Out-Null

# 2. 复制必需文件
Write-Host "`n📋 复制必需文件..." -ForegroundColor Yellow

$files = @(
    "main_train.py",
    "dataset.yaml",
    "requirements.txt",
    "app.py",
    "TRAINING_GUIDE.md",
    "CLOUD_DEPLOYMENT.md"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Copy-Item $file $deployFolder\
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $file (不存在，跳过)" -ForegroundColor Yellow
    }
}

# 3. 创建云端专用配置文件
Write-Host "`n📝 创建云端专用配置文件..." -ForegroundColor Yellow

$cloudConfig = @"
# 阿里云魔搭专用数据集配置
# 请根据您的实际数据集路径修改 path

path: /mnt/data/power_safety  # 修改为您云端数据集的实际路径
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
"@

$cloudConfig | Out-File -FilePath "$deployFolder\dataset_cloud.yaml" -Encoding UTF8
Write-Host "  ✓ dataset_cloud.yaml (云端专用)" -ForegroundColor Green

# 4. 创建云端快速启动脚本
$cloudScript = @"
#!/bin/bash
# 云端快速启动脚本

echo "====================================="
echo "  YOLOv11 云端训练环境配置"
echo "====================================="

# 1. 检查数据集
echo ""
echo "1️⃣ 检查数据集路径..."
if [ -d "/mnt/data/power_safety/train/images" ]; then
    echo "✓ 训练集路径正确"
    echo "  训练图片数量: \$(ls /mnt/data/power_safety/train/images | wc -l)"
else
    echo "❌ 训练集路径不存在，请检查 dataset_cloud.yaml 中的 path 配置"
    exit 1
fi

if [ -d "/mnt/data/power_safety/val/images" ]; then
    echo "✓ 验证集路径正确"
    echo "  验证图片数量: \$(ls /mnt/data/power_safety/val/images | wc -l)"
else
    echo "❌ 验证集路径不存在，请检查配置"
    exit 1
fi

# 2. 安装依赖
echo ""
echo "2️⃣ 安装 Python 依赖..."
pip install -q ultralytics
echo "✓ ultralytics 安装完成"

# 3. 检查 GPU
echo ""
echo "3️⃣ 检查 GPU 环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 4. 开始训练
echo ""
echo "====================================="
echo "  准备就绪! 开始训练..."
echo "====================================="
echo ""
echo "运行以下命令开始训练:"
echo ""
echo "  python main_train.py --data dataset_cloud.yaml --batch 16 --workers 8 --epochs 100"
echo ""
"@

$cloudScript | Out-File -FilePath "$deployFolder\setup_cloud.sh" -Encoding UTF8
Write-Host "  ✓ setup_cloud.sh (云端启动脚本)" -ForegroundColor Green

# 5. 创建 README
$readme = @"
# YOLOv11 云端部署包

## 📦 包含文件
- main_train.py - 主训练脚本
- dataset.yaml - 本地配置（参考）
- dataset_cloud.yaml - 云端配置（需修改路径）
- requirements.txt - Python 依赖
- app.py - 应用文件
- TRAINING_GUIDE.md - 训练指南
- CLOUD_DEPLOYMENT.md - 云端部署详细说明
- setup_cloud.sh - 云端快速启动脚本

## 🚀 云端部署步骤

### 1. 上传文件
将此压缩包上传到阿里云魔搭

### 2. 解压
```bash
unzip yolov11_code.zip -d ~/yolov11_project
cd ~/yolov11_project
```

### 3. 配置数据集路径
编辑 `dataset_cloud.yaml`，将 `path` 修改为您云端数据集的实际路径

### 4. 运行启动脚本
```bash
bash setup_cloud.sh
```

### 5. 开始训练
```bash
# 云端高性能训练
python main_train.py --data dataset_cloud.yaml --batch 16 --workers 8 --epochs 100
```

## 📝 注意事项
1. 确保云端数据集已上传并解压到正确位置
2. 修改 dataset_cloud.yaml 中的 path 指向数据集实际路径
3. 根据云端 GPU 性能调整 batch size

## 💡 本地验证命令
在上传云端之前，先在本地验证（5轮测试）：
```bash
python main_train.py --debug --epochs 5 --model yolo11l.pt
```

详细说明请查看 `CLOUD_DEPLOYMENT.md`
"@

$readme | Out-File -FilePath "$deployFolder\README.txt" -Encoding UTF8
Write-Host "  ✓ README.txt (部署说明)" -ForegroundColor Green

# 6. 打包
Write-Host "`n📦 打包文件..." -ForegroundColor Yellow

$zipFile = "yolov11_code.zip"
if (Test-Path $zipFile) {
    Remove-Item $zipFile -Force
}

Compress-Archive -Path "$deployFolder\*" -DestinationPath $zipFile -Force

# 7. 清理临时文件夹
Remove-Item $deployFolder -Recurse -Force

# 8. 显示结果
Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
Write-Host "✅ 打包完成!" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan

$zipSize = (Get-Item $zipFile).Length
Write-Host "`n📦 文件名: $zipFile" -ForegroundColor Yellow
Write-Host "📊 文件大小: $([math]::Round($zipSize / 1KB, 2)) KB" -ForegroundColor Yellow

Write-Host "`n📋 包含文件:" -ForegroundColor Yellow
Get-Content -Path (Join-Path (Split-Path $zipFile) "yolov11_code.zip") -Encoding Byte | Out-Null
Expand-Archive -Path $zipFile -DestinationPath temp_check -Force
Get-ChildItem temp_check | ForEach-Object {
    Write-Host "  ✓ $($_.Name)" -ForegroundColor Green
}
Remove-Item temp_check -Recurse -Force

Write-Host "`n🚀 下一步操作:" -ForegroundColor Cyan
Write-Host "  1. 将 $zipFile 上传到阿里云魔搭" -ForegroundColor White
Write-Host "  2. 解压并按照 README.txt 说明操作" -ForegroundColor White
Write-Host "  3. 开始训练!" -ForegroundColor White

Write-Host "`n💾 在上传云端前，建议在本地验证 (5轮测试):" -ForegroundColor Yellow
Write-Host "  python main_train.py --debug --epochs 5 --model yolo11l.pt" -ForegroundColor Cyan

Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
