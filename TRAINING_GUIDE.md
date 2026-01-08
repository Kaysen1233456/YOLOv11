# main_train.py 使用指南

## 📖 脚本说明

`main_train.py` 是您 YOLOv11 电力安全检测毕业设计项目的**唯一标准化训练入口**。

### ✨ 核心特性

1. **默认使用 YOLOv11-Large 模型**
   - 自动加载 `yolo11l.pt` (Large 模型，精度最高)
   - 如果本地没有，会自动从 ultralytics 下载

2. **针对 RTX 3060 优化**
   - 默认 batch size = 4 (防止显存溢出)
   - 默认 workers = 4 (适合本地环境)
   - 默认 epochs = 100

3. **集成增强训练策略**
   - Mosaic 数据增强 (图像拼接)
   - Mixup 数据增强 (图像混合)
   - 余弦退火学习率调度
   - HSV 颜色空间增强
   - 多种几何变换增强

4. **自动设备检测**
   - 自动检测并打印 GPU 型号
   - 显示显存大小、CUDA 版本等信息
   - 确认是否使用 RTX 3060

5. **训练后自动处理**
   - 自动在验证集上评估模型
   - 自动导出 ONNX 格式模型
   - 生成训练曲线、混淆矩阵等可视化

6. **Debug 模式**
   - 使用 `--debug` 参数快速验证代码
   - 只训练 1 个 epoch
   - 适合测试环境是否配置正确

---

## 🚀 使用方法

### 1️⃣ 本地训练 (RTX 3060)

**基础训练 - 使用所有默认参数:**
```bash
python main_train.py
```

这将使用以下默认配置:
- 模型: yolo11l.pt (Large)
- Epochs: 100
- Batch size: 4
- Workers: 4
- Image size: 640

---

### 2️⃣ 云端训练 (更大 batch size)

如果您在云端有更强的 GPU (如 V100, A100 等):

```bash
python main_train.py --batch 16 --workers 8
```

或者训练更多轮次:

```bash
python main_train.py --batch 16 --workers 8 --epochs 200
```

---

### 3️⃣ Debug 模式 - 快速验证代码

**在正式训练前，建议先运行 debug 模式确保一切正常:**

```bash
python main_train.py --debug
```

Debug 模式特点:
- ✓ 只训练 1 个 epoch
- ✓ 快速验证代码能否正常运行
- ✓ 检查数据集是否正确加载
- ✓ 确认 GPU 是否正常工作
- ✓ 通常 2-5 分钟即可完成

---

### 4️⃣ 使用其他模型

如果您想使用其他尺寸的 YOLOv11 模型:

```bash
# 使用 Nano 模型 (最快，但精度较低)
python main_train.py --model yolo11n.pt

# 使用 Small 模型
python main_train.py --model yolo11s.pt

# 使用 Medium 模型
python main_train.py --model yolo11m.pt

# 使用 Large 模型 (默认)
python main_train.py --model yolo11l.pt

# 使用 XLarge 模型 (最大，精度最高但最慢)
python main_train.py --model yolo11x.pt
```

---

### 5️⃣ 自定义数据集

如果您的数据集配置文件不是 `dataset.yaml`:

```bash
python main_train.py --data custom_dataset.yaml
```

---

## 📊 训练输出

训练完成后，结果保存在:

```
runs/detect/main_train/
├── weights/
│   ├── best.pt          # 最佳模型权重 (验证集 mAP 最高)
│   ├── last.pt          # 最后一轮的权重
│   └── best.onnx        # 导出的 ONNX 模型
├── results.csv          # 训练指标 CSV 文件
├── results.png          # 训练曲线图
├── confusion_matrix.png # 混淆矩阵
├── F1_curve.png         # F1 曲线
├── PR_curve.png         # 精确率-召回率曲线
└── ...                  # 其他可视化结果
```

---

## 📝 完整参数列表

```bash
python main_train.py --help
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | yolo11l.pt | 模型文件名 |
| `--data` | str | dataset.yaml | 数据集配置文件 |
| `--epochs` | int | 100 | 训练轮数 |
| `--batch` | int | 4 | 批次大小 (RTX 3060 推荐 4) |
| `--imgsz` | int | 640 | 输入图像尺寸 |
| `--workers` | int | 4 | 数据加载线程数 |
| `--debug` | flag | False | Debug 模式 (只训练 1 epoch) |

---

## 💡 常见使用场景

### 场景 1: 首次使用，验证环境

```bash
# 第一步: 运行 debug 模式确保一切正常
python main_train.py --debug

# 第二步: 如果 debug 成功，开始正式训练
python main_train.py
```

### 场景 2: 本地 RTX 3060 完整训练

```bash
python main_train.py --epochs 100
```

### 场景 3: 云端高性能训练

```bash
python main_train.py --batch 16 --workers 8 --epochs 200
```

### 场景 4: 快速实验不同模型

```bash
# 先用小模型快速测试
python main_train.py --model yolo11n.pt --epochs 50

# 确认效果后再用大模型
python main_train.py --model yolo11l.pt --epochs 100
```

---

## ⚠️ 注意事项

1. **显存不足错误**
   - 如果遇到 CUDA out of memory 错误
   - 减小 batch size: `--batch 2` 或 `--batch 1`

2. **首次运行会下载模型**
   - 如果本地没有 yolo11l.pt
   - 脚本会自动从 ultralytics 下载
   - 需要联网，下载大小约 50MB

3. **数据集路径**
   - 确保 dataset.yaml 中的路径正确
   - 路径可以是相对路径或绝对路径

4. **TensorBoard 实时监控**
   - 训练时可以在另一个终端运行:
   ```bash
   tensorboard --logdir=runs/detect/main_train
   ```
   - 然后在浏览器打开 http://localhost:6006

---

## 🎓 毕业设计建议

1. **训练记录**: 保存好每次训练的输出日志
2. **对比实验**: 可以尝试不同的模型尺寸对比效果
3. **可视化**: 使用训练输出的图表作为论文素材
4. **模型部署**: 使用导出的 ONNX 模型可以方便地部署到生产环境

---

## 📞 遇到问题?

如果遇到任何问题，请检查:
1. ✓ 是否安装了 ultralytics 包: `pip install ultralytics`
2. ✓ CUDA 和 PyTorch 是否正确安装
3. ✓ dataset.yaml 配置是否正确
4. ✓ 数据集图片和标签是否齐全

祝您毕业设计顺利! 🎉
