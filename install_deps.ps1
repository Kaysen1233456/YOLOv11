# 安装脚本（PowerShell）
# 用法：在项目根目录运行此脚本（以管理员身份视情况而定）
# 说明：使用清华镜像安装大多数 Python 包；PyTorch（含 CUDA）使用官方 PyTorch 索引以确保获取合适的 CUDA wheel。

$ErrorActionPreference = 'Stop'

Write-Host "创建并激活虚拟环境 .venv"
python -m venv .venv

Write-Host "激活虚拟环境"
# 若使用 PowerShell，执行 Activate.ps1
.\.venv\Scripts\Activate.ps1

Write-Host "升级 pip"
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host "尝试安装 PyTorch (CUDA 12.6) from PyTorch 官方索引。若需要其它 CUDA 版本，请手动调整下面的 index-url。"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Write-Host "使用清华镜像安装其余依赖（requirements.txt）"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host "验证安装"
python - <<'PY'
import sys
try:
    import torch
    print('torch', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    print('torch cuda version:', torch.version.cuda)
except Exception as e:
    print('torch import failed:', e)

try:
    import ultralytics
    print('ultralytics', ultralytics.__version__)
except Exception as e:
    print('ultralytics import failed:', e)

try:
    import streamlit
    print('streamlit', streamlit.__version__)
except Exception as e:
    print('streamlit import failed:', e)
    
try:
    import cv2
    print('opencv-python imported successfully')
except Exception as e:
    print('opencv-python import failed:', e)
    
try:
    import numpy
    print('numpy imported successfully')
except Exception as e:
    print('numpy import failed:', e)
    
try:
    from PIL import Image
    print('Pillow imported successfully')
except Exception as e:
    print('Pillow import failed:', e)
PY

Write-Host "安装完成。若需要其它 CUDA 版本（如 cu118、cu121 等），请修改 pip install torch 的 index-url 为相应的 cuXY。"