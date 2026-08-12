# 开发环境记录

## 已验证环境

- 操作系统：Windows
- Python：3.12.10
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（8 GB）
- NVIDIA 驱动：591.74
- PyTorch：2.12.1+cu130
- NumPy：2.5.2

虚拟环境与其他项目隔离，存放在 D 盘本地运行目录。数据集和模型产物后续也优先放在 D 盘，源代码保留在项目工作区。

## 已完成验证

1. `torch.cuda.is_available()` 返回 `True`。
2. PyTorch 正确识别 RTX 4060。
3. 1D 卷积在 GPU 上完成前向计算。
4. 损失能够执行反向传播并产生梯度。
5. NumPy 数组能够转换为 CUDA Tensor，再无损转换回 NumPy。

## GPU 依赖安装

```powershell
python -m pip install --no-cache-dir -r requirements-gpu.txt
```

该文件使用 PyTorch 官方 CUDA 13.0 wheel 源。CPU 环境需要使用不同的安装源，不能机械复用GPU配置。
