# Python 环境

已在 Windows x64 上使用 Python 3.13.5 创建项目独立 .venv。基础解释器来自本机 Anaconda；包安装在 .venv，不修改全局包。

## 使用（PowerShell）

```powershell
cd C:\Users\cdk\Desktop\NLP
.\.venv\Scripts\python.exe --version
```

无需激活也能使用完整解释器路径。编辑器请选择 .venv/Scripts/python.exe。

## 在另一台电脑重建

安装 Python 3.13 与 uv 后，在项目根目录执行：

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe -e .
```

requirements-lock.txt 记录本机实际安装的精确版本；它不是跨操作系统兼容保证。此环境用 uv 管理包，未安装 pip。

## 已验证

全部基础包和项目包可导入；逻辑回归最小训练预测、CRF 最小训练预测、Matplotlib 无窗口绘图通过。此为环境检查，不是正式实验结果。

尚未安装 PyTorch、CUDA 训练栈或下载模型；神经网络基线确认后再配置。尚未调用 DeepSeek API，不产生调用费用。
