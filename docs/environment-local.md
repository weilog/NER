# 本机环境（2026-10-22）

- 仓库：`https://github.com/weilog/NER`，拉取提交 `ce2a7a9`。
- 项目目录：`E:\nlp\NER`。
- 平台：Windows x64；Python 3.13.11（原项目记录为 3.13.5）。
- 独立解释器：`E:\nlp\NER\.venv\Scripts\python.exe`。
- 已按 `requirements-lock.txt` 安装全部 50 个精确版本依赖，并以可编辑模式安装项目。
- 本机使用 pip 管理此环境，没有修改全局 Python 包。

## 使用

PowerShell 中无需激活即可运行：

```powershell
cd E:\nlp\NER
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

编辑器选择上述独立解释器。若需要激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 验证结果

- `pip check` 通过，全部锁定依赖版本核对通过。
- 项目包及全部直接依赖导入通过。
- 逻辑回归、CRF 合成数据最小训练和预测通过。
- Matplotlib 无窗口 PNG 绘图通过。
- CLUENER 合成数据实体跨度、BIO 转换及无标签处理通过。
- 原始数据现已从用户提供的压缩包复制并校验，原有 4 项完整数据测试通过。
- 新增划分与双层评估器验收，详见 `docs/experiment-protocol-v2.md`。

本机检查脚本保存在 `.venv/check_environment.py`，可重新运行：

```powershell
.\.venv\Scripts\python.exe .venv\check_environment.py
```

## 数据和后续运行

压缩包已保存在 `E:\nlp\NER\data\raw\cluener2020\cluener_public.zip`。
可重新执行：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_cluener.py
.\.venv\Scripts\python.exe scripts\prepare_split.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

已经使用本地 CPU 完成词典、HMM、CRF 的 Dev 训练与评估，详情见 `docs/traditional-experiments.md`。
2026-10-24 已按用户提供的密钥配置本机 .env，并完成 DeepSeek 两版共 40 次试跑调用，累计保守记账约 0.0298 元。
没有下载预训练模型。密钥不进入代码、报告、提交或本地结果包。
PyTorch/CUDA 不是当前项目依赖，未额外安装。
