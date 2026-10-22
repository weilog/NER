# 传统模型 Dev 实验复现

本阶段使用冻结的有效 Train 9,538 条训练，官方 Dev 1,343 条选参。
运行器不读取项目 Test JSONL，不调用 API；配置来自 `configs/experiment_v2.json`。

## 算法与参数

- 词典：统计训练实体表面形式的类别频次，频次相同按类别名排序。从左到右最长匹配，不使用 Dev 扩展词典。
- HMM：21 个字符级 BIO 状态；监督估计初始、转移与发射概率；全部分布使用同一个 alpha 加性平滑；不建模 EOS。
  词表仅由有效 Train 构造，低于频次阈值及未见字符都映射到独立 UNK 观测。
  alpha 为 0.01 / 0.1 / 1.0，频次阈值为 1 / 2 / 3。
  每组参数只训练一次，比较普通与 BIO 约束的对数域 Viterbi。约束仅屏蔽非法路径，不重新估计或归一化模型概率。
- CRF：字符及前后两位上下文、四个局部二元字符、数字/字母/ASCII 字母/标点和 Unicode 类别。
  CPU L-BFGS，最多 100 次迭代；(c1,c2) 为 (0,0.1) / (0.1,0.1) / (0.1,1.0)。不做参数网格外的追加调参。

共 22 个 Dev 评估配置：词典 1、HMM 18、CRF 3。HMM 主结果仅从九组约束解码中选择；普通解码作为同参数对照。
各方法均按 Dev 实体 micro F1 选参，同分依次比较较低训练耗时和配置顺序。Token F1 不参与选参。
传统方法选择完成后仍需等待 LLM 方案与全局配置冻结，才进入正式 Test。

## 运行

在项目根目录使用已有 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\run_traditional_dev.py
.\.venv\Scripts\python.exe scripts\analyze_traditional_dev.py
.\.venv\Scripts\python.exe scripts\verify_traditional_dev.py
```

默认运行目录为 `runs/traditional_dev`。相同输入与源码下再次运行会校验缓存哈希，并重算已保存预测的完整指标。
源码、依赖、数据或实验配置变化时，不覆盖旧实验；重新训练用新的 `--run-dir runs/另一个名字`。
独立重复训练的耗时允许不同，因此同分候选的耗时平局规则可能在不同机器上选出不同配置；原实验选择和耗时始终保留。
分析、验证和打包脚本默认使用 `configs/traditional_selected.json` 对应的已选方案（分析脚本读取默认 Dev 运行的汇总）。

离线重算一组保存预测：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --gold data\processed\experiment_v2\dev.jsonl --predictions runs\traditional_dev\dictionary\predictions.jsonl --output runs\traditional_dev\verification\dictionary-offline.json
```

这不会重新训练。`verify_traditional_dev.py` 则会加载三种选定模型，对全部 Dev 重新预测，排除耗时字段后要求预测逐条相同，指标也必须完全一致。

## 记录与报告

- `runs/traditional_dev/context.json`：源码逐文件哈希、依赖版本、数据哈希、参数、计时定义与实验指纹。
- 每组 `predictions.jsonl`：ID、状态、实体、原始 BIO（词典直接输出跨度）、逐句推理耗时。
- 每组 `metrics.json`：Token/实体原始计数及分数、每类结果、固定十类 macro F1、诊断、敏感性结果和逐句计数。
- 每组 `normalized.jsonl`：统一规范化后的实体与处理日志。
- 每组 `trial.json`：参数、训练耗时、推理中位数/P95、词表及 UNK 统计、模型与输出哈希。
- `artifacts/models/traditional_dev/实验指纹前缀/`：本地模型；CRF 原生训练日志也随对象保存，可查看实际迭代数。
- `configs/traditional_selected.json`：Dev 选定参数与模型哈希；不是允许正式 Test 运行的全局冻结文件。
- `reports/traditional-dev.md`：三种方案的 Dev 双层指标主表。
- `reports/traditional-dev-grid.csv`：22 组参数及解码对照的完整数值。
- `reports/traditional-dev-per-type.csv`：选定方案的十类实体结果。
- `reports/traditional-dev-analysis.md`：词表和约束实验、错误统计及六个真实 Dev 预测案例。
- `reports/traditional-dev-errors.csv`：实体一对一归因、字符漏标/多标/B-I 混淆/类别混淆计数。

训练时间排除共用的 Python CRF 特征提取；该特征提取时间在运行汇总中单独记录。
逐句推理时间包含特征计算、解码和预测接口转换，不含 IO、评分与模型加载。HMM 两种解码共享完全相同的拟合参数。

错误归因先移除严格匹配，再按“同边界、同类别、更大 IoU、起止位置和类别”确定候选优先级，仅配对相交跨度，每条实体最多使用一次。
边界和类别同时错误单列，其余未配对记录为漏抽或误抽；验证其与严格 FP/FN 数守恒。
六个案例为固定规则自动选出的真实预测，尚未经过两名人工独立核验，不当作随机错误样本。

## 本地结果包

```powershell
.\.venv\Scripts\python.exe scripts\package_traditional_dev.py
```

生成 `artifacts/result-packages/cluener-traditional-dev-2026-10-22.zip` 与 SHA256 文件。
包内含数据、全部本阶段模型和 Dev 预测、代码、配置、报告及逐文件哈希清单；打包后检查 CRC 和每个文件哈希。
排除 `.env`、密钥、虚拟环境和缓存。已有同名包时拒绝覆盖，保留已交付快照。
原始数据、模型、预测及结果包继续只存本地，不自动上传 GitHub。
