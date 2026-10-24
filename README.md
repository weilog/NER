# CLUENER2020 命名实体识别课程实验

## 当前状态

本机独立环境与原始数据已准备。压缩包已重新校验、完整转换与审计。
新版 Train/Dev/Test 已固定为 9,539 / 1,343 / 1,209；有效 Train 为 9,538。
双层评估器已实现，Token 与实体主指标采用同一规范化预测集合。
词典、HMM、CRF 已完成 22 组 Dev 配置评估；按实体 F1 选出的传统方案为 CRF（c1=0.1、c2=0.1），Dev 实体 F1 为 70.97%。
DeepSeek 已完成两版固定 20 条 Dev 试跑；第二版通过格式和位置检查，接口参数已冻结，累计保守记账约 0.0298 元。
正式 Test 和 300 条 LLM 调参尚未运行，不能将 Dev 或试跑分数当成最终结果。

当前配置为 `configs/experiment_v2.json`；`configs/experiment.json` 为保留的历史草稿。
详细规则和复现方式见 [双层评估协议](docs/experiment-protocol-v2.md)。
传统实验见 [Dev 主结果](reports/traditional-dev.md)、[原理与错误分析](reports/traditional-dev-analysis.md) 和 [复现说明](docs/traditional-experiments.md)。
LLM 试跑见 [当前结果](reports/llm-pilot.md)、[第一版诊断](reports/llm-pilot-v1.md) 和 [预算及缓存协议](docs/llm-pilot.md)。

## 数据

- data/raw/cluener2020：原压缩包、原文件和原字段 JSONL。
- data/processed/cluener2020：统一字符级实体跨度、BIO 格式。
- data/manifests/cluener2020：统计、哈希、标签映射、重复和重叠记录。
- docs/data_preparation.md：数据审计结果与限制。

保留原始划分，不去重、不抽样。test 无真实标签，不能直接评估；训练中的重叠样本保留并显式标记。许可未在随包 README 中明确说明。

## 使用

Git 仓库不包含原始语料、处理后数据或虚拟环境。克隆后，先按 `docs/environment.md` 创建环境，再将自行取得的 `cluener_public.zip` 放入 `data/raw/cluener2020/`（目录不存在时创建）。来源项目：https://github.com/CLUEbenchmark/CLUENER2020 。

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_cluener.py
.\.venv\Scripts\python.exe scripts\prepare_split.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

环境说明见 docs/environment.md；精确依赖见 requirements-lock.txt。密钥只放环境变量或本机 .env；LLM 运行器显式读取 .env，其他模块不会自动加载。

划分 ID 和哈希见 `data/splits/experiment_v2.json`；唯一重叠样本 `train:1506` 固定在 Train，但训练与提示示例统一跳过。
官方无标签 Test 单独保留。传统模型选择见 `configs/traditional_selected.json`；仍需完成 LLM Dev 实验并冻结全局配置，正式 Test 尚未运行。

## 旧项目归档

Few-NERD 数据与处理材料存放在 C:/Users/cdk/Desktop/NLP_archives/fewnerd_before_cluener_2026-10-22.zip。归档含逐文件 SHA256 清单，移出活动目录前已验证全部归档文件。
