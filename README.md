# CLUENER2020 命名实体识别课程实验

## 当前状态

Python 独立环境已准备。用户提供的 CLUENER 压缩包已校验、完整转换与审计。尚未训练模型或调用 API。

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
.\.venv\Scripts\python.exe -m pytest tests -q
```

环境说明见 docs/environment.md；精确依赖见 requirements-lock.txt。密钥只放环境变量，.env.example 是模板，未自动加载 .env。

模型组合、独立评估划分与重叠实体训练策略尚待确定，配置见 configs/experiment.json。

## 旧项目归档

Few-NERD 数据与处理材料存放在 C:/Users/cdk/Desktop/NLP_archives/fewnerd_before_cluener_2026-10-22.zip。归档含逐文件 SHA256 清单，移出活动目录前已验证全部归档文件。
