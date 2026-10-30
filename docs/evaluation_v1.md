# 原始双层评估口径

正式报告使用 `src/ner_experiment/evaluation/metrics_v1.py`，通过 `scripts/integrate_v1_results.py` 离线整合保存预测。模型原始生成文件、提示、参数、划分均保持原样。

## Token

Token为原文Unicode字符。预测跨度按起点优先、同起点较长优先、同跨度按固定类别顺序投影为非重叠实体，再转换成单标签BIO。对20个非O完整BIO标签汇总TP/FP/FN，计算micro P/R/F1；O→O不增加TP，错误类型或B/I混淆计FP和FN。Accuracy仅为辅助。

## 实体

实体按同一句中的起点、终点、类别严格匹配，合法跨度集合去重并保留重叠；可枚举的越界、未知类型等无效项计整体实体FP，不能映射为字符标签的项不强行投影。非法BIO按固定I→B解释为跨度，修复前非法率另报。

## 解释与失败

两级指标采用不同的重叠处理。指标差异可能来自粒度和字符投影，不能全部解释成边界粒度差异。重叠、重复、无效项和失败率同时保存。

缺失、重复、多余ID停止评分。显式API或格式失败保留并按空预测评分。先汇总计数后算micro，零分母为0。固定十类实体macro F1不把未知类型强塞给某一类；未知类型的额外FP体现在整体实体micro和无效项诊断。

## 复现

运行 `.\.venv\Scripts\python.exe scripts/reproduce_main_results.py`。源表在reports/tables/integrated_v1，图在reports/figures/integrated_v1，逐句计数与核验在runs/integrated_v1_analysis。本入口不调用API。历史文件保留归档，不与正式表混合。
