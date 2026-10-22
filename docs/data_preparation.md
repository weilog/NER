# CLUENER2020 数据准备

原始划分完整保留，未去重、未抽样、未重新划分。
用户提供的压缩包已复制归档，并保留原始 JSON；原 JSON 实际为逐行 JSON，同时提供原字段不变的 .jsonl。

| 划分 | 文本数 | 带标签文本数 | 实体出现次数 | 重叠实体文本数 |
|---|---:|---:|---:|---:|
| train | 10748 | 10748 | 23971 | 1 |
| dev | 1343 | 1343 | 3072 | 0 |
| test | 1345 | 0 | None | 0 |

精确重复文本组：0；跨划分重复组：0；带标签记录间冲突组：0。仅记录，未删除。

原实体位置是字符级闭区间 [start,end]；标准格式改为 [start,end)，并逐实体校验实体文字与原文切片一致。
重叠实体完整保留，bio_tags=null、bio_compatible=false，单层 BIO 模型的处理策略需在训练前明确，不静默覆盖。
test 没有真实标签，entities/bio_tags=null，不能当成没有实体的负样本或直接用于有监督评分。cluener_predict.json 是随包预测示例，不能当测试真值。
当前未指定独立的项目最终评估集；在划分协议确定前不进行正式选参或测试。
许可：随包 README 未明确说明数据许可；保留来源记录，不自行标成开放许可证。

运行：`.venv\Scripts\python.exe scripts\prepare_cluener.py`。详细审计、重复和重叠样本位于 data/manifests/cluener2020。
