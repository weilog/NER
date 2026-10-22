# CLUENER2020 数据格式

原始 JSON/JSONL 完整保留 text、label 和可选 id，原位置为闭区间。标准 JSONL 字段：id（源划分:行号）、source_id、text、tokens（逐字符）、has_labels、entities、bio_tags、bio_compatible。

entities 每项包含 start、end、type、text，位置统一为字符索引 [start,end)。不修改原文、不规范化空白。

无标签 test：has_labels=false，entities/bio_tags/bio_compatible=null，不允许当成负样本评分。

重叠实体：完整保存实体列表，bio_compatible=false、bio_tags=null。单层序列模型必须显式处理，不得静默丢弃或覆盖。

未去重、未抽样、未重新划分。当前实体级统计按每个标注跨度出现一次计数，而非每句每种实体文字计一次。
