# V3：保留重叠实体的字符BIO集合评估

本协议取代V2作为当前默认评估器；V1/V2保留用于旧结果复核。历史34份预测已在独立目录完成V1/V2/V3重评，见reports/evaluation_versions/comparison.md；原报告保持历史版本，不直接改标签称V3。此为查看既有Test后进行的协议修订。

## 统一表示与主指标

每个方法的预测先转换为字符级[start,end)、type跨度。精确重复实体按集合去重。BIO输入保留原序列、记录非法I转移并按固定I→B规则解释为跨度。所有合法跨度均保留，包括重叠或嵌套；不调用resolve_overlaps选择实体。

实体级micro：比较(句子ID,start,end,type)集合。
Token级micro：将每个实体展开成(句子ID,字符位置,完整BIO标签)集合；实体首字符为B-type，其余为I-type。比较集合交集、预测差集、金标差集，分别统计TP/FP/FN。没有O条目，O→实体会产生FP；一个字符可对应多个标签项。全体句子计数汇总后计算micro，不平均逐句F1。

借鉴来源：https://aphp.github.io/edsnlp/v0.22.0/metrics/ner/
EDS-NLP的NerTokenMetric使用Token位置与实体类型，本项目采用字符并额外保留B/I。因此称“字符位置＋BIO标签集合式micro”，不是EDS-NLP原生指标，也不是普通单标签分类。没有重叠时与原完整BIO统计等价。

## 集合评估的边界

相同位置、相同BIO标签只计一次。因此同起点、同类型的较短重复覆盖实体可能不增加Token错误，但会增加严格实体FP。Token集合无法还原全部实体边界结构，必须并列报告实体指标。
辅助character_accuracy定义为每个字符的BIO标签集合完全一致率；双方空集合表示O→O。不将多个标签压成单标签。

## 无效输出与诊断

越界/未知类型等不能形成有效字符标签，排除于两套合法跨度主指标之外。另强制报告entity_with_invalid_fp（无效项计实体FP）、invalid_entities/invalid_sample_rate。不能只展示清洗后的高分。
保留调用失败样本并按空预测评分。重复、非法BIO、原始记录和合法实体均保留诊断。
overlapping_entities表示参与至少一对重叠的不同实体个数，overlap_pairs表示重叠实体对数，overlap_samples表示有重叠预测的句数。重叠只统计、不删除。
V3不再报告“去重叠前后F1差”，因为没有去重叠步骤。V2的entity_before_normalization字段仅属于V2，不与V3混用。

## 实现与执行范围

common.py：entities_to_bio_set展开全部跨度；overlapping_entities只诊断。
metrics.py：默认V3，result.protocol_version=v3。
metrics_v1.py/metrics_v2.py：历史实现；replay.py按结果版本分派，未知版本报错。
首次实现时只运行人工小例子及单元测试；之后已按用户要求独立重评34份历史预测，并在新版Prompt四档Test中使用V3。新调用与评估器离线重评属于不同操作，不改动旧划分、原预测及原冻结记录。
V3重评保存于runs/evaluation_versions/v3，保留V1报告；新版Prompt结果保存于runs/llm_text_test_v1，不根据Test分数调参。当前版本与异常追踪见reports/README.md。
