# 评估协议V2：统一预测集合

本次是查看既有Test结果后进行的评估协议修订，不声称在原实验Test前固定。
不改模型参数、提示、样本或已有预测。已有runs及报告仍是V1结果，尚未进行V2整批重评。

## 实现顺序

1. checked_entities检查跨度/类别并去重。BIO输入先记录非法转移，按固定I-X→B-X规则转换为跨度。
2. resolve_overlaps按起点升序、同起点终点降序、同跨度LABELS固定顺序选择实体，不访问金标，不修改输入。
3. 保留集合p直接用于主实体评分，并转换为BIO用于主Token评分。两套主指标都排除无效项，失败句子仍按空预测评分。
4. entity_before_normalization保存去重叠之前的辅助实体评分：已做BIO解释和精确去重，包含全部合法重叠跨度，无效条目计FP。
5. entity_f1_normalization_delta = 主实体F1 - 辅助实体F1；差值可正可负，不保证清洗改善结果。

## 保留的诊断

metrics.json保留invalid_entities、duplicate_entities、illegal_bio_positions、projected_out_entities、overlap_projection_samples及相关比率。
有评分错误、规范化动作或调用失败的句子均进入cases，即使规范化后全部正确也不会丢掉诊断。
cases包含raw_prediction（原始记录的副本）、normalization.kept_entities、removed_overlap_entities及数量统计。
主errors/error_counts针对规范化后的集合；errors_before_normalization/error_counts_before_normalization另存原始跨度归因。
逐句counts同时保存主entity计数和entity_before_normalization计数。原始预测文件不被修改。

## 版本与执行

当前metrics.py默认V2，结果标明protocol_version=v2；旧实现保存在metrics_v1.py。
replay.py根据保存指标的版本选择评估器，旧结果没有版本字段时按V1处理，未知版本报错。
原有冻结配置保持不动，所以不能把源码已变的项目伪装成原冻结实验直接重跑。
不要运行all或覆盖既有runs来做学习练习；后续V2离线重评应输出到独立目录，同时保留V1表。
本次仅修改评估器及小例子测试，不进行模型训练、API调用或整批实验重评。
