test_cluener_data.py：原始数据无损转换、跨度/BIO 一致性、重叠实体和无标签测试集检查。

test_integrated_results.py：15组整合的严格ID校验、42组配对设计、统一网络诊断子集（保留格式/内容失败）、成功恢复不被旧错误字段误判、历史未知费用预留不被删去。测试使用构造记录，不发送API请求。pytest默认只收集tests，避免误导入runs中的历史运行器源码。

本地完整结果另用scripts/verify_integrated_artifacts.py核验150条逐类、300条BIO、84条配对指标行、6张图、报告链接和来源哈希。缺少本地数据/结果时不自动付费重跑。
