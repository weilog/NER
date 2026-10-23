# CLUENER2020 命名实体识别课程实验

## 当前状态

已实现词典、HMM、CRF、DeepSeek零样本/少样本，字符BIO与严格实体双重P/R/F1。实际结果由`reports/experiment_results.md`汇总。人工复核、贡献声明与演示录屏不能由自动实验替代。

## 数据

- data/raw/cluener2020：原压缩包、原文件和原字段 JSONL。
- data/processed/cluener2020：统一字符级实体跨度、BIO 格式。
- data/manifests/cluener2020：统计、哈希、标签映射、重复和重叠记录。
- docs/data_preparation.md：数据审计结果与限制。

原始数据保持不变；项目另划分9539/1343/1209。官方test无标签不评分；唯一嵌套样本train:1506固定在Train并跳过模型训练及提示，有效Train9538。许可仍待确认。

## 使用

Git 仓库不包含原始语料、处理后数据或虚拟环境。克隆后，先按 `docs/environment.md` 创建环境，再将自行取得的 `cluener_public.zip` 放入 `data/raw/cluener2020/`（目录不存在时创建）。来源项目：https://github.com/CLUEbenchmark/CLUENER2020 。

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_cluener.py
.\.venv\Scripts\python.exe scripts\refresh_split.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

环境说明见 docs/environment.md；精确依赖见 requirements-lock.txt。密钥放本地.env（自动读取）或环境变量，.env.example是空模板，已有.env不要覆盖。

系统随机种子718853259只生成一次并持久保存，未按表现挑选。Dev字节不变，Test从原Train多标签迭代分层抽取。ID、哈希与统计见data/splits/cluener2020；旧seed42划分已归档。

## 一键实验与离线复核

```powershell
# 会进行付费API调用，累计100元上限，自动复用已有缓存
.\.venv\Scripts\python.exe scripts\run_experiment.py all
# 无API调用：要求保存预测离线重算与原指标完全一致，再生成报告
.\.venv\Scripts\python.exe scripts\run_experiment.py offline
```

分阶段依次运行`run_baselines.py`、`run_llm.py --stage pilot`、`run_llm.py --stage dev`、`run_final.py`、`replay.py`、`report.py`、`package_results.py`，均位于scripts下。正式Test需完整Dev选参后冻结配置；冻结后代码或配置变化会拒绝继续。

Token指原文字符，不是模型子词。对20个非O完整BIO标签做micro统计，实体要求起点/终点/类别同时匹配，Accuracy仅作辅助。失败计空预测、无效实体计总体实体FP、重叠按统一规则投影；详见docs/experiment_protocol.md。

`runs`保存预测、模型、API原始响应与预算账本，仅留本地。`configs/frozen_experiment.json`为Test前冻结记录；`reports`存放结果表、图和人工审核空表。结果包保存到同级NLP_archives，不包含.env。中断可用缓存恢复；遗留process.lock必须先确认记录PID已退出才人工删除。不要删除账本继续同一轮付费调用。

全新克隆中没有本地训练结果时，`all`将已发布冻结记录备份至`runs/reference`，本轮Dev完成后生成新的冻结；已有结果时严格保留冻结检查。缓存重放可完全一致，但重新调用远程API即使temperature=0也不保证逐字节复现。异机复现仍应由成员实际核验。

已完成同机完整缓存复现：26项测试通过，34份预测离线重算一致且未新增API请求，证据见reports/reproduction.log及reports/tables/verification.json。原冻结记录与一次仅JSON键顺序的元数据修正均保留于configs，未按Test改动模型或提示。

## 旧项目归档

Few-NERD 数据与处理材料存放在 C:/Users/cdk/Desktop/NLP_archives/fewnerd_before_cluener_2026-10-22.zip。归档含逐文件 SHA256 清单，移出活动目录前已验证全部归档文件。
