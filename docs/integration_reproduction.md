# 15组结果的离线整合

最新报告：`reports/integrated_results.md`。传统词典/HMM/CRF及三家API各0/2/4/8-shot，共15组，统一1209句Test和原始评分规则；没有新API调用。

## 已有本地结果上重建

2026-10-30新增一键离线入口，已在本机实际重评15组并通过136项测试：

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_main_results.py
```

该入口不会读密钥或调用模型；缺少本地数据/预测包时明确停止。日志保存在runs/integrated_v1_analysis/reproduction_current.log。下方命令用于逐步检查同一流程。

在NLP项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/integrate_v1_results.py
.\.venv\Scripts\python.exe scripts/plot_v1_results.py
.\.venv\Scripts\python.exe scripts/verify_v1_artifacts.py
.\.venv\Scripts\python.exe -m pytest -q
```

首个脚本校验固定数据SHA256、预测ID和15组保存指标，保护原结果及账本；输出13组Dev参数的原始口径重评，不改变历史选参。每次句子级Bootstrap固定2000次、seed718853259；重新汇总TP/FP/FN计算micro F1。第二个脚本仅从源表绘图。测试仅收集`tests`，不导入`runs`内历史源码快照。

## 输入与输出

| 内容 | 本地路径 |
|---|---|
| 固定数据 | `data/processed/cluener2020_project/{train,dev,test}.jsonl` |
| 划分与哈希 | `data/splits/cluener2020/manifest.json` |
| 传统Test预测 | `runs/final/{dictionary,hmm,crf}/predictions.jsonl` |
| 当前原始口径指标 | `runs/integrated_v1_analysis/{method}/metrics.json` |
| Dev参数预测 | `runs/baselines/`，含9组HMM无约束对照 |
| DeepSeek | `runs/llm_text_test_v1/` |
| GLM | `runs/glm_flashx_test_v1/` |
| MiMo | `runs/mimo_flash_test_v1/` |
| 主表、逐类、配对区间、费用、诊断 | `reports/tables/integrated_v1/` |
| 6图PNG/SVG/PDF与QA | `reports/figures/integrated_v1/` |
| 重算计数、错误与3例来源 | `runs/integrated_v1_analysis/` |

原始结果不在GitHub中：只克隆源码无法直接重算本次结果。需取得有权使用的本地结果包与语料，保持上述路径，先核对`reports/tables/integrated_v1/verification.json`源哈希。许可未解决前，完整数据、模型和原始响应仅保留本地。不得为解决缺文件而直接发起付费重跑；新实验需另建结果目录并保留冻结记录。

## 本地私有结果包

`scripts/package_integrated_results.py`会先核对整合来源，再保存源码、依赖、数据、传统模型、三家API的请求响应与预测、历史费用账本及报告。它在`runs/submission_packages/`下新建带时间标识的目录，验证ZIP CRC、每个成员SHA256与源文件不变；不覆盖旧包，不包含真实.env、私钥或虚拟环境，也不会调用API。生成后按脚本打印的位置读取verification.json。由于包含语料和原始响应，此包只留本地，不上传公开仓库。

```powershell
.\.venv\Scripts\python.exe scripts/package_integrated_results.py
.\.venv\Scripts\python.exe scripts/package_integrated_results.py --verify <实际ZIP路径>
```

## 阅读边界

- 完整Test是主成绩，含网络/格式/内容失败。共同1088句表是事后条件诊断，所有15组同样排除121个网络失败ID；不是修复后的全量结果。
- 三API共享Prompt和八条示例；思考模式、输出限制、调度和预训练不同，比较的是系统。
- 历史Dev选传统参数。新版四档LLM按用户指令直接Test，不把Test最高分当Dev选择；多次观察Test及Prompt修订的限制保留。
- 已知usage估算、未知预留、保守账本合计分别报告；控制台实际扣费未核验。
- 42组区间为探索性，未校正多重比较；不包含训练/API/换例句的波动。
- 本机重算和132项测试不能替代异机复现、成员贡献声明或真实录屏。

## 图表检查

可选的PDF检查依赖见`requirements-figure-qa.txt`。如本地有对应QA工具，显式传入工具目录：

```powershell
.\.venv\Scripts\python.exe scripts/plot_v1_results.py --qa-tools-dir <本地QA脚本目录>
.\.venv\Scripts\python.exe scripts/qa_v1_figures.py --qa-tools-dir <本地QA脚本目录>
```

检查脚本为`validate_figure.py`、`audit_pdf_text.py`、`audit_figure_collisions.py`、`audit_panel_alignment.py`。它们属于可选图表制作工具，不是评估器运行依赖。源图文件、脚本和工具哈希随结果保存；六张最终图已逐张查看。PNG 300dpi仅作课程预览，PDF/SVG保留可编辑矢量文字；未声明期刊投稿认证。
