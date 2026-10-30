# CLUENER2020 中文命名实体识别课程实验

本项目比较词典、HMM、CRF，以及DeepSeek、GLM、MiMo各0/2/4/8-shot，共15组方法。全部方法使用同一份1,209句带标签Test，同时报告逐字符Token和严格实体级P/R/F1。

最终主分数见[15组完整指标表](reports/tables/network_recovery/main_results.csv)，输入路径与SHA256见[独立输入清单](reports/demo_inputs_manifest.json)，评分规则见[当前评分规则](docs/evaluation_v1.md)。GLM 4-shot和MiMo 8-shot已使用网络恢复后的合并预测。

**报告状态：本地Word报告尚未同步补跑后的数字，不作为本次上传的最终结果文件。仓库中早期报告与表格为历史记录；当前主分数以本节链接为准。**

## 1 环境与依赖

已验证环境为Windows x64、Python 3.13.5。传统模型使用CPU；复核已保存结果无需GPU、PyTorch、API Key或平台余额。

依赖精确版本见`requirements-lock.txt`，项目安装配置见`pyproject.toml`。主要依赖包括NumPy、scikit-learn、sklearn-crfsuite、Matplotlib、pytest等；不要仅按本段名称安装最新版本。

首次在另一台电脑准备环境：先安装Python 3.13.5与uv，在项目根目录执行：

```powershell
uv venv --python 3.13.5 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
```

这三步只用于首次安装环境。主运行命令见第3节。原环境由uv管理，不依赖venv内自带pip。依赖锁来自上述Windows环境，不承诺未经核验的跨系统兼容性。

## 2 运行前需要哪些文件

完整源码包括`src/`、`scripts/`、`tests/`、`configs/`、`pyproject.toml`和依赖锁文件。

Git仓库不包含虚拟环境、原始语料、模型、原始API响应和完整预测。**仅克隆源码不能直接复核本次主结果。**请先取得有权使用的本地实验结果包，将其按原相对路径解压到项目根目录。结果包不能放在额外套一层的子目录中。

核心输入应当位于：

| 内容 | 路径 |
|---|---|
| 固定Train/Dev/Test | `data/processed/cluener2020_project/{train,dev,test}.jsonl` |
| 划分ID、数量及哈希 | `data/splits/cluener2020/` |
| 传统参数选择与Dev实验 | `runs/baselines/` |
| 传统Test预测 | `runs/final/{dictionary,hmm,crf}/predictions.jsonl` |
| DeepSeek四档结果 | `runs/llm_text_test_v1/` |
| GLM四档结果 | `runs/glm_flashx_test_v1/` |
| MiMo四档结果 | `runs/mimo_flash_test_v1/` |
| 网络恢复后的合并预测、核验记录 | `runs/network_recovery_20261030/`（整个目录） |
| 原结果来源核验记录 | `runs/integrated_v1_analysis/verification.json` |
| 原训练冻结与配置 | `configs/frozen_experiment.json`、`configs/experiment.json` |
| 当前完整输入清单 | `reports/demo_inputs_manifest.json`，核对其中全部相对路径 |
| 历史用量和费用 | 各轮`runs/`目录下的`ledger.json` |

应使用与当前代码、评分规则配套的完整结果包，不能只复制某一份主表CSV。打包与验证入口为`scripts/package_integrated_results.py`；原始数据、模型、完整请求响应仅私下交付有权使用者，不上传公开仓库。

## 3 一条主运行命令

环境安装完毕、配套数据和结果包就位后，在**包含本README、scripts及src的项目根目录**打开PowerShell，执行：

```powershell
.\.venv\Scripts\python.exe scripts/demo_offline.py --name "离线复核" --student-id "NOT_APPLICABLE" --auto
```

无需激活环境或修改绝对路径。这里的身份字段是复现用途的固定标识，不是学生身份；个人录屏请换成真实姓名学号，并去掉`--auto`。

此入口实际完成：

1. 按独立SHA256清单校验45个输入，覆盖固定数据、全部15组最终预测、参考CSV、原CRF模型与关键代码。
2. 加载原类CRF，对完整1,209句Test现场推断，与保存输出逐条核对。
3. 对15组最终保存预测重新评分：TP/FP/FN严格一致，P/R/F1按CSV精度设置容差，包含恢复后的GLM 4-shot与MiMo 8-shot。
4. 现场读取并重新评估边界错误和类型错误两个真实难例，输出原文、Gold、Prediction、跨度、类型与指标。

这是最终主分数的离线复核。CRF进行现场推断，LLM从保存预测重新评分；不重新训练、不读取密钥、不重新请求大模型、不重建图表或Bootstrap区间。远程API重复调用不保证逐字一致。先前整合脚本保留供历史实验追溯，不将补跑前结果称为最终结果。

## 4 预期输出与通过标准

将依次出现下面三个阶段，数值以实际终端输出为准：

```text
1 / 3 现场使用原训练流水线CRF，推断完整1209句Test
独立SHA256清单通过：45个输入，含全部15组最终预测与数据。
完整1209条现场预测与原保存输出逐条一致。
CRF Token: TP=8430  FP=2551  FN=3461
  Precision=76.77%  Recall=70.89%  F1=73.71%
CRF Entity: TP=1713  FP=629  FN=935
  Precision=73.14%  Recall=64.69%  F1=68.66%
2 / 3 对最终15组保存预测逐组重新评分（含网络恢复结果）
方法                    Token F1    Entity F1
... 共15组方法 ...
glm_4                     76.04%      68.22%
... 中间行省略 ...
mimo_8                    77.78%      70.87%
15组TP/FP/FN严格一致，P/R/F1按CSV精度校验通过
3 / 3 现场读取最终预测并验证：实体边界错误与实体类型错误
... train:2506：Token F1=87.50%，Entity F1=0.00% ...
... train:50：Token F1=0.00%，Entity F1=0.00% ...
演示完成：现场计算合计...秒，不含讲解与等待。
本地记录：runs/demo_recordings/demo_....json
```

通过标准：三个阶段完成、退出码为0、没有“演示停止”或Traceback，并输出“演示完成”。省略号表示省略中间行，不能逐字匹配；耗时因电脑不同而变化。最近一次完整离线测试为**153项通过**，其中演示专用测试17项。需要单独运行测试时执行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

当前可核验的产出：

| 内容 | 路径 |
|---|---|
| 最终15组Token/实体P/R/F1、TP/FP/FN | `reports/tables/network_recovery/main_results.csv` |
| 配置、15组最终预测路径与45个输入的SHA256 | `reports/demo_inputs_manifest.json` |
| 每次现场推断、15组重评、两类难例与用时记录 | `runs/demo_recordings/demo_*.json`（本地生成） |
| 补跑前后对比与费用 | `reports/network_recovery_results.md` |
| 恢复原响应、合并预测与核验 | `runs/network_recovery_20261030/`（仅本地结果包） |
| 两分钟录屏操作 | `docs/demo_recording.md` |

若文件缺失、配置重复或哈希不一致，程序会明确停止，不会自动训练或启动付费实验。请恢复完整配套结果包，不要删除/重建清单或冻结来绕过错误。`scripts/freeze_demo_inputs.py`是显式核验入口：已有清单仅验证，不自动覆盖；该清单是实验完成后的完整性快照，不冒称事前预注册。

## 5 数据 随机种子与评估口径

项目Train/Dev/Test为9,539/1,343/1,209句。Dev保持官方文件不变，Test从原官方Train按多标签迭代分层抽取。原始官方Test没有金标，不用于主评分。唯一嵌套记录`train:1506`保留在Train，统一跳过训练和提示示例，因此有效Train为9,538句。

随机种子固定为**718853259**，只生成并记录一次，不根据模型表现挑选。记录在`configs/split_seed.json`，划分ID与SHA256见`data/splits/cluener2020/manifest.json`；句子级Bootstrap使用同一种子、2,000次重采样。这个种子不代表能够控制外部API的所有随机性。

当前正式评分遵循`docs/evaluation_v1.md`：

- Token是原文Unicode字符；重叠预测按固定优先级投影成单标签BIO，对20个非O完整BIO标签计算micro P/R/F1，O→O不增加主指标TP。
- 实体按同一句的起点、终点、类别严格匹配，合法跨度去重并保留重叠；可枚举的无效项计整体实体FP。
- 两级指标的重叠处理不同，因此差异可能同时来自评估粒度和字符投影，不能全部归为边界粒度差异。
- 显式调用/格式失败保留并按空预测评分；缺失、重复或多余样本ID停止评分。Accuracy不能替代两套主指标。

完整1,209句是主结果。最终GLM 4-shot恢复117条网络失败，MiMo 8-shot恢复4条网络失败；成功预测和内容拒绝未重跑。恢复批次不按答案质量择优，不删除Test样本。补跑前的1,088句共同网络子集是历史诊断，不作为本次最终主结果。

## 6 重新训练或调用API

本README的默认命令不执行这部分。新实验需要另建结果目录，保留数据、模型、提示和参数冻结，不能覆盖现有结果。

传统模型运行保护见`reports/baseline_run_safety.md`；DeepSeek运行说明见`reports/llm_run_safety.md`；GLM和MiMo分别见`docs/run_glm_flashx.md`、`docs/mimo_comparison_plan.md`。API运行入口可能收费，不作为默认复核命令；传统模型重新训练使用CPU时间。

密钥只放环境变量或本地`.env`，不提交Git、不展示在录屏中。`.env.example`仅为空模板。数据许可记录仍有未解决事项，不将代码许可证自动推定为语料许可证。

## 7 代码结构与提交材料

```text
src/ner_experiment/  数据接口、传统模型、API客户端、评估器
scripts/            数据处理、实验、恢复、离线复核、报告与打包入口
tests/              数据、评估、Viterbi、缓存、预算和恢复等测试
configs/            配置、价格记录、种子、冻结记录
data/splits/        划分ID、规模、类别统计与哈希
docs/               环境、数据、当前协议、操作说明与贡献记录
reports/            报告、结果表、图表和AI声明
runs/               本地模型、响应、预测与账本，不进入公开Git
```

录屏要求为不超过2分钟、真实运行、至少一个难例，画面显示本人姓名与学号。缓存回放应明确标注，不能称为实时LLM调用。报告仍需同步最终数字；第⑧节与每人贡献声明必须据实填写；AI辅助范围见`docs/ai_usage.md`与`reports/ai_use_declaration.md`。

报告与图表的关键数字均应能定位到源CSV/JSON及运行日志。本机已有重放核验不等于另一环境已独立复现；异机核验完成后，另保存实际环境、命令和输出日志。
