# CLUENER2020命名实体识别：源码提交包

组长：王学锋（20246118）。共15组：词典/HMM/CRF，以及DeepSeek、GLM、MiMo各0/2/4/8-shot。

## 两个文件，各有用途

| 文件 | 用途 |
|---|---|
| 20246118_王学锋_CLUENER2020命名实体识别.zip | 源码交付ZIP，小于50MB；不含数据、权重、原始响应、视频、虚拟环境或真实密钥 |
| CLUENER2020_复现资源 .zip | 通过网盘链接提供的单一外部资源ZIP，约123.3MB；含实验资源、哈希清单及演示录屏 |

两包无重复文件路径。资源包额外包含成员提供的演示录屏，录屏不参与评分；原14,935个实验资源文件哈希保持一致。ZIP有一层外部目录，新版入口自动识别，仅恢复data/、runs/。资源包有原始/处理数据、固定划分、词典/HMM/CRF共13个已训练模型、15组预测、原始API响应、账本及网络恢复记录。源码已带的src/scripts/tests/docs/configs、最终结果CSV和冻结输入清单不重复放入资源包；无关历史备份和旧报告不放入资源包。

资源下载：[百度网盘](https://pan.baidu.com/s/13Dqdbj1yVe-sErZiAeTknQ?pwd=84e5)，提取码 `84e5`。文件大小、SHA256及文件名说明见[资源链接.md](资源链接.md)。网盘分享页需先在浏览器中下载，不可直接作为 `--url` 参数。

## 环境与依赖

已知原实验环境：Windows x64、Python3.13.5、CPU，无GPU要求。先解压源码ZIP，进入其根目录，安装依赖（以下使用uv）：

```powershell
uv venv --python 3.13.5 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
```

依赖精确版本见requirements-lock.txt；requirements-figure-qa.txt只用于可选图表检查。不要将新生成的.venv打回提交包。原锁来自Windows，不冒称已经完成其他操作系统验证。

## 一条可执行复现命令

下载外部资源ZIP，保持文件名CLUENER2020_复现资源 .zip，将其放在源码解压目录内或该目录的同级位置。资源ZIP不用手工解压。文件名在`.zip`前有一个空格；如改名，请用`--materials`指定实际路径。

在源码解压根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_submission.py
```

如资源ZIP放在其他位置，仍是一条命令：

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_submission.py --materials "D:\下载\CLUENER2020_复现资源 .zip"
```

--materials接受单一ZIP文件，或包含该ZIP的目录。程序会核对配套资源ZIP的大小、SHA256、内部文件清单、与源码的路径交集及最终45个输入；遇到已有文件内容不同会停止，不覆盖。随后加载原训练流水线的CRF，现场推断完整1209句Test，重评15组最终预测并核验两类真实难例。

复现不重新训练、不读取.env、不调用大模型API、不需要API Key、不产生模型调用费用。CRF为现场推断，LLM为原保存响应的离线重评，不把回放称为现场生成。运行时数据和权重还原在源码解压目录的data/、runs/；首次需写入较多文件，建议预留1GB磁盘空间。

## 预期输出

```text
外部资源校验/还原：.../... 
完整1209条现场预测与原保存输出逐条一致。
方法                    Token F1    Entity F1
dictionary                60.41%      53.79%
hmm                       62.96%      48.12%
crf                       73.71%      68.66%
deepseek_0                74.44%      66.74%
deepseek_2                76.67%      70.18%
deepseek_4                75.94%      69.46%
deepseek_8                77.83%      71.88%
glm_0                     73.34%      64.26%
glm_2                     76.32%      67.91%
glm_4                     76.04%      68.22%
glm_8                     77.06%      70.10%
mimo_0                    74.06%      65.45%
mimo_2                    76.60%      69.63%
mimo_4                    77.89%      71.28%
mimo_8                    77.78%      70.87%
... train:2506：边界错误，Token F1=87.50%，Entity F1=0.00% ...
... train:50：类型错误，Token F1=0.00%，Entity F1=0.00% ...
PASS: SUBMISSION REPRODUCTION COMPLETE; 15 methods; no model API requests.
```

通过标准：退出码0，15组TP/FP/FN严格一致，P/R/F1按CSV精度比较通过，两条难例从最终文件读取并重新评分，出现最终PASS。省略号表示动态进度/时间或省略的讲解文本。仅运行--prepare-only只代表输入还原与校验通过，不等于评分通过。

运行记录：runs/submission_reproduction/verification.json中的status=PASS且scoring_completed=true；runs/demo_recordings/demo_*.json保存实际分数、案例、来源和计算耗时。

## 数据与随机种子

随机种子718853259，只记录一次、不按效果挑选。Train/Dev/Test=9539/1343/1209，官方Dev不变；唯一嵌套样本train:1506固定在Train并跳过训练与提示示例，实际有效Train9538句。具体ID、数据审计与文件哈希在资源包data/splits/cluener2020和data/manifests/cluener2020。官方无标签Test不参与评分。

GLM 4-shot仅恢复117条网络失败，MiMo 8-shot恢复4条；其他成功结果和原内容拒绝保留。每组均为完整1209句，没有删句或按答案质量挑选。

Token为Unicode字符，不是计费子词。字符评分按固定优先级将重叠跨度投影成单标签BIO，对20个非O完整标签计算micro P/R/F1。实体按样本、起点、终点、类别严格匹配，保留合法去重重叠，无效项计整体实体FP。两级投影不同，分别解释；规则见docs/evaluation_v1.md。

## 文件职责

- src/：数据格式、模型、API客户端、评估器完整源码，正式评分使用metrics_v1。
- scripts/：原训练/调用/统计/报告源码；当前交付入口reproduce_submission.py，单资源ZIP读取external_resources.py。
- configs/、tests/、docs/：配置、测试与实验说明；历史说明不替代本README的当前入口。
- reports/demo_inputs_manifest.json：45个最终输入与来源的冻结清单。
- reports/tables/network_recovery/main_results.csv：最终15组Token/Entity P/R/F1和计数。
- reproducibility/manifest.json：新版单一外部资源ZIP的名称、大小和哈希；已无164分块依赖。

环境与资源就绪后，完整测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

录屏（先完成资源恢复，姓名学号填本人信息）：

```powershell
.\.venv\Scripts\python.exe scripts/demo_offline.py --name "真实姓名" --student-id "真实学号"
```

录屏控制在2分钟以内，显示本人信息，展示真实难例，明确LLM为保存输出重评。不要在录屏时重新训练或跑新的付费实验。

## 提交范围与伦理

本源码ZIP含完整源码、依赖、README、项目摘要与AI声明。最终交付仍需把报告.pdf（正文8—12页，不含附录）与每人真实贡献声明合并进同一个提交ZIP；当前旧报告未同步补测结果，因此没有放入。视频填网盘链接，不进包。

数据和权重只放在外部资源ZIP，通过链接提供。当前外部资源ZIP附有录屏，源码提交ZIP不含录屏。完整原始响应不等于可任意再分发，数据许可记录据实保留。真实密钥、虚拟环境、checkpoint和私人数据不进入提交ZIP。AI使用范围和验证方式见AI使用声明.md；不虚构成员贡献、人审或异机复现。
