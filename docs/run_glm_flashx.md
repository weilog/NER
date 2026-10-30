# 自己运行 GLM-5.3-FlashX 完整 Test

代码已接入智谱官方标准 API。创建和检查配置不会调用 API；只有显式 `--run` 才请求模型。不要执行旧 DeepSeek 运行脚本来调用 GLM。

当前本机实验曾因热点断联产生大量连接失败并触及保守预算，现使用本文末尾的“网络恢复入口”续跑；不要用最初30元入口重复启动。

## 1. 保存本机密钥

在 NLP 根目录的 `.env` 保留原有内容，新增：

```dotenv
GLM_API_KEY=填写实际Key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
GLM_MODEL=glm-5.3-flashx
```

使用 BigModel 标准 API Key，不能默认 Coding Plan 订阅额度可用于本实验。不要把实际值发聊天、写进代码或提交 Git。

## 2. 离线检查（不收费）

在 PowerShell：

```powershell
cd C:\Users\cdk\Desktop\NLP
.\.venv\Scripts\python.exe scripts/run_glm_test.py --check-config
```

检查 `glm_key_present: true`、模型 `glm-5.3-flashx`、每档 1209 条、四档共 4836 个原始请求及 V3。此命令只检查 Key 存在，不验证 Key 的有效性/余额/权限。缺少 Key 会打印提示；不会发任何请求，也不会创建正式结果目录。

## 3. 自己启动正式 Test（收费）

```powershell
.\.venv\Scripts\python.exe scripts/run_glm_test.py --run --workers 4 --budget-cny 30
```

固定档位顺序 0/2/4/8-shot，全部使用同一 1209 条 Test；每档最多4条并发，按原始样本顺序保存预测。预算检查与账本写入共用线程锁。无新增 Dev 调参、无隐藏修复调用。不要同时启动第二个 LLM 进程。

本轮最初串行保存333条缓存（332条成功、1条HTTP400失败），后经用户授权切换4路；原缓存逐文件哈希未变。原冻结在frozen_protocol.json保留，后续执行协议在execution_protocol_v2.json，迁移、原源码/账本快照与333条缓存哈希在scheduling_migration_v1。模型、提示、示例、数据、评分与预算没有改变，失败未重新请求。首个HTTP400未保存错误正文，原因未知；后续错误增加脱敏业务码/信息记录，文档确认的1301拒绝计入单句失败，不重试；其他未知接口错误停止批量。不能把所有400都归因为内容拒绝。

第一条正式 Test 如果遇到认证、参数、余额等不可重试错误，会保存诊断并停止批量请求；不要为了换更好答案删缓存。网络/429/5xx 最多额外重试两次。缓存中标记为 fatal_error 的响应需要先核验，再由组长决定如何处理，不自动重发。API 超时等未知收费请求保留保守费用预留。

预算 30 元是本轮上限，不是预计花费。项目累计上限仍为 100 元；运行器会读取其他已有 ledger 并取剩余额度。尚无本轮真实 usage，不能保证 30 元覆盖全程；思考会增加输出消耗。达到上限时缓存保留、停止，不自动充值或缩小 Test。账户实扣以平台账单为准。

Ctrl+C 后等待在途请求完成并落盘，再使用完全相同的命令恢复，已完整保存的请求直接读缓存，失败也保留为失败。若进程被强杀、断电或出现“已预留/已计费但无缓存”的不确定请求，停止并审核，不自动重复付费。`runs/llm/process.lock` 残留时，先确认没有运行中进程，再处理，不自动删除。

运行中的账本上界包含最多4条在途请求的费用预留，不等于账户已实扣；正常响应后按usage结算，未知收费失败继续保守保留预留。原333条串行耗时与新4路请求的单条耗时分别标记，diagnostic_summary保存execution_workers_counts；吞吐、延迟和缓存优惠可能受调度改变影响，报告须披露混合运行，不能称全部4路运行。

## 4. 看结果

固定目录 `runs/glm_flashx_test_v1/`：

- `frozen_protocol.json`：完整提示、示例、参数、数据/源码哈希、价格记录、Test IDs。
- `cache/`：原始请求/响应及解析结果，不含 Authorization/API Key。
- `ledger.json`：每次尝试和真实 usage/保守收费上界，包含重试。
- `shot_0`、`shot_2`、`shot_4`、`shot_8`：每档完整 predictions、metrics、sentence_counts、errors、diagnostic_summary、run。
- `summary.json`、`summary.csv`：四档全部完成后产生，含两级 P/R/F1、费用和延迟。

未实际执行的句子不会被伪装成空预测参与完整 Test 评分；请求中断时只保存请求缓存，整档完成后才输出完整预测与指标。中断最终文件写入造成的部分组文件会停止自动覆盖，需要审核。

四档完整运行后，使用独立离线复核入口，不需要 Key，不发送请求：

```powershell
.\.venv\Scripts\python.exe scripts/run_glm_test.py --replay
```

该命令验证原始预测与指标哈希，并用同一 V3 评估器重新计算、比较保存的指标。缺少任何完整档位会停止，不补跑模型。

V3 评估保留全部合法去重跨度，包括重叠；Token 使用 `(样本ID,字符位置,完整BIO标签)` 集合。非法跨度另报辅助惩罚指标；text 与跨度不一致只诊断、不自动移位，口径与新版 DeepSeek 一致。缺失、重复或多余样本 ID 停止评分。API/格式失败按该句空预测计分并报告失败率。

## 5. 看代码的顺序

1. `configs/glm_flashx.json`：模型、参数、价格、示例 ID 与固定哈希。
2. `scripts/pilot_shot_budget.py::pilot_messages`：共用的新提示构造；这里不运行该脚本的 main。
3. `src/ner_experiment/models/glm.py`：Key 读取、缓存、费用、重试、JSON 解析。
4. `scripts/run_glm_test.py`：离线检查 → 冻结 → 四档请求 → V3 评分 → 汇总。
5. `src/ner_experiment/evaluation/metrics.py`：同一套 V3 评估器，没有因接入 GLM 修改。

## 参数差异与报告边界

FlashX 强制思考，固定 `thinking.type=enabled`、`reasoning_effort=low`、`temperature=0`、`max_tokens=8192`、JSON object、非流式。8192 是输出预算上限，不是保证实际消耗；比 DeepSeek 的 2048 更大以给思考和最终 JSON 留空间。仍可能截断，截断计格式失败，不隐藏重试修复。

DeepSeek 原实验为关闭思考。本轮只能表述为固定提示下两个 API 系统/配置的效果与费用比较，不能说严格隔离了模型本身的影响，也不声称温度 0 保证完全确定。`low` 是本轮预算取舍，不是依据 Test 选出的最佳参数。不能根据本轮 Test 调整参数或示例后继续声称同一轮盲测。

参考（核对于 2026-10-27）：

- 模型/强制思考：https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash
- 标准接口：https://docs.bigmodel.cn/api-reference/模型-api/对话补全
- 标准价格输入 2 元/百万、缓存输入 0.57 元/百万、输出 7 元/百万：https://docs.bigmodel.cn/cn/guide/start/pricing

价格和账户余额需以运行当日平台为准。报告保留模型实际返回标识、调用日期和完整参数，别名不等于不可变模型权重版本。原始数据、缓存和预测仅保存在本地，不提交公开 GitHub。

## 网络恢复入口（当前本机流程）

热点曾断联。旧日志仅保留URLError等异常类别，无法追溯每次失败的底层原因。实际项目Python当前不带Key的官方站点连接与证书校验成功；另一套系统Python的证书错误不作为旧实验的根因证据。没有禁用TLS校验。

停止时：0/2-shot各1209条完整，4-shot保存200条（83成功、117调用失败），8-shot未开始；共2618条缓存。已返回usage费用估算3.82845312元，原始保守账本29.94697512元，约26.12元是未核实收费的失败预留。用户要求先继续，未提供账单，因此全部预留保持原样；在原项目100元预算内，将本轮保守上限调整到40元。没有把未知费用当作零，也没有重置或删除账本。

```powershell
# 离线核对，无API调用
.\.venv\Scripts\python.exe scripts/resume_glm_test.py --retain-unknown-costs --budget-cny 40
# 用户明确授权后继续真实请求
.\.venv\Scripts\python.exe scripts/resume_glm_test.py --run --retain-unknown-costs --budget-cny 40
```

仍为4路，原始模型/提示/示例/参数/评分保持不变。2618条缓存与0/2-shot产物全部有恢复前哈希快照；缓存中的失败也按原样保留，不重新请求。4-shot分数会受到断网失败空预测影响，不能将下降全归因于模型或示例数量。

新增`glm_recovery.py`提供连续连接失败停止：8次连续连接失败尝试触发整批停发，等待在途请求落盘；明确的DNS、拒绝连接、TLS证书验证失败立即停发，并只对新记录中能证明未发出HTTP请求的预留释放金额。历史URLError没有证据，预留不释放。超时、远端断连等收费不确定仍保守记账。每句最多三次尝试的原规则不变，TLS验证始终开启。

执行记录在`network_recovery_v1/execution.json`，包含新源码哈希、预算40/100、原模型协议哈希及保留策略。原冻结和执行协议v2、已完成组的run/metrics不改写。恢复前曾因JSON数值30与30.0的哈希表示差异在发请求前停止，原记录单独归档；已修正为直接使用实际冻结记录的规范表示。

最终summary区分原始账本上界和恢复后预算口径，不把保守预留冒充平台实扣。如果之后核对官方账单，`reconcile_glm_bill.py`可另存带账本前缀哈希的核验记录，原账本不改写；本次用户未核验，因此没有生成该核验文件。

## 充值后与MiMo同时运行（当前后续流程）

用户报告已充值并明确要求同时启动GLM，使用独立入口`resume_glm_after_topup.py`。启动前核对了原全局锁PID确实为MiMo进程；保留MiMo的全局锁，GLM另用本目录provider.lock，防止重复GLM进程。并行期间只运行这两个付费实验，不再启动第三个付费入口。

预算采取预分配上限：GLM40元（包含全部旧未知预留）、MiMo20元，其他已有实验保守账本12.2326584元，两个模型即使各自用满合计也至多72.2326584元，低于项目100元；不能按“另一个模型此刻只花了多少”独立花掉同一份剩余额度。两组各4路，MiMo额外全局90 RPM。

两条8-shot余额不足记录train:1138、train:1148，原先仅尝试1次、无模型答案。充值后按用户授权恢复：原JSON逐字节归档至quota_recovery_v1/prior_cache，原账本所有条目不改写；按原请求继续，每条累计最多三次，实际第二次均已成功。已有成功、格式失败、网络失败及0/2/4-shot评分不重跑。初始余额错误与最终样本状态应分别报告，不声称从未发生失败。

本次实际启动使用已核验的MiMo锁PID291244：

```powershell
.\.venv\Scripts\python.exe scripts/resume_glm_after_topup.py --run --mimo-owner-pid 291244
```

该PID仅对应本次进程；以后不能照抄过期PID，需先确认锁持有者确为MiMo。重复启动会被provider.lock拒绝；不要删除锁强行开第二个GLM。模型、参数、提示、示例与V3协议未改；新源码/预算分配/历史归档校验记录在quota_recovery_v1，最终完成后写同一GLM summary。之前GLM暂停ZIP仍保留，暂停状态是当时快照，不代表最新进程状态。
