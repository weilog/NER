# DeepSeek 试跑、预算与缓存协议

仅执行固定 20 条 Dev 零样本试跑；本阶段不读取项目 Test。
目标是检查输出格式、字符定位、实际 usage 和延迟，不用这 20 条成绩选择最终方法。

## 本机凭据

运行器先读当前进程 `DEEPSEEK_API_KEY`，否则显式读取项目根目录 `.env`。
本机 `.env` 已被 Git 忽略；密钥不进入提示词、请求记录、报告、缓存键或结果包。
不输出 SDK 异常字符串或请求头，只记录经过限定的异常类别和 HTTP 状态码。
默认使用 `https://api.deepseek.com`；本实验拒绝非官方域名，避免把密钥发往其他地址。

## 候选运行参数

请求 `deepseek-flash`，关闭 Thinking，temperature=0，JSON 对象输出，max_tokens=2048，超时 45 秒，串行调用。
SDK 自带重试关闭。连接错误、超时、429、5xx 最多重试两次，总尝试不超过三次；其他 4xx 不重试。
退避等待为 1、2 秒。不对空响应、非法 JSON、截断或实体错误发起隐藏修复调用。
temperature=0 不保证完全确定；始终保留原始响应。

参考：[官方模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)、[Thinking 控制](https://api-docs.deepseek.com/guides/thinking_mode/)、[JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)。

## 提示与版本

- v1：字符索引为 `[index,字符]`；2026-10-24 首次固定 20 条试跑完成。20 条均返回合法顶层 JSON，但 44 个实体条目中 7 个把闭区间终点用于左闭右开跨度，另有 1 个重叠条目。因此未冻结参数。
- v2：索引改为 `[start,字符,end]`，直接给出每个字符之后的位置；加强 end 与末字符索引的区别、长度检查和非重叠规则。

v2 同组 20 条复核全部成功，无无效跨度或重叠输出；接口冻结文件为 `configs/llm_runtime_frozen.json`。
两版总计 40 次真实请求，累计保守记账 0.02982192 元，未知费用预留为 0。
v2 网络耗时中位数约 0.801 秒、P95 约 1.159 秒。20 条上的 Token F1 为 74.11%、实体 F1 为 63.27%，仅作试跑记录，不作主方法比较。

两版使用完全相同的 20 个 Dev ID；不换样本、不改标注、不在后处理中加一修正。
这是可见、留痕的 Dev 提示修订，不是同一次调用后的隐式修复。
v1 代码与结果摘要保存在 Git 提交 `be5e59d`，完整请求和响应在本地旧运行目录保留。

请求只有原文及索引，不含当前 Dev 样本的真实标签。十类定义、边界规则、输出 schema 均在系统提示中。
少样本构建器只允许 BIO 可转换的有效 Train 示例；当前试跑固定为 0-shot，没有真实训练示例。
实体要求 start/end/type/text；缺字段、越界、类型未知、文本不一致等进入统一评估器诊断，不自动纠正。
顶层 JSON、finish_reason、空响应或重复 JSON 键不合法时，整个样本显式 `parse_error`，按空预测评分。

## 全局预算

SQLite 账本：`runs/llm/budget.sqlite3`，全实验共用，禁止每阶段重置。
项目上限 100 元；两版试跑共用同一个阶段上限 5 元。
每次实际网络尝试前事务性预留费用，并发/重启也不会忽略已预留的费用。
预留按高峰未缓存输入价及输出上限计算；输入上界采用序列化请求 UTF-8 字节数加 512 个 framing 余量，依赖 byte-level BPE token 不超过字节数的保守假设。
收到响应后按 usage 结算，若用量超过预留则记录实额并停止后续请求，不隐瞒超额。

2026-10-24 的官方人民币快照见 `configs/deepseek_pricing_2026-10-24.json`。
分别记录缓存命中、未命中和输出用量；缓存字段缺失时按未命中价保守估算。
开始和结束跨价段时按更高价格；周末采用空闲价，工作日法定假日若未额外核验则保守按高峰价。
记账是基于 usage 的计费估算，不冒充供应商账单；没有余额查询、充值或自动修改预算。
网络超时、连接错误、5xx 或 usage 缺失时保留整个费用预留；明确被拒绝的 4xx 尝试按零计费。
未来日期发起付费调用前必须重新核验官方价格，保存当天快照。

## 缓存与恢复

缓存键包括样本 ID、完整请求（模型、温度、输出上限、Thinking、JSON 格式、提示和示例）、endpoint、提示及解析器版本。
阶段名称不进入请求缓存键，因此相同冻结请求可在试跑、300 条调参、完整 Dev 之间复用。
每个缓存记录保存 SHA256，恢复时检查校验和；运行上下文也保存数据及源码哈希。
已有运行上下文不同则拒绝覆盖，必须使用新的配置和目录。

在 SQLite 中持久化每个尝试的 raw response 后再解析；若响应已落盘但解析/导出中断，可以离线恢复，不再付费调用。
若进程在请求发出后、响应落盘前终止，是否扣费和是否完成无法确认：保留全额预留，标记失败，不自动重发。
显式失败响应也缓存，重跑不会悄悄重试直到成功。未发送的预算停止不缓存为终态。
`run.lock` 防止同时导出同一试跑；异常崩溃留下锁时，须先确认旧进程已结束，再移除该锁。

## 运行与验收

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\run_llm_pilot.py --dry-run
.\.venv\Scripts\python.exe scripts\run_llm_pilot.py
```

默认读取 `configs/llm_pilot_v2.json`。旧 v1 配置为留档，旧请求可离线评分；不能用新版源码覆盖旧运行目录。
dry-run 只准备请求并检查预算，不发网络请求。真实运行保存：

- `context.json`、`requests.jsonl`：上下文、原始请求，无凭据。
- `records.jsonl`：原始响应、逐次尝试、模型标识、usage、费用及安全的错误记录。
- `predictions.jsonl`：全部 20 个 ID，统一跨度与显式状态。
- `metrics.json`、`normalized.jsonl`：双层指标、原始跨度敏感性结果和规范化诊断。
- `summary.json`：状态、费用、用量、延迟、输出哈希与冻结检查。

网络耗时按每个样本所有尝试的网络秒数合计，排除退避等待。中位数/P95 的该口径在报告中明确标注。
若 20 条全部成功、无无效实体/重叠、无未确认费用且模型标识一致，则生成 `configs/llm_runtime_frozen.json`。
该文件只冻结运行接口和提示版本，不能作为运行正式 Test 的授权或全局配置冻结。
冻结检查失败时保留诊断并审核；不自动把任意无效条目“修到正确”，也不按较高 F1 挑试跑样本。

离线重算示例：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --gold data\processed\experiment_v2\dev_pilot.jsonl --predictions runs\llm\pilot_20_zero_shot_v2\predictions.jsonl --output runs\llm\verification\pilot-v2-offline.json
```

下一阶段才是固定 300 条 Dev 的 0/2/4/8-shot 比较，并基于试跑实际 usage 估算完整 Dev/Test 预算。
提示接口冻结前，不扩大到该阶段。

真实缓存离线复核与后续费用预测：

```powershell
.\.venv\Scripts\python.exe scripts\verify_llm_pilot.py
.\.venv\Scripts\python.exe scripts\estimate_llm_budget.py
.\.venv\Scripts\python.exe scripts\package_llm_pilot.py
```

费用预测使用实际试跑的最大输入 token/字节比例、输出 P95，分别增加 1.5 倍余量，按高峰未缓存价格估计。
报告 `reports/llm-budget-projection.json` 估计后续约 47.16 元；这不是费用保证，实际请求仍受全局账本约束。
该估计只从清单读取 Test 数量，不读取 Test 原文。预测次数计入试跑与调参/完整 Dev 间的缓存复用，后续无重试时为 5,684 次。
打包输出 `artifacts/result-packages/cluener-llm-pilot-2026-10-24.zip`，包含数据、传统模型、Dev 预测和两版 LLM 原始记录。
使用 SQLite 一致性备份，检查数据库完整性、逐文件哈希及密钥排除；该完整结果包仅保存在本地。
