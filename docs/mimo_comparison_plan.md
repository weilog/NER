# MiMo-V2.6-Flash 接入准备

用户选择小米官方按量API，模型为`mimo-v2.6-flash`。最新状态：2026-10-28四档完整Test已全部完成，各1209条、共4836条，离线V3重评通过，结果在summary.json/csv，完成校验在completion_verification.json。已返回usage估算3.86101944元，非平台实扣声明。历史paused_status.json和user_resume_20261028保留，仅表示当时的暂停与恢复。最后32条因连接被拒绝短暂停批后续跑；不重跑缓存中的失败。最终保留0/2-shot格式失败17/11条、8-shot网络失败4条，分析中需要披露。

在现有本机`.env`中追加（不要覆盖DeepSeek/GLM配置）：

```dotenv
MIMO_API_KEY=实际Key只保存在本机
MIMO_BASE_URL=https://api.xiaomimimo.com/v1
MIMO_MODEL=mimo-v2.6-flash
```

按量API与Token Plan的Key/端点不同，本轮按用户确认使用按量接口。实际Key不发聊天、不提交GitHub。

沿用9538条有效Train、固定1209条Test、八条示例嵌套前缀和V3评估；独立MiMo输出目录/缓存/账本，不复用其他厂商响应。固定比较0/2/4/8-shot。官方文档确认`thinking.type=disabled`、`temperature=0`、`max_completion_tokens=2048`、JSON object模式及非流式均有对应参数；本轮显式固定这些值以接近DeepSeek配置。JSON只保证语法，实体内容仍需同一解析器与评估器检查；参数是否在真实调用生效还未实测。

官方模型页列出100 RPM，运行器最多4路并发并设置全局90 RPM节流（含重试），重启时从账本加载最近请求时间；并发数不能替代速率限制，账户其他程序的调用仍可能占额度。连续8次连接/限流/服务临时错误停批；明确DNS、拒绝连接、TLS验证失败立即停批，仅新记录能证明未发送HTTP的费用预留可释放。超时等未知费用继续保留。421内容拒绝保留为单句失败不重试，400/401/402/403等接口、认证、余额问题立即停批；网络/429/5xx单句最多额外重试两次，没有LLM修复调用。缓存完整性或冻结不符直接停止，不能删除失败刷答案。

## 现在可用的命令

在NLP根目录的PowerShell：

```powershell
# 完全离线，不构造调用器、不联网、不收费；只显示Key是否存在
.\.venv\Scripts\python.exe scripts/run_mimo_test.py --check-config
# 收费：只有用户决定启动后才执行，四档完整Test，共4836个原始请求
.\.venv\Scripts\python.exe scripts/run_mimo_test.py --run --budget-cny 20
# 四档全部完成后，离线复核预测与两级指标，不请求模型
.\.venv\Scripts\python.exe scripts/run_mimo_test.py --replay
```

默认输出目录`runs/mimo_flash_test_v1`。仅显式`--run`发送请求；中断后使用同一命令复用已完成缓存。失败保持失败，未请求样本不伪装成空预测完成评分。非空部分评分目录、无法判断是否扣费的无缓存请求、旧锁等需先审核，不自动重写。

默认本轮上限20元，原项目累计上限100元；20元是保护额度，不是已验证报价。启动前账本约45.0313元（包含GLM未知预留），因此还有约54.9687元保守项目余额，不能因换厂商而清零。实际账户余额和单价另以平台为准，不自动充值。即使API每次都很快，90 RPM下4836次请求也至少约54分钟，额外重试或延迟会增加时间。

每档目录保存predictions、metrics、sentence_counts、errors、diagnostic_summary、rate_timing和run；cache保存脱敏请求/原始响应，ledger记录usage、单次HTTP耗时与费用预留。完整完成后生成summary.json/csv，两级P/R/F1、每类指标、失败诊断及成本均可离线检查。latency_seconds包含客户端限速等待，额外保存throttle_seconds/http_seconds，不能把限速后的总耗时直接当作与GLM完全相同条件的服务延迟。

代码阅读顺序：configs/mimo_flash.json → scripts/pilot_shot_budget.py::pilot_messages（仅共用构造函数，不跑其main）→ src/ner_experiment/models/mimo.py → scripts/run_mimo_test.py → 原V3评估器。GLM调用源码与其冻结协议未改动。

截至2026-10-27官方价格为：未缓存输入1元/百万Token、缓存输入0.02元/百万Token、输出2元/百万Token。使用官方模型页为来源，运行当天再核对；未产生实际MiMo usage，因此没有声称试跑或费用验证已完成。

原项目累计预算100元继续有效。GLM约32.80元保守记账（包含未知费用预留）和此前DeepSeek约12.23元不能因换模型而清零。新模型额度需计入同一项目预算；不自动充值。

GLM已暂停保存：0/2/4-shot完整；8-shot113/1209，尚余1096条未请求；小米模型不能代替GLM剩余样本，必须另做同一Test完整对照。GLM余额不足的fatal缓存保留，未来恢复需要明确审核，不删除缓存刷答案。4-shot包含网络故障失败，报告不得将差异全部归因于模型能力。

GLM本地检查点：`runs/checkpoints/glm_flashx_paused_20261027_225536/`。ZIP和逐文件哈希校验通过，仅组内保存，包含语料与原始响应，不公开上传。

官方资料：

- 模型/价格：https://mimo.mi.com/models/en-US/mimo-v2.6-flash
- 参数及usage字段：https://mimo.mi.com/docs/en-US/api/chat/openai-api
- JSON模式：https://mimo.mi.com/docs/en-US/quick-start/usage-guide/text-generation/structured-output
- 频率限制：https://mimo.mi.com/docs/en-US/api/guidance/rate-limit
- 错误码：https://mimo.mi.com/docs/en-US/api/guidance/error-codes
