"""Render the offline integrated report from verified computed results."""
import json
from ner_experiment.common import ROOT


def write_report(specs, metrics, diagnostics, costs, protocols, bootstrap, subset_rows, subset_record,
                 parameters, selection, cases, project_cost, now):
    names = {s['method']: s['display'] for s in specs}
    pct = lambda x: f'{100*x:.2f}'
    lines = ['# CLUENER2020：15组方法的统一实验结果', '', f'生成时间：{now}。本报告仅离线读取已有预测，新增 API 请求为零。', '',
        '## 数据与评估口径', '',
        'Train 9,539、Dev 1,343、Test 1,209；有效训练样本9,538条。唯一嵌套记录固定在Train并跳过训练和示例选择。划分种子718853259；官方Dev不变，官方无标签Test不参与评分。文件哈希和预测来源见本地 `runs/integrated_v1_analysis/verification.json`。', '',
        '全部15组使用相同1,209句Test及原始评估器。实体指标要求起点、终点、类型完全相同；Token指标先按起点优先、同起点较长优先、同跨度固定类别顺序将重叠预测投影为单标签BIO，再对20个非O标签计算micro P/R/F1。实体指标保留全部合法去重跨度，越界或未知类型等无效项另计实体FP。两级指标的投影处理不同，须分别解释。调用失败和格式失败按空预测评分。Accuracy只作辅助指标。', '',
        '这是一组系统比较：词典/HMM/CRF使用完整有效Train，LLM使用预训练知识及0/2/4/8个Train示例；模型容量、预训练数据、推理模式和输出上限未控制。不能据此推导某种架构普遍更优。公开数据可能进入模型预训练，零样本不等于从未见过相关数据。', '',
        '## 完整Test主结果', '', '以下P/R/F1单位均为%。顺序固定，不按Test分数重新挑选方案。传统参数来自历史Dev选择；新版LLM直接运行预定四档Test，没有新的完整Dev选参，不把Test最高档冒称为Dev选出的最佳模型。', '',
        '| 方法 | Token P | Token R | Token F1 | Entity P | Entity R | Entity F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in specs:
        m = metrics[s['method']]
        lines.append('| '+s['display']+' | '+' | '.join(pct(m[l][k]) for l in ['token','entity'] for k in ['precision','recall','f1'])+' |')
    lines += ['', '![15组双层F1](figures/integrated_v1/main_results.png)', '',
        '原始TP/FP/FN、固定10类macro F1、字符准确率和95%区间见 [主表CSV](tables/integrated_v1/main_results.csv)；[逐类实体指标](tables/integrated_v1/entity_per_type.csv)；[20个BIO标签指标](tables/integrated_v1/token_bio_per_label.csv)。', '',
        '## Shot数量实验', '',
        '三家API共享同一份基础Prompt和同一套八条例句，2/4/8-shot分别用前2/4/8条。2-shot共8个实体、覆盖8类；4-shot共13个实体、覆盖10类；8-shot共22个实体、覆盖10类。0-shot不提供示例。新增示例同时改变内容、类别覆盖和长度，只有一套示例池，不能把差异全归因于数量，也不能估计换一套例句的波动。部分示例为截断片段，此局限保留。', '',
        '| 模型 | 0-shot Entity F1 | 2-shot | 4-shot | 8-shot |', '|---|---:|---:|---:|---:|']
    for family in ['deepseek','glm','mimo']:
        lines.append('| '+family+' | '+' | '.join(pct(metrics[f'{family}_{n}']['entity']['f1']) for n in [0,2,4,8])+' |')
    lines += ['', '以下是实体F1的句子配对差值，单位为百分点；区间为未经多重比较校正的探索性95%区间。', '',
              '| A−B | 差值 | 95%区间 |', '|---|---:|---|']
    highlighted={('deepseek_8','deepseek_0'),('glm_8','glm_0'),('mimo_4','mimo_0'),('mimo_8','mimo_4'),('deepseek_4','mimo_4'),('deepseek_8','mimo_8')}
    for p in bootstrap['comparisons']:
        if (p['a'],p['b']) in highlighted:
            d=p['entity'];lo,hi=d['ci95']
            lines.append(f"| {names[p['a']]} − {names[p['b']]} | {100*d['difference_a_minus_b']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['',
        'DeepSeek的4-shot点估计低于2-shot，8-shot最高；MiMo的8-shot点估计低于4-shot。增加示例没有在所有模型和所有档位持续改善效果。MiMo的8−4差值区间包含0，不能将一次下降解释为稳定退化；DeepSeek与MiMo在8-shot的差值区间也包含0，不能据此宣称前者稳定领先。GLM的4-shot受117句网络失败影响，不能将其下降直接解释为示例过多。', '',
        '![实体Shot曲线](figures/integrated_v1/entity_shot_curve.png)', '',
        '![Token Shot曲线](figures/integrated_v1/token_shot_curve.png)', '',
        '## 失败记录与网络影响', '',
        '| 方法 | 网络失败 | 其他API失败（含拒绝） | 格式失败 | 失败句含真实实体数 | 占该组实体FN |',
        '|---|---:|---:|---:|---:|---:|']
    for d in diagnostics:
        if d['failure_rate']:
            lines.append(f"| {names[d['method']]} | {d['network_error_count']} | {d['other_api_error_count']+d['provider_refusal_count']} | {d['format_error_count']} | {d['failed_gold_entities']} | {pct(d['failed_entity_fn_share'])}% |")
    lines += ['',
        'GLM 4-shot的117条连接失败保留为失败；另有1条内容拒绝。旧日志仅记录URLError类，不能追溯断言每次连接失败的底层原因。MiMo 8-shot有4条建立HTTP连接前失败。其他HTTP错误、格式错误也如实保留。无隐藏LLM修复调用、无为了提高Test分数重跑。GLM充值后仅恢复当时未得到模型回答的两条余额失败，历史请求和费用预留保留。', '',
        f"补充诊断：删除所有方法明确网络失败句ID的并集，共{subset_record['removed_samples']}句；全部15组统一使用剩余{subset_record['kept_samples']}句。格式失败、内容拒绝及未知HTTP错误仍保留。这是事后、非随机的条件子集，既不替代完整Test，也不代表修复网络后的真实全量分数。", '',
        '| 方法 | 全量Entity F1 | 共同子集Entity F1 |', '|---|---:|---:|']
    for r in subset_rows:
        lines.append(f"| {names[r['method']]} | {pct(r['full_entity_f1'])} | {pct(r['subset_entity_f1'])} |")
    lines += ['', '![共同子集诊断](figures/integrated_v1/network_subset_curve.png)', '',
        '[失败及结构诊断CSV](tables/integrated_v1/diagnostics.csv)含重叠、重复、非法实体、BIO非法转移与文本/跨度不符；[固定一对一错误归因统计](tables/integrated_v1/error_counts.csv)含边界、类型、边界兼类型、漏抽、误抽。文本不符不会自动移动跨度，避免悄悄修正输出。MiMo 0/2-shot的17/11条格式失败中，分别16/11条返回finish_reason=abort；仅凭该字段不能确定服务器中断原因，不能全称为JSON语法错误。', '',
        '## 配对不确定性', '',
        '固定种子718853259，句子级有放回重采样2,000次。每次所有模型使用相同抽样ID，先汇总TP/FP/FN再算全局micro F1，绝非平均每句F1。报告百分位95%区间；42组配对同时覆盖12个LLM对CRF、同Shot跨模型及各模型Shot间比较。它们是探索性、未经多重比较校正的区间，不包括重新训练、API随机性和示例池改变的波动。', '',
        '| 比较（A−B） | Entity F1差值/百分点 | 95%区间 |', '|---|---:|---|']
    for p in bootstrap['comparisons']:
        if p['b'] == 'crf':
            d=p['entity'];lo,hi=d['ci95']
            lines.append(f"| {names[p['a']]} − CRF | {100*d['difference_a_minus_b']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['', '![与CRF的配对差值](figures/integrated_v1/paired_differences.png)', '',
        '[全部配对区间](tables/integrated_v1/bootstrap_comparisons.csv)。GLM 4-shot与CRF的差值包含网络失败影响；不能用边际区间是否重叠代替配对差值分析。', '',
        '## 成本、延迟与接口条件', '',
        '费用沿用调用时记录的定价，以实际usage估算，不重新按今天价格计费。DeepSeek用历史峰时估算；GLM、MiMo按各自记录价格和缓存命中计。不是控制台账单。未知请求费用预留单列，不冒称实际扣费。', '',
        '| 方法 | 已返回usage估算/元 | 未知费用预留/元 | 原请求延迟中位数/s | P95/s |', '|---|---:|---:|---:|---:|']
    for c in costs:
        lines.append(f"| {names[c['method']]} | {c['known_usage_estimate_cny']:.4f} | {c['unknown_reservations_cny']:.4f} | {c['latency_median_seconds']:.4f} | {c['latency_p95_seconds']:.4f} |")
    lines += ['',
        f"全项目（含旧Prompt、Dev和试跑）usage已知部分约{project_cost['known_usage_estimate_cny']:.4f}元，未知预留{project_cost['unknown_reservations_cny']:.4f}元，账本保守合计{project_cost['gross_upper_cny']:.4f}元。仍低于100元实验预算，但实际扣费须以平台账单核对。", '',
        '![四档已知费用](figures/integrated_v1/cost_curve.png)', '',
        '传统模型表中0元指无API收费，不含CPU、训练和人工成本。LLM原请求延迟包含客户端等待/限流和重试；MiMo有90 RPM限制，GLM先串行后并行。不能把此表当作纯模型推理速度排名。MiMo另保存HTTP耗时，其他服务没有同口径字段，因此不补造跨服务HTTP延迟。MiMo 2-shot费用低于0-shot与实际缓存命中有关，不代表增加示例一般会降价。', '',
        '| 模型 | 推理模式 | 推理强度 | 输出上限 | temperature | 实际并发分布（条数） |', '|---|---|---|---:|---:|---|']
    for p in protocols:
        if p['shots'] == 0:
            lines.append(f"| {p['requested_model']} | {p['thinking']} | {p['reasoning_effort'] or '—'} | {p['output_limit']} | {p['temperature']} | `{p['worker_counts']}` |")
    lines += ['',
        '并发分布在四档间可能不同；完整每档协议、返回模型标识、UTC调用日期和并发见 [接口记录](tables/integrated_v1/api_protocols.csv)。模型别名不等于不可变权重版本。固定temperature也不保证API完全确定。DeepSeek/MiMo关闭thinking，GLM开启thinking且输出上限不同，模式本身构成比较条件差异。', '',
        '[逐档费用和延迟](tables/integrated_v1/cost_latency.csv)；[全项目分账本汇总](tables/integrated_v1/project_cost.json)。', '',
        '## 课程原理证据与参数实验', '',
        '### 词表与未登录字符', '',
        '以下固定HMM平滑系数0.01，仅改变最低字符频次；Dev结果统一离线重评。词频阈值改变发射词表及UNK比例，而不是单纯缩小文件。', '',
        '| 配置 | 词表大小 | Dev未登录字符率 | Token F1 | Entity F1 |', '|---|---:|---:|---:|---:|']
    for r in parameters:
        cfg=json.loads(r['config'])
        if r['method']=='hmm' and cfg.get('alpha')==.01:
            lines.append(f"| `{r['config']}` | {r['vocabulary_size']} | {pct(r['dev_unknown_char_rate'])}% | {pct(r['v3_dev_token_f1'])} | {pct(r['v3_dev_entity_f1'])} |")
    hmms=[r for r in parameters if r['method']=='hmm']
    lines += ['', '在alpha=0.01下，阈值1→2使词表3593→3019、未登录率0.1811%→0.3741%，实体F1由48.29%微升到48.45%；阈值3时为48.42%。这是有限的平滑/稀疏权衡证据，不支持“词表越小越好”的普遍结论。']
    max_delta=max(abs(r['v3_dev_entity_f1']-r['unconstrained_entity_f1']) for r in hmms)
    max_illegal=max(r['unconstrained_illegal_bio_rate'] for r in hmms)
    lines += ['',
        '### 序列标签约束', '',
        f'9组HMM均在相同参数下比较普通与BIO合法约束Viterbi。观察到最大实体F1差值{max_delta*100:.4f}个百分点；普通解码最大非法BIO转移率{max_illegal*100:.4f}%。约束在数学上排除非法路径，但本数据和参数下是否改善必须看上述实测，不能因为加入约束就宣称有增益。', '',
        '### 指标口径', '',
        'HMM的Token F1高于词典，严格实体F1却低于词典，证明部分字符命中与完整边界类型正确会产生不同排序。逐字符O准确率不能替代实体标签P/R/F1。下方真实边界例同时给出两级计数，支撑这一差异。', '',
        '### CRF正则与Dev选参', '',
        '| 配置 | Dev Token F1 | Dev Entity F1 | 历史选中 |', '|---|---:|---:|---|']
    for r in parameters:
        if r['method']=='crf':
            lines.append(f"| `{r['config']}` | {pct(r['v3_dev_token_f1'])} | {pct(r['v3_dev_entity_f1'])} | {'是' if r['selected'] else '否'} |")
    lines += ['',
        f"历史选择：`{json.dumps(selection['selected'],ensure_ascii=False)}`。本次仅重评已有Dev预测，不根据Test重新选参。全部13组参数、训练/预测耗时及9组无约束结果见 [参数表](tables/integrated_v1/baseline_dev_parameters.csv)。", '',
        '## 三个真实失败案例', '',
        '以下由AI基于保存的文本、标注和原始预测归因；是描述性例证，不冒充两人独立人工审核，也不代表错误频率。错误原因描述限于可观察的边界/类型/漏抽表现，不推测模型内部思维。']
    kind_names={'boundary':'边界错误','type':'类型错误','missed':'漏抽'}
    for c in cases:
        lines += ['', f"### {kind_names[c['kind']]}：{names[c['method']]}，{c['id']}", '',
            f"原文：{c['text']}", '',
            '真实标注：`'+json.dumps(c['gold_entities'],ensure_ascii=False)+'`', '',
            '实际配对错误：`'+json.dumps(c['matched_error'],ensure_ascii=False)+'`', '',
            f"Token TP/FP/FN={c['token']['tp']}/{c['token']['fp']}/{c['token']['fn']}，F1={pct(c['token']['f1'])}%；实体 TP/FP/FN={c['entity']['tp']}/{c['entity']['fp']}/{c['entity']['fn']}，F1={pct(c['entity']['f1'])}%。",
            {'boundary':'预测与真实实体有字符重合，但边界不同；字符部分命中仍能得分，严格实体整体不匹配。',
             'type':'跨度相同而类别不同，属于类别决策错误；对预测类型计FP，对真实类型计FN。',
             'missed':'真实实体没有匹配预测；该句调用成功，故这次漏抽不能归为网络失败。'}[c['kind']]]
    lines += ['', '完整原始预测与选例依据：本地 `runs/integrated_v1_analysis/representative_cases.json`；更广的历史八例见 [精简错误分析](concise_error_analysis.md)。', '',
        '## 复现与提交边界', '',
        '在项目根目录执行（完全离线，不读.env）：', '',
        '```powershell', '.\\.venv\\Scripts\\python.exe scripts/integrate_results.py',
        '.\\.venv\\Scripts\\python.exe scripts/plot_integrated_results.py', '.\\.venv\\Scripts\\python.exe -m pytest -q', '```', '',
        '本机重算已保存预测不等于另一台电脑独立复现；同一数据下的点估计也不等于可推广到所有领域。历史Test已被多次观察、Prompt曾修订，本报告不是首次盲测声明。不同示例池/随机调用稳定性仍未测。数据许可未解决，原始数据、模型和完整响应仅保留本地，不补写开放许可。', '',
        '提交前仍需成员核对贡献声明和AI使用声明终稿、录制真实运行与难例演示、完成另一环境独立复现。原40例/20例人工表是可选扩展，未冒称已完成。', '']
    (ROOT/'reports/integrated_results.md').write_text('\n'.join(lines),encoding='utf-8')
