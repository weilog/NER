"""Generate evidence tables/figures/review forms from actual saved predictions only."""
import csv
from collections import defaultdict
import json
import numpy as np
from ner_experiment.common import ROOT, LABELS, load_json, read_jsonl, save_json, object_hash
from ner_experiment.evaluation.metrics import paired_bootstrap

def csv_file(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    tables=ROOT/'reports/tables';tables.mkdir(parents=True,exist_ok=True)
    selection=load_json(ROOT/'runs/baselines/selection.json');seed=load_json(ROOT/'configs/split_seed.json')['seed']
    final=all((ROOT/f'runs/final/{name}/metrics.json').exists() for name in ['dictionary','hmm','crf','llm_zero','llm_few'])
    if final:
        folders={name:ROOT/'runs/final'/name for name in ['dictionary','hmm','crf','llm_zero','llm_few']};split='Test'
    else:
        folders={k:ROOT/'runs/baselines'/v for k,v in selection['selected'].items()};split='Dev (preliminary)'
        llm_selection=ROOT/'runs/llm/selection.json'
        if llm_selection.exists():
            shots=load_json(llm_selection)['best_shots']
            folders.update(llm_zero=ROOT/'runs/llm/dev_0',llm_few=ROOT/f'runs/llm/dev_{shots}')
    metrics={name:load_json(p/'metrics.json') for name,p in folders.items()};main_rows=[];per_class=[];latency=[]
    for name,m in metrics.items():
        main_rows.append({'method':name,'split':split,**{f'{level}_{k}':m[level][k] for level in ['token','entity'] for k in ['precision','recall','f1','tp','fp','fn']},
            'entity_macro_f1':m['entity_macro_f1'],'character_accuracy':m['character_accuracy'],'failure_rate':m['failure_rate'],
            'invalid_sample_rate':m['invalid_sample_rate'],'illegal_bio_rate':m['illegal_bio_rate']})
        for c,v in m['entity_per_type'].items():per_class.append({'method':name,'category':c,**v})
        run=load_json(folders[name]/'run.json')
        latency.append({'method':name,'split':split,'median_seconds':run['latency_median_seconds'],'p95_seconds':run['latency_p95_seconds']})
    csv_file(tables/'main_results.csv',main_rows);csv_file(tables/'entity_per_type.csv',per_class);csv_file(tables/'latency.csv',latency)
    csv_file(tables/'baseline_parameters.csv',[{k:r.get(k,'') for k in ['name','method','config','entity_f1','token_f1','fit_seconds','predict_seconds','vocabulary_size','dev_unknown_char_rate','unconstrained_entity_f1','unconstrained_token_f1','unconstrained_illegal_bio_rate']} for r in selection['runs']])
    csv_file(tables/'error_counts.csv',[{'method':n,**{k:m['error_counts'].get(k,0) for k in ['missed','spurious','boundary','type','boundary_and_type','invalid']}} for n,m in metrics.items()])
    if (ROOT/'runs/llm/selection.json').exists():csv_file(tables/'prompt_parameters.csv',load_json(ROOT/'runs/llm/selection.json')['options'])
    ledger=load_json(ROOT/'runs/llm/ledger.json') if (ROOT/'runs/llm/ledger.json').exists() else {'attempts':[]}
    attempts=ledger['attempts'];upper=sum(a['charged_upper_cny'] for a in attempts)
    uncertain=sum(a['charged_upper_cny'] for a in attempts if a['status']!='response')
    cost={'attempts':len(attempts),'prompt_tokens':sum(a.get('usage',{}).get('prompt_tokens',0) for a in attempts),
         'completion_tokens':sum(a.get('usage',{}).get('completion_tokens',0) for a in attempts),
         'peak_price_upper_cny':upper,'off_peak_lower_cny':(upper-uncertain)/2,'uncertain_reserved_cny':uncertain,
         'note':'Price-range estimate, not invoice. All billed usage includes pilot and Dev tuning. Pending unknown requests retain worst-case reservations.'}
    save_json(tables/'cost.json',cost)
    intervals={}
    if final:
        best_method=next(k for k,v in selection['selected'].items() if v==selection['best_traditional'])
        baseline=read_jsonl(folders[best_method]/'sentence_counts.jsonl')
        for name in ['llm_zero','llm_few']:
            intervals[name]=paired_bootstrap(read_jsonl(folders[name]/'sentence_counts.jsonl'),baseline,seed,2000)
            intervals[name]['baseline']=best_method
        save_json(tables/'bootstrap.json',intervals)
    # Human review forms are never overwritten after a person has filled them.
    review=ROOT/'reports/review';review.mkdir(parents=True,exist_ok=True)
    pool=defaultdict(list)
    for name,folder in folders.items():
        for case in read_jsonl(folder/'errors.jsonl'):
            for i,e in enumerate(case['errors']):
                pool[e['kind']].append({'method':name,'sample_id':case['id'],'text':case['text'],'automatic_kind':e['kind'],
                    'gold_span':json.dumps(e['gold'],ensure_ascii=False),'predicted_span':json.dumps(e['predicted'],ensure_ascii=False),
                    'token_f1':case['token']['f1'],'entity_f1':case['entity']['f1'],'item_key':object_hash([seed,name,case['id'],i])})
    for rows in pool.values():rows.sort(key=lambda r:r['item_key'])
    selected=[]
    while len(selected)<40 and any(pool.values()):
        for kind in sorted(pool):
            if pool[kind] and len(selected)<40:selected.append(pool[kind].pop(0))
    for i,r in enumerate(selected):r.update(review_id=f'E{i+1:03d}',split=split,reviewer='',human_kind='',agreement='',notes='')
    suffix='test' if final else 'dev'
    for name,rows in [('review_40',selected),('independent_A_20',selected[:20]),('independent_B_20',selected[:20])]:
        path=review/f'{name}_{suffix}.csv'
        if not path.exists():csv_file(path,rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=ROOT/'reports/figures';figures.mkdir(parents=True,exist_ok=True)
    names=list(metrics);x=np.arange(len(names));fig,ax=plt.subplots(figsize=(9,4))
    ax.bar(x-.18,[metrics[n]['token']['f1'] for n in names],.36,label='Character BIO micro F1')
    ax.bar(x+.18,[metrics[n]['entity']['f1'] for n in names],.36,label='Strict entity micro F1')
    ax.set_xticks(x,names);ax.set_ylim(0,1);ax.set_ylabel('F1');ax.set_title(split);ax.legend();fig.tight_layout();fig.savefig(figures/'dual_metrics.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for alpha in [.01,.1,1.]:
        runs=[r for r in selection['runs'] if r['method']=='hmm' and r['config']['alpha']==alpha]
        ax.plot([r['config']['min_freq'] for r in runs],[r['entity_f1'] for r in runs],marker='o',label=f'alpha={alpha}')
    ax.set_xticks([1,2,3]);ax.set_xlabel('Minimum character frequency');ax.set_ylabel('Dev strict entity F1');ax.legend();fig.tight_layout();fig.savefig(figures/'hmm_parameters.png',dpi=180);plt.close(fig)
    lines=['# CLUENER2020 实验结果（自动生成实测部分）','',f'评估集合：**{split}**。本报告仅包含已保存预测的实测结果。人工核验、成员贡献声明与录屏仍须本人完成。','',
       '## 数据与评估协议','',f'系统随机种子 `{seed}` 只生成一次。Train/Dev/Test=9539/1343/1209；有效训练9538。唯一嵌套样本train:1506固定在Train并统一跳过。官方Dev逐字节保留；原Train完整分配。源数据许可仍待确认，仓库不发布语料。','',
       '字符级指标对20个非O BIO标签做micro统计，O→O不增加TP；严格实体指标要求起点、终点、类别同时一致。主指标使用统一BIO规范化。非法I转换、重复输出、重叠投影、无效输出及失败率见每组metrics.json。含O的accuracy只作辅助。','',
       '## 主结果','', '| 方法 | Token P | Token R | Token F1 | 实体 P | 实体 R | 实体 F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():lines.append('| '+name+' | '+' | '.join(f'{100*m[level][k]:.2f}%' for level in ['token','entity'] for k in ['precision','recall','f1'])+' |')
    token_ranking=' > '.join(sorted(metrics,key=lambda n:-metrics[n]['token']['f1']))
    entity_ranking=' > '.join(sorted(metrics,key=lambda n:-metrics[n]['entity']['f1']))
    lines+=['',f'Token F1排序：{token_ranking}。实体F1排序：{entity_ranking}。排名差异说明局部字符正确不等于完整实体正确，评估口径会影响方案判断。','',
        '![双层评估](figures/dual_metrics.png)','','## 原理实验：词表、序列约束与指标','',
        'HMM只使用Train字符计数构建词表，小于阈值的字符统一映射UNK；9组参数均在Dev选参。下表同一组概率参数仅切换Viterbi合法BIO约束，因此可单独观察解码约束的作用。非法率分母为全部字符位置，评分前统一修复非法I。','',
        '| alpha | 最低频次 | 词表大小 | Dev未登录字符率 | 无约束实体F1 | 约束实体F1 | 无约束非法率 |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in selection['runs']:
        if r['method']=='hmm':lines.append(f"| {r['config']['alpha']} | {r['config']['min_freq']} | {r['vocabulary_size']} | {r['dev_unknown_char_rate']:.4%} | {r['unconstrained_entity_f1']:.4f} | {r['entity_f1']:.4f} | {r['unconstrained_illegal_bio_rate']:.4%} |")
    hmm_runs=[r for r in selection['runs'] if r['method']=='hmm']
    low=[r for r in hmm_runs if r['config']['alpha']==.01]
    constraint_delta=max(abs(r['entity_f1']-r['unconstrained_entity_f1']) for r in hmm_runs)
    lines += ['',f"词表证据：固定alpha=0.01，阈值1→3使词表从{low[0]['vocabulary_size']}缩至{low[2]['vocabulary_size']}，Dev未登录字符率从{low[0]['dev_unknown_char_rate']:.3%}升至{low[2]['dev_unknown_char_rate']:.3%}；F1并不随词表增大单调改善，阈值2在这组中最高。说明UNK汇聚与字符稀疏性之间存在取舍，本实验不能推广为阈值2对其他语料也最优。",'',
        f"约束证据：9组中约束/无约束规范化后实体F1最大绝对差为{constraint_delta:.6f}。本次约束没有带来可见F1提升；无约束非法转移本来极少，且统一I→B修复会消除部分差异。不能据此声称约束提高了准确率，也不能推出标签约束没有作用。",'']
    lines+=['','选择规则为Dev严格实体micro F1最大；同分依次比较训练加预测耗时、配置顺序。未按Test调整参数。完整CRF/HMM参数与耗时见tables/baseline_parameters.csv。','',
       'Token只要求单个字符BIO标签正确，一个实体边界错一位仍可能保留很多正确字符，但严格实体会同时增加一个FP和一个FN。下面实际案例同时展示两种分数；不能以accuracy替代Token P/R/F1。','',
       '## 真实错误案例与局限','', '以下归因为自动的一对一跨度匹配结果，尚未经两名成员人工核验。先移除完全匹配实体，再最大权重一对一匹配：同边界优先，同类重叠其次，其他重叠再次；余下分别为漏抽、误抽。边界与类型同时错误独列，避免重复计数。','']
    for i,r in enumerate(selected[:6]):
        lines += [f"### 案例{i+1}：{r['method']} / {r['sample_id']} / {r['automatic_kind']}",'',r['text'],'',
            f"真实跨度：`{r['gold_span']}`；预测跨度：`{r['predicted_span']}`。该句Token F1={r['token_f1']:.3f}，实体F1={r['entity_f1']:.3f}。",'']
    if (ROOT/'runs/llm/selection.json').exists():
        prompt=load_json(ROOT/'runs/llm/selection.json')
        lines+=['## 少样本数量实验','','固定300条Dev子集；表中是调参结果，不是Test。','',
            '| 示例数 | Token F1 | 实体 F1 |','|---:|---:|---:|']
        for r in prompt['options']:lines.append(f"| {r['shots']} | {r['token_f1']:.4f} | {r['entity_f1']:.4f} |")
        lines+=['',f"按预定规则选择{prompt['best_shots']}个示例。4-shot优于8-shot，说明这组固定示例中增加示例并不保证提升；不能区分示例内容与数量的全部影响，未验证LoRA或微调。",'']
    lines+=['局限：词典缺少未见名称且存在同名歧义；HMM只使用一阶状态和字符发射，较难利用长上下文；CRF依赖局部特征；LLM可能与数据集类别边界约定不同。例如提示要求排除书名号，而语料部分游戏名包含书名号，这可能造成边界错误；本轮Test后不再修提示，应留作下一轮预注册对照。具体占比见error_counts.csv；这些解释需结合人工核验，不能将标注差异一概归为模型错误。','',
       '## 成本、延迟与不确定性','',f"累计请求尝试{cost['attempts']}次；输入{cost['prompt_tokens']}、输出{cost['completion_tokens']} tokens。按官网低/高峰价格估算累计费用范围为{cost['off_peak_lower_cny']:.4f}–{upper:.4f}元；高峰价格控制100元预算，真实账单以账户为准。未知计费失败保守预留{uncertain:.4f}元。缓存重放不产生新调用。",'',
       '单价来源：[DeepSeek官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)，核对日期2026-10-23。deepseek-flash高峰价每百万token：缓存命中输入0.04元、未命中输入2元、输出8元；空闲时段分别为0.02、1、4元。预算使用高峰价，不将区间估算冒充精确账单。','',
       '延迟为每句请求/推断耗时，LLM含必要重试；缓存样本沿用首次真实调用耗时，不将读缓存耗时混入。见tables/latency.csv。8线程只是API吞吐设置，单句延迟与本地CPU模型不可直接解释为计算效率。','']
    for name,interval in intervals.items():lines.append(f"{name} − {interval['baseline']}：实体F1差值{interval['difference_a_minus_b']:.4f}，2000次句子级配对bootstrap 95%百分位区间{interval['ci95']}；seed={seed}。该区间不覆盖训练随机性、API版本变化或人工标注不确定性。")
    lines+=['','## 复现与待完成的人工工作','','运行 `python scripts/replay.py` 可离线重算全部已保存预测，要求指标与原文件完全一致。实际代码、数据/模型哈希、冻结配置及API原始响应均保存在本地结果包中。','',
       '两名成员各自填写review/independent_A_20_test.csv、independent_B_20_test.csv，再共同复核40条；不要把自动归因当成人工确认。至少核验报告的6个案例并写出具体原因。贡献声明、AI使用声明与真实演示录屏由成员补充。','',
       'AI辅助完成了代码实现、自动运行、图表与报告草稿；没有替代两人的独立人工判断。大模型零/少样本实验不支持有关未实施LoRA或微调的结论。']
    (ROOT/'reports/experiment_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report generated:',split, 'cost upper=',upper)

if __name__=='__main__':main()
