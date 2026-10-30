"""Offline unified V3 evaluation and shared sentence-paired bootstrap; no API imports."""
import csv
from datetime import datetime,timezone,timedelta
import io
from pathlib import Path
import sys
import numpy as np
from ner_experiment.common import ROOT,LABELS,project_data,read_jsonl,load_json,save_json,write_jsonl,file_hash
from ner_experiment.evaluation.metrics import evaluate

METHODS=['dictionary','hmm','crf','llm_0','llm_2','llm_4','llm_8']
DISPLAY={'dictionary':'Dictionary','hmm':'HMM','crf':'CRF',**{f'llm_{n}':f'LLM {n}-shot' for n in [0,2,4,8]}}
PAIRS=[(f'llm_{n}','crf') for n in [0,2,4,8]]+[(f'llm_{n}','llm_0') for n in [2,4,8]]+[
    ('llm_4','llm_2'),('llm_8','llm_4'),('llm_8','llm_2')]
OUT=ROOT/'reports/tables/current_v3'
LOCAL=ROOT/'runs/current_v3_analysis'

def csv_write(path,rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(stream.getvalue(),encoding='utf-8-sig',newline='')

def micro_f1(counts):
    a=np.asarray(counts,dtype=float)
    denom=2*a[...,0]+a[...,1]+a[...,2]
    return np.divide(2*a[...,0],denom,out=np.zeros_like(denom,dtype=float),where=denom!=0)

def shared_bootstrap(all_counts,seed,repeats=2000,pairs=PAIRS):
    names=list(all_counts)
    if not names or repeats<1:raise ValueError('Nonempty counts and positive repeats required')
    ids=[r['id'] for r in all_counts[names[0]]]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Nonempty unique sentence IDs required')
    for name in names:
        if [r['id'] for r in all_counts[name]]!=ids:raise ValueError('Paired sentence order differs')
    result={'seed':seed,'repeats':repeats,'samples':len(ids),'interval':'95% sentence-paired percentile; no multiple-comparison adjustment',
            'scope':'fixed predictions; exploratory after Test inspection; no training/API/example-pool uncertainty',
            'methods':{n:{} for n in names},'comparisons':[]}
    pair_results={(a,b):{'a':a,'b':b} for a,b in pairs}
    for a,b in pairs:
        if a not in names or b not in names:raise ValueError('Unknown comparison method')
    # Reinitialize the same seed for each metric, preserving identical draws.
    for level in ['entity','token']:
        values=np.array([[r[level] for r in all_counts[n]] for n in names],dtype=np.int64)
        if values.shape!=(len(names),len(ids),3) or np.any(values<0):raise ValueError('Invalid TP/FP/FN array')
        point=micro_f1(values.sum(axis=1))
        draws=np.empty((repeats,len(names)));rng=np.random.default_rng(seed)
        for i in range(repeats):
            idx=rng.integers(0,len(ids),len(ids))
            draws[i]=micro_f1(values[:,idx,:].sum(axis=1))
        for j,name in enumerate(names):
            result['methods'][name][level]={'estimate':float(point[j]),'ci95':np.quantile(draws[:,j],[.025,.975]).tolist()}
        for (a,b),entry in pair_results.items():
            ia,ib=names.index(a),names.index(b);delta=draws[:,ia]-draws[:,ib]
            entry[level]={'difference_a_minus_b':float(point[ia]-point[ib]),'ci95':np.quantile(delta,[.025,.975]).tolist()}
    result['comparisons']=list(pair_results.values())
    return result

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    gold=project_data('test');ids=[r['id'] for r in gold]
    if len(gold)!=1209:raise RuntimeError('Fixed Test must contain 1209 sentences')
    seed=load_json(ROOT/'configs/split_seed.json')['seed']
    folders={n:ROOT/'runs/final'/n for n in METHODS[:3]}
    folders.update({f'llm_{n}':ROOT/f'runs/llm_text_test_v1/shot_{n}' for n in [0,2,4,8]})
    metric_paths={n:ROOT/f'runs/evaluation_versions/v3/final/{n}/metrics.json' for n in METHODS[:3]}
    metric_paths.update({f'llm_{n}':folders[f'llm_{n}']/'metrics.json' for n in [0,2,4,8]})
    protected={p for folder in folders.values() for p in folder.glob('*.json*')}
    protected.update(metric_paths.values())
    protected.update((ROOT/'data/processed/cluener2020_project').glob('*.jsonl'))
    protected.update((ROOT/'runs').rglob('ledger.json'))
    before={p.relative_to(ROOT).as_posix():file_hash(p) for p in protected}
    metrics={};counts={}
    for name,folder in folders.items():
        predictions=read_jsonl(folder/'predictions.jsonl')
        lookup={p['id']:p for p in predictions}
        if len(lookup)!=len(predictions) or set(lookup)!=set(ids):raise RuntimeError('Prediction ID mismatch')
        ordered=[lookup[i] for i in ids]
        m,c,e=evaluate(gold,ordered)
        if m!=load_json(metric_paths[name]):raise RuntimeError(f'V3 replay mismatch: {name}')
        metrics[name]=m;counts[name]=c
        save_json(LOCAL/name/'metrics.json',m);write_jsonl(LOCAL/name/'sentence_counts.jsonl',c);write_jsonl(LOCAL/name/'errors.jsonl',e)
        print('VERIFIED',name,round(m['entity']['f1'],6),flush=True)
    bootstrap=shared_bootstrap(counts,seed)
    save_json(OUT/'bootstrap.json',bootstrap)
    main_rows=[];per_class=[];error_rows=[]
    new=load_json(ROOT/'runs/llm_text_test_v1/summary.json')
    extra={f'llm_{g["shots"]}':g for g in new['groups']}
    for name,m in metrics.items():
        row={'method':name,'display':DISPLAY[name],'samples':len(gold),'evaluation_protocol':'v3',
            'selection':'historical Dev-selected traditional configuration' if name in METHODS[:3] else 'fixed four-shot Test extension; no new Dev selection',
            'entity_macro_f1':m['entity_macro_f1'],'entity_with_invalid_fp_f1':m['entity_with_invalid_fp']['f1'],
            'failure_rate':m['failure_rate'],'invalid_sample_rate':m['invalid_sample_rate'],
            'overlap_samples':m['diagnostics'].get('overlap_samples',0),'invalid_entities':m['diagnostics'].get('invalid_entities',0),
            'text_span_mismatches':extra[name]['text_span_mismatches'] if name in extra else '',
            'api_peak_cost_cny':extra[name]['cost_peak_ledger_cny'] if name in extra else ''}
        for level in ['token','entity']:
            row.update({f'{level}_{key}':m[level][key] for key in ['tp','fp','fn','precision','recall','f1']})
            row[f'{level}_f1_ci_low'],row[f'{level}_f1_ci_high']=bootstrap['methods'][name][level]['ci95']
        main_rows.append(row)
        for category,score in m['entity_per_type'].items():per_class.append({'method':name,'category':category,**score})
        error_rows.append({'method':name,**{kind:m['error_counts'].get(kind,0) for kind in ['boundary','type','boundary_and_type','missed','spurious']}})
    csv_write(OUT/'main_results.csv',main_rows);csv_write(OUT/'entity_per_type.csv',per_class);csv_write(OUT/'error_counts.csv',error_rows)
    comparison_rows=[]
    for pair in bootstrap['comparisons']:
        for level in ['entity','token']:
            r=pair[level];comparison_rows.append({'a':pair['a'],'b':pair['b'],'metric':level,
                'difference_pp':100*r['difference_a_minus_b'],'ci95_low_pp':100*r['ci95'][0],'ci95_high_pp':100*r['ci95'][1],
                'contains_zero':r['ci95'][0]<=0<=r['ci95'][1],'multiplicity':'uncorrected exploratory pairwise interval'})
    csv_write(OUT/'bootstrap_comparisons.csv',comparison_rows)
    assert all(file_hash(ROOT/p)==h for p,h in before.items())
    save_json(LOCAL/'verification.json',{'checked_at':now,'samples':len(gold),'methods':METHODS,'source_hashes':before,
        'source_files_unchanged':True,'all_saved_metrics_replayed_exactly':True,'new_api_calls':0,
        'seed':seed,'repeats':2000,'comparison_family':PAIRS})
    lines=['# 当前统一 V3 结果与配对统计','',f'生成时间：{now}。固定1209条Test，全部保存预测已离线重评一致，无新增API调用。',
        '', '传统模型沿用历史Dev选参；LLM使用新版Prompt和同一套嵌套示例，按用户要求直接比较完整Test的0/2/4/8-shot，没有新版Dev选参，不以Test最高分产生选参结论。',
        '', '## 两级主结果','', '| 方法 | Token P | Token R | Token F1 | 实体 P | 实体 R | 实体 F1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():lines.append('| '+DISPLAY[name]+' | '+' | '.join(f'{m[level][key]:.2%}' for level in ['token','entity'] for key in ['precision','recall','f1'])+' |')
    lines+=['','主表全部使用V3合法去重跨度：重叠保留；Token是(样本ID,字符位置,BIO标签)集合。无效项不进入合法跨度主指标，另外报告带无效FP的辅助实体指标。4-shot有1个无效实体，其辅助F1见main_results.csv；不只展示清洗后分数。',
        '', '## 配对区间','', '2000次句子有放回重采样，seed=718853259。每次先汇总TP/FP/FN再计算micro F1；同一轮索引用于全部方法。以下是实体F1差值和95%百分位区间，单位为百分点。',
        '', '| 比较 A − B | 差值 | 95%区间 | 区间是否包含0 |', '|---|---:|---|---|']
    for pair in bootstrap['comparisons']:
        r=pair['entity'];lo,hi=[v*100 for v in r['ci95']]
        lines.append(f'| {DISPLAY[pair["a"]]} − {DISPLAY[pair["b"]]} | {100*r["difference_a_minus_b"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | '+('是' if lo<=0<=hi else '否')+' |')
    lines+=['','这些是观察Test后的探索性、逐比较区间，没有多重比较校正，不给出显著性星号或全家族显著性结论。区间含0表示现有样本不确定性下差值方向不稳定；区间不含0也不表示架构普遍优越。',
        '区间只反映当前固定模型/提示/预测的Test句子波动，不包含训练重跑、API随机性、示例池选择、模型版本或标注不确定性。',
        '', '## 当前图表','', '图表由plot_current_results.py读取本目录CSV/JSON生成，图形只展示实际观测和区间，不插值外推。',
        '', '![两级结果](figures/current_v3/main_results.png)','', '![相对CRF配对差值](figures/current_v3/paired_differences.png)','', '![Shot曲线](figures/current_v3/shot_curve.png)',
        '', '## 错误分析与解释限制','', '2026-10-27按用户要求采用AI辅助的精简逐类归因，每类一例，共8例，见[精简错误分析](concise_error_analysis.md)。40条/20条独立表是原先自定的扩展工作量，现保留为可选材料，不作为当前额外前置任务；未填写或冒充两名成员独立审核。',
        '本轮单套示例不均衡且包含截断片段；数量、类别覆盖、内容及长度共同变化。新旧Prompt边界和text字段同时改变，没有单因素消融；修订发生在查看历史Test之后。',
        '当前统一表便于系统比较，传统模型使用9538条任务标注训练，LLM只提供0/2/4/8条上下文示例；预训练数据接触不可验证，不能把差异直接归因于架构。',
        '', '## 离线复现','', '先运行python scripts/analyze_current_results.py，再运行python scripts/prepare_current_review.py与python scripts/plot_current_results.py。这些脚本不导入API客户端、不读取密钥。复核与源码哈希见runs/current_v3_analysis/verification.json。']
    (ROOT/'reports/current_v3_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ANALYSIS COMPLETE; 7 methods; 10 paired comparisons; API calls=0',flush=True)

if __name__=='__main__':main()
