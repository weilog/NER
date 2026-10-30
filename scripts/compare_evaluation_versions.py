"""Offline V1/V2/V3 comparison. Never imports model runners or an API client."""
from pathlib import Path
from ner_experiment.common import ROOT, read_jsonl, write_jsonl, load_json, save_json, file_hash, utc_now
from ner_experiment.evaluation.metrics_v1 import evaluate as v1
from ner_experiment.evaluation.metrics_v2 import evaluate as v2
from ner_experiment.evaluation.metrics import evaluate as v3, paired_bootstrap

EVALUATORS = {'v1': v1, 'v2': v2, 'v3': v3}

def main():
    output = ROOT/'runs/evaluation_versions'
    report_dir = ROOT/'reports/evaluation_versions'
    report_dir.mkdir(parents=True, exist_ok=True)
    data = {s: read_jsonl(ROOT/f'data/processed/cluener2020_project/{s}.jsonl') for s in ['dev','test']}
    dev_lookup = {r['id']:r for r in data['dev']}
    sources = sorted(p for stage in ['baselines','llm','final'] for p in (ROOT/'runs'/stage).rglob('predictions.jsonl') if (p.parent/'metrics.json').exists())
    protected = sources + [p.parent/name for p in sources for name in ['metrics.json','sentence_counts.jsonl']]
    protected += [ROOT/name for name in ['runs/llm/ledger.json','runs/llm/selection.json','runs/baselines/selection.json','configs/frozen_experiment.json']]
    protected += list((ROOT/'data/processed/cluener2020_project').glob('*.jsonl'))
    hashes = {p.relative_to(ROOT).as_posix():file_hash(p) for p in protected}
    summary = {}; all_counts = {}; affected = []
    for path in sources:
        relative = path.parent.relative_to(ROOT/'runs')
        run = relative.as_posix()
        if relative.parts[0]=='final': gold = data['test']
        elif relative.parts[0]=='llm' and relative.name=='pilot':
            gold = [dev_lookup[i] for i in load_json(ROOT/'data/splits/cluener2020/pilot_ids.json')]
        elif relative.parts[0]=='llm' and relative.name.startswith('tune_'):
            gold = [dev_lookup[i] for i in load_json(ROOT/'data/splits/cluener2020/dev_tuning_ids.json')]
        else: gold = data['dev']
        predictions = read_jsonl(path)
        summary[run] = {}; all_counts[run] = {}
        for version, evaluate in EVALUATORS.items():
            metrics, counts, cases = evaluate(gold, predictions)
            if version=='v1':
                assert metrics==load_json(path.parent/'metrics.json'), f'V1 metric mismatch: {run}'
                assert counts==read_jsonl(path.parent/'sentence_counts.jsonl'), f'V1 count mismatch: {run}'
            destination = output/version/relative
            save_json(destination/'metrics.json', metrics)
            write_jsonl(destination/'sentence_counts.jsonl', counts)
            write_jsonl(destination/'errors.jsonl', cases)
            save_json(destination/'source.json', {'predictions':path.relative_to(ROOT).as_posix(), 'sha256':file_hash(path), 'evaluation_version':version})
            summary[run][version] = metrics
            all_counts[run][version] = counts
        pred_lookup = {r['id']:r for r in predictions}
        for i, row in enumerate(gold):
            counts = {v:{k:all_counts[run][v][i][k] for k in ['entity','token']} for v in EVALUATORS}
            if counts['v1']!=counts['v2'] or counts['v1']!=counts['v3']:
                affected.append({'run':run,'id':row['id'],'text':row['text'],'gold':row['entities'],
                                 'prediction':pred_lookup[row['id']], 'counts_by_version':counts})
        print('Re-evaluated',run,flush=True)

    baseline_selection = load_json(ROOT/'runs/baselines/selection.json')
    llm_selection = load_json(ROOT/'runs/llm/selection.json')
    selection_audit = {}
    for version in EVALUATORS:
        chosen = {}
        for method in ['dictionary','hmm','crf']:
            runs = [r for r in baseline_selection['runs'] if r['method']==method]
            chosen[method] = min(runs,key=lambda r:(-summary['baselines/'+r['name']][version]['entity']['f1'],r['fit_seconds']+r['predict_seconds'],r['order']))['name']
        shots = min([r for r in llm_selection['options'] if r['shots']>0],key=lambda r:(-summary[f"llm/tune_{r['shots']}"][version]['entity']['f1'],r['latency'],r['shots']))['shots']
        selection_audit[version] = {'selected':chosen,'best_shots':shots,
            'same_as_frozen': chosen==baseline_selection['selected'] and shots==llm_selection['best_shots']}
    best_method = next(k for k,v in baseline_selection['selected'].items() if v==baseline_selection['best_traditional'])
    seed = load_json(ROOT/'configs/split_seed.json')['seed']
    bootstrap = {v:{m:paired_bootstrap(all_counts['final/'+m][v],all_counts['final/'+best_method][v],seed,2000)
                    for m in ['llm_zero','llm_few']} for v in EVALUATORS}
    save_json(report_dir/'all_metrics.json',summary)
    save_json(report_dir/'selection_audit.json',selection_audit)
    save_json(report_dir/'bootstrap.json',bootstrap)
    write_jsonl(output/'affected_samples.jsonl',affected)

    def table(run_map):
        lines=['| 方法 | 版本 | Token P | Token R | Token F1 | 实体 P | 实体 R | 实体 F1 |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
        for method,run in run_map.items():
            for version,m in summary[run].items():
                values=[100*m[level][k] for level in ['token','entity'] for k in ['precision','recall','f1']]
                lines.append('| '+method+' | '+version+' | '+' | '.join(f'{x:.4f}%' for x in values)+' |')
        return lines
    test_map={m:'final/'+m for m in ['dictionary','hmm','crf','llm_zero','llm_few']}
    dev_map={m:'baselines/'+name for m,name in baseline_selection['selected'].items()}
    dev_map.update(llm_zero='llm/dev_0',llm_few=f"llm/dev_{llm_selection['best_shots']}")
    lines=['# V1 / V2 / V3 离线重评对比','',
           '固定原预测、模型参数、提示、样本及原选参结果，只改变评估协议。此次是查看Test后的协议修订，不能称为测试前预注册。旧runs/final及原报告未覆盖。','',
           '| 版本 | Token评分 | 实体评分 |','|---|---|---|',
           '| V1 | 去重叠后单标签BIO | 保留重叠，包含无效项FP |',
           '| V2 | 统一去重叠后单标签BIO | 同一份去重叠集合 |',
           '| V3（当前） | 保留重叠的字符位置＋完整BIO标签集合 | 全部合法、去重跨度 |','',
           'V2/V3主指标排除无法映射到字符的无效项，但另报带无效项FP的辅助实体指标；不能混淆这项差异与重叠处理。V3不是EDS-NLP原生的类型级指标，而是保留B/I的集合扩展。','',
           '## 固定Test主结果','']+table(test_map)
    lines+=['','## 完整Dev（保持原来选中的模型）','']+table(dev_map)
    lines+=['','## Test变化与原始计数','', '| 方法 | V3−V1 Token F1（百分点） | V2−V1 实体F1（百分点） | V3−V1 实体F1（百分点） | V3重叠句数 | 无效项 |',
            '|---|---:|---:|---:|---:|---:|']
    for method,run in test_map.items():
        a,b,c=[summary[run][v] for v in EVALUATORS]
        lines.append(f"| {method} | {100*(c['token']['f1']-a['token']['f1']):+.6f} | {100*(b['entity']['f1']-a['entity']['f1']):+.6f} | {100*(c['entity']['f1']-a['entity']['f1']):+.6f} | {c['diagnostics'].get('overlap_samples',0)} | {c['diagnostics'].get('invalid_entities',0)} |")
    lines+=['','V3与V1的实体计数在无无效项时应相同；Token差异来自被V1投影丢弃的BIO项重新参与评分。V2则改变实体集合。不得把版本间分数变化解释为模型能力变化。','',
            '## 真实受影响Test案例','']
    for case in [r for r in affected if r['run'].startswith('final/')]:
        lines += [f"### {case['run']} / {case['id']}",'',case['text'],'',
                  '| 版本 | Token TP/FP/FN | 实体 TP/FP/FN |','|---|---|---|']
        for version,counts in case['counts_by_version'].items():
            lines.append(f"| {version} | {counts['token']} | {counts['entity']} |")
    lines+=['','## Dev选参敏感性检查','']
    for version,audit in selection_audit.items():
        lines.append(f"- {version}: {audit['selected']}, few-shot={audit['best_shots']}; 与原冻结选择一致={audit['same_as_frozen']}。")
    lines+=['','此处只检查新口径是否会改变Dev选择，不修改已有冻结文件或模型。','',
            '## 配对Bootstrap','',f'固定Test预测，2000次句子级配对重采样，先累加TP/FP/FN再计算micro F1。seed={seed}；基线为原Dev选出的{best_method}。区间不包含重新训练或API随机性。','',
            '| 版本 | 比较 | 实体F1差（百分点） | 95%百分位区间（百分点） |','|---|---|---:|---|']
    for version,results in bootstrap.items():
        for method,r in results.items():lines.append(f"| {version} | {method} − {best_method} | {100*r['difference_a_minus_b']:.4f} | [{100*r['ci95'][0]:.4f}, {100*r['ci95'][1]:.4f}] |")
    unchanged=all(file_hash(ROOT/p)==h for p,h in hashes.items())
    assert unchanged,'Source files or API ledger changed'
    save_json(output/'manifest.json',{'created_at':utc_now(),'sources':len(sources),'evaluations':len(sources)*3,
        'v1_exact_replay_passed':True,'original_files_unchanged':unchanged,'protected_files':hashes,
        'code_hashes':{p.relative_to(ROOT).as_posix():file_hash(p) for p in [Path(__file__),ROOT/'src/ner_experiment/common.py',
        ROOT/'src/ner_experiment/evaluation/metrics.py',ROOT/'src/ner_experiment/evaluation/metrics_v1.py',ROOT/'src/ner_experiment/evaluation/metrics_v2.py']}})
    lines+=['','## 验证','',f'对{len(sources)}份已有预测进行了{len(sources)*3}次评估，全部V1结果与历史指标、逐句计数完全一致。原预测、原指标、数据、选参、冻结记录和API账本哈希均未变化。新增API调用为0。','',
            '详细指标：all_metrics.json；全部逐句计数和错误：runs/evaluation_versions；受影响句子详情：该目录affected_samples.jsonl。复现命令：python scripts/compare_evaluation_versions.py。']
    (report_dir/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPLETE:',len(sources),'sources;',len(sources)*3,'evaluations; unchanged source hashes.',flush=True)
    for method,run in test_map.items():
        print(method, {v:{k:summary[run][v][k] for k in ['token','entity']} for v in EVALUATORS},flush=True)

if __name__=='__main__':main()
