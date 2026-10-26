"""Preregistered Test diagnostics: paired intervals and Train novelty slices."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from ner_experiment.data.splitting import read_jsonl, json_bytes, jsonl_bytes, sha
from ner_experiment.evaluation.error_analysis import attribute, character_errors
from ner_experiment.evaluation.metrics import evaluate, span_set, spans_to_bio, prf
from ner_experiment.experiments.formal_test import load, verify_gate
from ner_experiment.experiments.index_analysis import f1
from ner_experiment.experiments.llm_dev import unchanged_write
from ner_experiment.experiments.traditional import atomic_write, csv_bytes


def paired_bootstrap(sentence_counts, contrasts, seed, repeats):
    names = list(sentence_counts)
    ids = [r['id'] for r in sentence_counts[names[0]]]
    if not ids or any([r['id'] for r in sentence_counts[n]] != ids for n in names):
        raise ValueError('Paired bootstrap requires identical nonempty ordered IDs')
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(repeats,len(ids)))
    observed, distributions = {}, {}
    for name in names:
        counts = np.asarray([r['entity'] for r in sentence_counts[name]], dtype=np.int64)
        observed[name] = float(f1(counts.sum(axis=0)))
        distributions[name] = f1(counts[draws].sum(axis=1))
    output = []
    for a,b in contrasts:
        delta = distributions[a] - distributions[b]
        output.append({'contrast': a+' - '+b, 'entity_f1_difference': observed[a]-observed[b],
            'ci95_low': float(np.quantile(delta,.025)), 'ci95_high': float(np.quantile(delta,.975)),
            'repeats': repeats, 'seed': seed})
    return output


def novelty_counts(rows, normalized, seen):
    """Partition both sets by (surface,type) so all TP/FP/FN remain additive."""
    result = {name: [0,0,0] for name in ('seen_train_surface_type','unseen_train_surface_type')}
    for row in rows:
        gold = span_set(row['entities'])
        predicted = span_set(normalized[row['id']]['entities'])
        def bucket(span):
            s,e,t = span
            return 'seen_train_surface_type' if (row['text'][s:e],t) in seen else 'unseen_train_surface_type'
        for span in gold & predicted:
            result[bucket(span)][0] += 1
        for span in predicted-gold:
            result[bucket(span)][1] += 1
        for span in gold-predicted:
            result[bucket(span)][2] += 1
    return {name: {**prf(*counts), 'gold_support':counts[0]+counts[2], 'predicted_support':counts[0]+counts[1]}
            for name,counts in result.items()}


def review_templates(directory, errors, methods, seed):
    selected = []
    for method in methods:
        own = [e for e in errors if e['method'] == method]
        key = lambda e: sha(f'{seed}|review|{method}|{e["id"]}|{e["gold"]}|{e["prediction"]}'.encode())
        for rank,e in enumerate(sorted(own,key=key)[:8]):
            selected.append({**e, 'independent_double_review': rank < 4})
    if len(selected) != 40:
        raise ValueError('Expected eight review errors per method')
    selected.sort(key=lambda e: sha(f'{seed}|review-order|{e["method"]}|{e["id"]}|{e["gold"]}|{e["prediction"]}'.encode()))
    tasks, key = [], []
    for i,e in enumerate(selected,1):
        identifier = f'R{i:02d}'
        def describe(span):
            return json.dumps(None if span is None else {'text':e['text'][span[0]:span[1]],
                'start':span[0],'end':span[1],'type':span[2]},ensure_ascii=False)
        tasks.append({'review_id':identifier,'sample_id':e['id'],'text':e['text'],
            'gold_entity':describe(e['gold']),'predicted_entity':describe(e['prediction']),
            'judged_category':'','annotation_issue':'','notes':''})
        key.append({'review_id':identifier, **e})
    second = [task for task,k in zip(tasks,key) if k['independent_double_review']]
    unchanged_write(directory / 'reviewer_A_40.csv', csv_bytes(tasks))
    unchanged_write(directory / 'reviewer_B_independent_20.csv', csv_bytes(second))
    unchanged_write(directory / 'adjudication_40.csv', csv_bytes([{
        'review_id':t['review_id'],'reviewer_A_judgment':'','reviewer_B_judgment':'',
        'final_category':'','annotation_issue':'','resolution_notes':''} for t in tasks]))
    unchanged_write(directory / 'review_key_keep_separate.json', json_bytes(key))
    unchanged_write(directory / 'instructions.txt', (
        'A independently reviews 40 rows; B independently reviews assigned 20 rows before discussion.\n'
        'Keep the key and other reviewer judgments hidden until both sets are saved.\n'
        'Categories: missed, spurious, boundary, type, boundary_and_type, annotation_issue, uncertain.\n'
        'After independent ratings, jointly resolve disagreements in adjudication_40.csv.\n'
        'Templates are unfilled. Automated labels are not human judgments.\n'
        'The 40 error events are sampled eight per method, not a random prevalence estimate.\n').encode())


def main(root):
    root = Path(root)
    gate = verify_gate(root)
    config, directory = gate['config'], root / gate['config']['run_directory']
    summary = load(directory / 'summary.json')
    if not summary['complete'] or summary['gate_sha256'] != sha((root/'configs/test_frozen_v1.json').read_bytes()):
        raise ValueError('Completed frozen Test results required')
    data = root / 'data/processed/experiment_v2'
    for name,digest in gate['input_sha256'].items():
        if sha((data/name).read_bytes()) != digest:
            raise ValueError('Frozen input changed: '+name)
    rows, train, dev = (read_jsonl(data/name) for name in ('test.jsonl','train_valid.jsonl','dev.jsonl'))
    if {r['text'] for r in rows} & ({r['text'] for r in train}|{r['text'] for r in dev}):
        raise ValueError('Test text leakage')
    seen = {(e['text'],e['type']) for r in train for e in r['entities']}
    model_data, table, per_type, novel, aggregates, errors = {}, [], [], [], [], []
    for method in config['methods']:
        folder, stats = directory/method, summary['methods'][method]
        for name,digest in stats['output_sha256'].items():
            if sha((folder/name).read_bytes()) != digest:
                raise ValueError('Saved Test output changed: '+method+'/'+name)
        metrics = evaluate(rows,read_jsonl(folder/'predictions.jsonl'))
        metrics.pop('normalized_predictions')
        if metrics != load(folder/'metrics.json'):
            raise ValueError('Offline metrics differ: '+method)
        normalized = {r['id']:r for r in read_jsonl(folder/'normalized.jsonl')}
        diagnostic = load(folder/'diagnostics.json')
        model_data[method] = metrics
        table.append({'method':method, **{f'{level}_{key}':metrics[level][key]
            for level in ('token','entity') for key in ('precision','recall','f1','tp','fp','fn')},
            'macro_f1_fixed_10':metrics['entity_macro_f1_fixed_10'],
            'character_accuracy_including_O':metrics['character_accuracy_including_O'],
            'surface_type_bag_f1':diagnostic['surface_type_bag']['f1'],
            'raw_span_entity_f1':metrics['sensitivity']['raw_span_entity']['f1'],
            'illegal_bio_rate_before_repair':metrics['rates']['illegal_bio_transition_rate_before_repair'],
            'raw_bio_token_f1_on_bio_samples':metrics['sensitivity']['raw_bio_token_on_bio_samples_only']['f1'],
            'strict_bio_entity_f1_on_bio_samples':metrics['sensitivity']['strict_bio_entity_on_bio_samples_only']['f1'],
            'position_error_sample_rate':diagnostic['position_error_sample_rate'],
            'invalid_entity_rate':metrics['rates']['invalid_entity_rate'], 'failure_rate':metrics['rates']['failure_rate'],
            'duplicate_entities':metrics['diagnostics'].get('duplicate_entities',0),
            'overlap_affected_samples':metrics['diagnostics'].get('overlap_affected_samples',0),
            'median_seconds':stats['latency_median_seconds'],'p95_seconds':stats['latency_p95_seconds'],
            'accounted_cny':stats.get('accounted_cny',0),
            **{k:stats.get('usage',{}).get(k,0) for k in ('prompt_tokens','completion_tokens','cache_hit_tokens','cache_miss_tokens')}})
        for typ, score in metrics['per_type_entity'].items():
            per_type.append({'method':method,'type':typ,**score})
        slices = novelty_counts(rows,normalized,seen)
        assert [sum(s[k] for s in slices.values()) for k in ('tp','fp','fn')] == [metrics['entity'][k] for k in ('tp','fp','fn')]
        for name,score in slices.items():
            novel.append({'method':method,'slice':name,**score})
        kinds,chars = Counter(),Counter()
        per_sentence = {r['id']:r for r in metrics['sentence_counts']}
        for row in rows:
            spans = span_set(normalized[row['id']]['entities'])
            chars.update(character_errors(row['bio_tags'],spans_to_bio(spans,len(row['text']))))
            for error in attribute(span_set(row['entities']),spans):
                kinds[error['category']] += 1
                errors.append({'method':method,'id':row['id'],'text':row['text'],**error,
                    'sentence_token_counts':per_sentence[row['id']]['token'],
                    'sentence_entity_counts':per_sentence[row['id']]['entity']})
        paired = sum(kinds[k] for k in ('boundary','type','boundary_and_type'))
        assert paired+kinds['missed'] == metrics['entity']['fn']
        assert paired+kinds['spurious'] == metrics['entity']['fp']
        assert sum(chars[k] for k in ('token_missed','token_bi_confusion','token_type_confusion')) == metrics['token']['fn']
        assert sum(chars[k] for k in ('token_spurious','token_bi_confusion','token_type_confusion')) == metrics['token']['fp']
        aggregates.append({'method':method,**{k:kinds[k] for k in ('missed','spurious','boundary','type','boundary_and_type')},
            **{k:chars[k] for k in ('token_missed','token_spurious','token_bi_confusion','token_type_confusion')}})
    counts = {n:model_data[n]['sentence_counts'] for n in ('crf','llm_zero','llm_4_shot')}
    contrasts = paired_bootstrap(counts,config['planned_entity_f1_contrasts'],config['bootstrap_seed'],config['bootstrap_repeats'])
    analysis = directory/'analysis'
    unchanged_write(analysis/'paired-bootstrap.json',json_bytes(contrasts))
    unchanged_write(analysis/'errors.jsonl',jsonl_bytes(errors))
    review_templates(directory/'human-review',errors,config['methods'],config['human_review']['seed'])
    cases,used = [],set()
    for method in ('llm_zero','llm_4_shot','crf'):
        for kind in ('boundary','type'):
            own = [e for e in errors if e['method']==method and e['category']==kind and e['id'] not in used]
            if own:
                case=min(own,key=lambda e:(len(e['text']),e['id'],e['gold'],e['prediction']))
                cases.append(case);used.add(case['id'])
    unchanged_write(analysis/'illustrative-cases.json',json_bytes(cases))
    for filename,content in [('formal-test-results.csv',table),('formal-test-per-type.csv',per_type),
            ('formal-test-novelty.csv',novel),('formal-test-errors.csv',aggregates),('formal-test-bootstrap.csv',contrasts)]:
        atomic_write(root/'reports'/filename,csv_bytes(content))
    lines=['# 本分支正式 Test：五方法、定位诊断与 Train 词面覆盖', '',
        '使用本分支固定 Test 1,209 句；有效 Train 9,538 句；官方 Dev 1,343 句。所有配置在 Test 预测前冻结，未根据 Test 分数修改。',
        '这是当前语料、监督及资源条件下的系统比较；零样本仅指未提供任务示例，无法排除预训练数据接触。', '',
        '| 方法 | Token P (%) | Token R (%) | Token F1 (%) | 实体 P (%) | 实体 R (%) | 实体 F1 (%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append('| '+' | '.join([row['method'],*[f'{row[level+"_"+key]*100:.2f}' for level in ('token','entity') for key in ('precision','recall','f1')]])+' |')
    lines += ['', '## 预先声明的配对 Bootstrap', '',
        '以句子为单位有放回抽样，两方法共享每次抽样索引，汇总实体 TP/FP/FN 后重算全局 micro F1。2,000 次，种子 141595842。',
        '下列为描述性百分位 95% 区间，未作多重比较校正；只反映当前固定预测下的测试样本不确定性，不包含重新训练或重新生成的波动。', '',
        '| 差值方向 | 实体 F1 差值 (百分点) | 95% 区间 (百分点) |', '|---|---:|---|']
    for c in contrasts:
        lines.append(f'| {c["contrast"]} | {c["entity_f1_difference"]*100:+.2f} | [{c["ci95_low"]*100:+.2f}, {c["ci95_high"]*100:+.2f}] |')
    lines += ['', '## 独立诊断一：识别与位置填写', '',
        '两套主指标使用同一合法、去重、非重叠预测。原始跨度 F1 与忽略坐标的 text+type 多重集 F1 仅作为诊断，后者无法区分重复名称的具体出现位置，不是主指标上界。', '',
        '| 方法 | 坐标错误句 (%) | text+type F1 (%) | 原始跨度 F1 (%) | 无效实体 (%) |', '|---|---:|---:|---:|---:|']
    for row in table:
        lines.append('| '+' | '.join([row['method'],*[f'{row[k]*100:.2f}' for k in ('position_error_sample_rate','surface_type_bag_f1','raw_span_entity_f1','invalid_entity_rate')]])+' |')
    lines += ['', '## 独立诊断二：训练中见过与未见过的实体词面及类别', '',
        '按实体原文及类别二元组是否在有效 Train 出现分组，金标准和规范化预测按同一规则分组，各组 TP/FP/FN 可加回主表。不是按整句是否见过分组。',
        '这属于本项目 Train 覆盖诊断；对商业 LLM 不代表预训练未知实体。各组长度、类别和难度可能不同，不能将差异直接归因为记忆或推理能力。', '',
        '| 方法 | 分组 | Gold 数 | 实体 P (%) | 实体 R (%) | 实体 F1 (%) |', '|---|---|---:|---:|---:|---:|']
    for row in novel:
        lines.append('| '+' | '.join([row['method'],row['slice'],str(row['gold_support']),*[f'{row[k]*100:.2f}' for k in ('precision','recall','f1')]])+' |')
    lines += ['', '## 一对一归因与六个真实案例', '',
        '去掉严格匹配后，按边界相同、类别相同、较大 IoU、跨度字典序贪心配对相交实体，每条实体最多配对一次；剩余为漏抽或误抽。配对错误各消耗一个 FP 和 FN，已核对守恒。',
        '案例由固定规则自动选择，不能替代两人独立人工核验；局部错误对不涵盖整句全部错误。', '']
    def describe(span,text):
        return '无' if span is None else f'`{text[span[0]:span[1]]}` [{span[0]},{span[1]}) {span[2]}'
    for i,c in enumerate(cases,1):
        lines += [f'{i}. {c["method"]} / {c["id"]} / {c["category"]}', '',c['text'],'',
            f'金标准：{describe(c["gold"],c["text"])}；预测：{describe(c["prediction"],c["text"])}。','',
            f'整句 Token TP/FP/FN：{c["sentence_token_counts"]}；实体 TP/FP/FN：{c["sentence_entity_counts"]}。','']
    lines += ['## 费用、延迟与人工任务', '',
        f'本轮 API 记账 {summary["stage_accounted_cny"]:.8f} 元；首次结果导出时全项目累计 {summary["accounting_at_first_export"]["accounted_cny"]:.8f} 元。实际 usage 和价格快照保存在本地。',
        'LLM 时间包含网络和服务开销、重试，不含退避；传统方法为本机特征及预测。两者不是同条件纯算法速度比较。', '',
        '| 方法 | 中位数 (s) | P95 (s) | 失败率 (%) |', '|---|---:|---:|---:|']
    for row in table:
        lines.append(f'| {row["method"]} | {row["median_seconds"]:.6f} | {row["p95_seconds"]:.6f} | {row["failure_rate"]*100:.2f} |')
    lines += ['', '本地 human-review 目录已生成每方法 8 个错误事件，共 40 个；其中每方法 4 个、合计 20 个交给第二位核验人独立判断。隐藏方法名和自动分类，独立完成后共同复核。模板为空，人工任务尚未完成，不伪填判断。',
        '本轮不再用 Test 改提示、示例、模型或主协议；后续改进需另立实验，保留本轮冻结结果。', '']
    atomic_write(root/'reports/formal-test-results.md','\n'.join(lines).encode())
    verification={'saved_predictions_rescored_identically':True,'novelty_counts_additive':True,
        'error_counts_conserved':True,'bootstrap_repeats':config['bootstrap_repeats'],
        'bootstrap_seed':config['bootstrap_seed'],'review_errors_prepared':40,'independent_review_tasks':20,
        'human_ratings_completed':False,'illustrative_cases':len(cases)}
    unchanged_write(analysis/'verification.json',json_bytes(verification))
    print(json.dumps({'table':table,'contrasts':contrasts,'novelty':novel,'verification':verification},indent=2))
    return verification
