"""Validate saved Dev predictions, export five-method results and selection.

No network calls, no Test access, and no human-review claims.
"""
from collections import Counter
import json
from pathlib import Path

from analyze_traditional_dev import attribute, character_errors
from ner_experiment.data.splitting import read_jsonl, json_bytes, sha
from ner_experiment.evaluation.metrics import evaluate, span_set, spans_to_bio
from ner_experiment.experiments.llm_dev import choose_shots, prepare, unchanged_write
from ner_experiment.experiments.traditional import atomic_write, csv_bytes
from ner_experiment.llm.budget import Ledger


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validated(root, directory, digests, gold):
    for name, digest in digests.items():
        if sha((directory / name).read_bytes()) != digest:
            raise ValueError('Saved output hash changed: ' + str(directory / name))
    metrics = load(directory / 'metrics.json')
    recomputed = evaluate(gold, read_jsonl(directory / 'predictions.jsonl'))
    recomputed.pop('normalized_predictions')
    if recomputed != metrics:
        raise ValueError('Saved prediction rescoring differs: ' + str(directory))
    return metrics, {r['id']: r for r in read_jsonl(directory / 'normalized.jsonl')}


def flat(method, metrics, **extra):
    return {'method': method, **{f'{level}_{key}': metrics[level][key]
        for level in ('token', 'entity') for key in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn')},
        'entity_macro_f1_fixed_10': metrics['entity_macro_f1_fixed_10'],
        'character_accuracy_including_O': metrics['character_accuracy_including_O'], **extra}


def main(root):
    root = Path(root)
    config, pricing, gold, requests, context, _ = prepare(root, 'full')
    directory = root / config['run_directory']
    tune = load(directory / 'tune/summary.json')
    full = load(directory / 'full/summary.json')
    if not tune['complete'] or not full['complete'] or full['context_sha256'] != sha(json_bytes(context)):
        raise ValueError('Completed identical Dev context required')
    selection = load(directory / 'selection.json')
    selected = choose_shots(tune['arms'])
    if selected != selection['selected_shots']:
        raise ValueError('Shot selection changed')
    tune_gold = read_jsonl(root / 'data/processed/experiment_v2/dev_tune.jsonl')
    tuning_rows = []
    for shots in config['shots']:
        name = f'shot_{shots}'
        arm = tune['arms'][name]
        metrics, _ = validated(root, directory / 'tune' / name, arm['output_sha256'], tune_gold)
        tuning_rows.append(flat(name, metrics, selected_few_shot=(shots == selected),
            median_network_seconds=arm['latency_median_seconds'], p95_network_seconds=arm['latency_p95_seconds'],
            position_error_sample_rate=arm['position_error_sample_rate'], **arm['usage']))
    rows, per_type, model_data = [], [], {}
    traditional = load(root / 'configs/traditional_selected.json')
    traditional_context = load(root / traditional['run_directory'] / 'context.json')
    if traditional_context['dev_sha256'] != context['input_sha256']['dev.jsonl']:
        raise ValueError('Traditional and LLM Dev datasets differ')
    for method, chosen in traditional['selected'].items():
        folder = root / traditional['run_directory'] / chosen['name']
        trial = load(folder / 'trial.json')
        if trial['context_sha256'] != traditional['context_sha256']:
            raise ValueError('Traditional context changed')
        if sha((root / chosen['model_path']).read_bytes()) != chosen['model_sha256']:
            raise ValueError('Traditional model hash changed')
        metrics, normalized = validated(root, folder, trial['output_sha256'], gold)
        rows.append(flat(method, metrics, configuration=chosen['name'],
            median_network_seconds='', p95_network_seconds='',
            median_local_prediction_ms=trial['latency_median_ms'], p95_local_prediction_ms=trial['latency_p95_ms']))
        model_data[method] = (metrics, normalized)
    for shots in (0, selected):
        name, method = f'shot_{shots}', ('llm_zero' if shots == 0 else f'llm_{shots}_shot')
        arm = full['arms'][name]
        metrics, normalized = validated(root, directory / 'full' / name, arm['output_sha256'], gold)
        rows.append(flat(method, metrics, configuration=name,
            median_network_seconds=arm['latency_median_seconds'], p95_network_seconds=arm['latency_p95_seconds'],
            median_local_prediction_ms='', p95_local_prediction_ms='',
            position_error_sample_rate=arm['position_error_sample_rate'], failure_rate=arm['rates']['failure_rate'],
            unique_prediction_cost_cny=arm['unique_prediction_cost_cny_including_reused_requests'], **arm['usage']))
        model_data[method] = (metrics, normalized)
    errors, aggregates = [], []
    for method, (metrics, normalized) in model_data.items():
        counts, characters = Counter(), Counter()
        per_sentence = {r['id']: r for r in metrics['sentence_counts']}
        for row in gold:
            predicted = span_set(normalized[row['id']]['entities'])
            characters.update(character_errors(row['bio_tags'], spans_to_bio(predicted, len(row['text']))))
            for error in attribute(span_set(row['entities']), predicted):
                counts[error['category']] += 1
                errors.append({'method': method, 'id': row['id'], 'text': row['text'], **error,
                    'sentence_token_counts': per_sentence[row['id']]['token'],
                    'sentence_entity_counts': per_sentence[row['id']]['entity']})
        paired = sum(counts[k] for k in ('boundary', 'type', 'boundary_and_type'))
        assert paired + counts['missed'] == metrics['entity']['fn']
        assert paired + counts['spurious'] == metrics['entity']['fp']
        assert sum(characters[k] for k in ('token_missed', 'token_bi_confusion', 'token_type_confusion')) == metrics['token']['fn']
        assert sum(characters[k] for k in ('token_spurious', 'token_bi_confusion', 'token_type_confusion')) == metrics['token']['fp']
        aggregates.append({'method': method, **{k: counts[k] for k in ('missed','spurious','boundary','type','boundary_and_type')},
            **{k: characters[k] for k in ('token_missed','token_spurious','token_bi_confusion','token_type_confusion')}})
        for typ, score in metrics['per_type_entity'].items():
            per_type.append({'method': method, 'type': typ, **score})
    # Selection of examples is deterministic and descriptive, not a random audit.
    cases, used = [], set()
    for method in ('llm_zero', f'llm_{selected}_shot', 'crf'):
        for kind in ('boundary', 'type'):
            candidates = [e for e in errors if e['method'] == method and e['category'] == kind and e['id'] not in used]
            if candidates:
                case = min(candidates, key=lambda e: (len(e['text']), e['id'], e['gold'], e['prediction']))
                cases.append(case); used.add(case['id'])
    if len(cases) != 6:
        raise ValueError('Expected six distinct illustrative boundary/type cases')
    analysis = directory / 'analysis'
    atomic_write(analysis / 'errors.jsonl', b''.join(json.dumps(e, ensure_ascii=False).encode()+b'\n' for e in errors))
    atomic_write(analysis / 'illustrative-cases.json', json_bytes(cases))
    atomic_write(root / 'reports/llm-dev-tuning.csv', csv_bytes(tuning_rows))
    # CSV rows have different optional cost/timing fields; give them one schema.
    all_fields = list(dict.fromkeys(key for row in rows for key in row))
    atomic_write(root / 'reports/five-method-dev.csv', csv_bytes([{key: row.get(key, '') for key in all_fields} for row in rows]))
    atomic_write(root / 'reports/five-method-dev-per-type.csv', csv_bytes(per_type))
    atomic_write(root / 'reports/five-method-dev-errors.csv', csv_bytes(aggregates))
    freeze = {'status': 'dev_selection_complete; test_not_evaluated',
        'prompt_version': context['prompt_version'], 'parser_version': context['parser_version'],
        'selected_few_shot_count': selected, 'selection_rule': config['selection'],
        'example_ids': context['example_ids'][:selected],
        'parameters': {k: config[k] for k in ('model','temperature','thinking','response_format','max_tokens','timeout_seconds','max_retries','concurrency')},
        'full_dev_context_sha256': full['context_sha256'],
        'full_dev_summary_sha256': sha((directory / 'full/summary.json').read_bytes()),
        'tune_context_sha256': selection['tune_context_sha256'], 'tune_summary_sha256': selection['tune_summary_sha256'],
        'source_sha256': context['source_sha256'], 'input_sha256': context['input_sha256'],
        'response_models': sorted({m for a in full['arms'].values() for m in a['response_models']}),
        'system_fingerprints': sorted({m for a in full['arms'].values() for m in a['system_fingerprints']}),
        'test_accessed': False}
    unchanged_write(root / 'configs/llm_selected.json', json_bytes(freeze))
    ledger = Ledger(root / config['ledger_path'], config['budget_cny'])
    try:
        accounting = ledger.summary()
        stage_cost = ledger.spent('llm-dev-v1/tune') + ledger.spent('llm-dev-v1/full')
    finally:
        ledger.close()
    lines = ['# 五种方法：完整 Dev 对比', '',
        '所有方法使用本分支同一份官方 Dev（1,343 句）及 dual-evaluation-v2；这是开发集成绩，不是 Test 泛化成绩。',
        '传统模型用完整有效 Train（9,538 句）训练；LLM 权重未作任务微调，few-shot 仅提供固定 Train 示例前缀。', '',
        '| 方法 | Token P (%) | Token R (%) | Token F1 (%) | 实体 P (%) | 实体 R (%) | 实体 F1 (%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + ' | '.join([row['method'], *[f'{row[level+"_"+key]*100:.2f}' for level in ('token','entity') for key in ('precision','recall','f1')]]) + ' |')
    lines += ['', '## 固定 300 条 Dev 的示例数实验', '',
        '提示保持试跑 v2 原样，不按后续索引实验挑选最高分组；2/4/8 条示例为同一列表的嵌套前缀。',
        '按实体 F1 选少样本配置；同分按平均原始网络耗时，再按 2/4/8 顺序。零样本独立保留。', '',
        '| 示例数 | Token F1 (%) | 实体 F1 (%) | 位置错误句 (%) | 输入 tokens |',
        '|---|---:|---:|---:|---:|']
    for row in tuning_rows:
        lines.append(f'| {row["method"]} | {row["token_f1"]*100:.2f} | {row["entity_f1"]*100:.2f} | {row["position_error_sample_rate"]*100:.2f} | {row["prompt_tokens"]} |')
    lines += ['', f'选定 {selected}-shot。完整 Dev 包含用于选择的 300 句，完整 Dev 检查不会触发重新选择或改提示；最终独立检验留给冻结后的项目 Test。', '',
        '## 成本与延迟', '',
        f'本轮新增调参与完整 Dev 调用合计记账 {stage_cost:.8f} 元；全项目累计 {accounting["accounted_cny"]:.8f} 元，未决记账 {accounting["unknown_or_pending_attempts"]} 次。',
        '按各次请求实际 usage 和当日价格快照记账；阶段成本只计新增调用。调参及完整 Dev 重叠请求使用缓存，两个表的单方案成本不能相加当作新增总成本。',
        'LLM 时间为原始网络请求耗时，含重试、不含退避；本地模型时间含特征与解码。缓存保留原始时间，两类测量不是同一硬件条件下的算法速度比较。', '',
        '| LLM | 失败率 (%) | 位置错误句 (%) | 中位数 (s) | P95 (s) |',
        '|---|---:|---:|---:|---:|']
    for name, arm in full['arms'].items():
        lines.append(f'| {name} | {arm["rates"]["failure_rate"]*100:.2f} | {arm["position_error_sample_rate"]*100:.2f} | {arm["latency_median_seconds"]:.3f} | {arm["latency_p95_seconds"]:.3f} |')
    lines += ['', '## 错误归因及六个真实案例', '',
        '先去掉严格正确实体；相交剩余跨度按相同边界、相同类别、IoU、跨度字典序贪心配对，每个实体只用一次。配对错误各贡献一个 FP/FN，漏抽和误抽为未配对实体；已核对计数守恒。',
        '案例按固定规则选择，尚未经过要求的两人独立人工核验；不代表随机错误样本。以下局部实体对不一定涵盖整句其他错误。', '']
    def describe(span, text):
        if span is None:
            return '无'
        start, end, typ = span
        return f'`{text[start:end]}` [{start},{end}) {typ}'
    for i, case in enumerate(cases, 1):
        lines += [f'{i}. {case["method"]} / {case["id"]} / {case["category"]}', '',
            case['text'], '', f'金标准：{describe(case["gold"], case["text"])}；预测：{describe(case["prediction"], case["text"])}。', '',
            f'整句 Token TP/FP/FN：{case["sentence_token_counts"]}；实体 TP/FP/FN：{case["sentence_entity_counts"]}。', '']
    lines += ['本报告比较当前数据、标注规范和资源条件下的系统；不能据此声称某类架构普遍优于另一类。公开语料可能进入大模型预训练，零样本仅指请求未提供训练示例。', '',
        '已对五种方法与四组调参结果保存的预测逐份离线重算；正式 Test、Test 配对 bootstrap、40 个错误的两人核验仍未完成。', '']
    atomic_write(root / 'reports/five-method-dev.md', '\n'.join(lines).encode('utf-8'))
    verification = {'all_saved_predictions_rescored_identically': True, 'methods': 5,
        'tune_arms': 4, 'error_count_conservation': True, 'illustrative_cases': len(cases),
        'stage_accounted_cny': stage_cost, 'accounting': accounting, 'test_accessed': False}
    atomic_write(analysis / 'verification.json', json_bytes(verification))
    print(json.dumps({'selected_shots': selected, 'rows': rows, 'verification': verification}, indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
