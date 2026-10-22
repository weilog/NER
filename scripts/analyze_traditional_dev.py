"""Deterministic error attribution and examples from actual Dev predictions.

This is an automated audit, not the required independent human review.
"""
from collections import Counter
from pathlib import Path
import json
import joblib

from ner_experiment.data.splitting import read_jsonl, json_bytes, jsonl_bytes
from ner_experiment.evaluation.metrics import span_set, spans_to_bio
from ner_experiment.experiments.traditional import atomic_write, csv_bytes


def attribute(gold, predicted):
    """Remove exact matches, greedily pair overlapping remainders once.

    Candidate priority: equal boundaries, equal type, larger IoU, then sorted
    gold/predicted (start,end,type). Non-overlapping remainders remain FN/FP.
    """
    gold_left, pred_left = set(gold - predicted), set(predicted - gold)
    candidates = []
    for g in sorted(gold_left):
        for p in sorted(pred_left):
            intersection = min(g[1], p[1]) - max(g[0], p[0])
            if intersection > 0:
                iou = intersection / (max(g[1], p[1]) - min(g[0], p[0]))
                candidates.append((-(g[:2] == p[:2]), -(g[2] == p[2]), -iou, g, p))
    errors = []
    for _, _, _, g, p in sorted(candidates):
        if g not in gold_left or p not in pred_left:
            continue
        kind = 'type' if g[:2] == p[:2] else 'boundary' if g[2] == p[2] else 'boundary_and_type'
        errors.append({'category': kind, 'gold': g, 'prediction': p})
        gold_left.remove(g)
        pred_left.remove(p)
    errors += [{'category': 'missed', 'gold': g, 'prediction': None} for g in sorted(gold_left)]
    errors += [{'category': 'spurious', 'gold': None, 'prediction': p} for p in sorted(pred_left)]
    return errors


def character_errors(gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError('Character sequence length mismatch')
    result = Counter()
    for g, p in zip(gold, predicted):
        if g == p:
            continue
        if p == 'O':
            result['token_missed'] += 1
        elif g == 'O':
            result['token_spurious'] += 1
        elif g[2:] == p[2:]:
            result['token_bi_confusion'] += 1
        else:
            result['token_type_confusion'] += 1
    return result


def main(root):
    root = Path(root)
    summary = json.loads((root / 'runs/traditional_dev/summary.json').read_text(encoding='utf-8'))
    run = root / summary['run_directory']
    gold = read_jsonl(root / 'data/processed/experiment_v2/dev.jsonl')
    errors, aggregates = [], []
    for family, trial in summary['selected'].items():
        normalized = {row['id']: row for row in read_jsonl(run / trial['name'] / 'normalized.jsonl')}
        metrics = json.loads((run / trial['name'] / 'metrics.json').read_text(encoding='utf-8'))
        per_sentence = {row['id']: row for row in metrics['sentence_counts']}
        kinds = Counter()
        char_errors = Counter()
        paired, missed, spurious = 0, 0, 0
        for row in gold:
            char_errors.update(character_errors(row['bio_tags'], spans_to_bio(
                span_set(normalized[row['id']]['entities']), len(row['text']))))
            for error in attribute(span_set(row['entities']), span_set(normalized[row['id']]['entities'])):
                kinds[error['category']] += 1
                paired += int(error['gold'] is not None and error['prediction'] is not None)
                missed += int(error['category'] == 'missed')
                spurious += int(error['category'] == 'spurious')
                errors.append({'family': family, 'configuration': trial['name'], 'id': row['id'],
                               'text': row['text'], **error,
                               'sentence_token_counts': per_sentence[row['id']]['token'],
                               'sentence_entity_counts': per_sentence[row['id']]['entity']})
        # Each paired error consumes exactly one strict FP and one strict FN.
        assert paired + missed == trial['entity']['fn']
        assert paired + spurious == trial['entity']['fp']
        assert char_errors['token_missed'] + char_errors['token_bi_confusion'] + char_errors['token_type_confusion'] == trial['token']['fn']
        assert char_errors['token_spurious'] + char_errors['token_bi_confusion'] + char_errors['token_type_confusion'] == trial['token']['fp']
        aggregates.append({'family': family, 'configuration': trial['name'],
                           **{kind: kinds[kind] for kind in ('missed', 'spurious', 'boundary', 'type', 'boundary_and_type')},
                           'strict_fp': trial['entity']['fp'], 'strict_fn': trial['entity']['fn'],
                           **{key: char_errors[key] for key in ('token_missed', 'token_spurious', 'token_bi_confusion', 'token_type_confusion')}})
    output = run / 'analysis'
    atomic_write(output / 'errors.jsonl', jsonl_bytes(errors))
    atomic_write(root / 'reports/traditional-dev-errors.csv', csv_bytes(aggregates))

    # Deterministic, diverse illustrative cases; never claim these were reviewed.
    cases, used = [], set()
    for family in ('dictionary', 'hmm', 'crf'):
        for kind in ('boundary', 'type'):
            choices = [e for e in errors if e['family'] == family and e['category'] == kind and e['id'] not in used]
            if choices:
                case = min(choices, key=lambda e: (len(e['text']), e['id'], e['gold'], e['prediction']))
                cases.append(case)
                used.add(case['id'])
    for error in errors:
        if len(cases) >= 6:
            break
        if error['id'] not in used:
            cases.append(error)
            used.add(error['id'])
    atomic_write(output / 'illustrative-cases.json', json_bytes(cases))
    report = ['# Dev 原理实验与自动错误审计', '',
              '所有数值和案例来自实际 Dev 预测；不是项目 Test 结果，也未完成两人独立人工复核。', '',
              '## HMM 词表阈值', '',
              '下表固定 Dev 选出的 alpha，只改变最低字符频次。OOV 为有效 Train 词表之外的 Dev 字符比例。', '',
              '| min count | 词表字符数 | Dev UNK (%) | Token F1 (%) | 实体 F1 (%) |',
              '|---|---:|---:|---:|---:|']
    selected_hmm = summary['selected']['hmm']
    alpha = selected_hmm['parameters']['alpha']
    for trial in summary['trials']:
        if trial['family'] == 'hmm' and trial['parameters']['constrained'] and trial['parameters']['alpha'] == alpha:
            v = trial['vocabulary']
            report.append(f'| {trial["parameters"]["min_char_count"]} | {v["vocabulary_characters"]} | {v["dev_unk_rate"]*100:.3f} | {trial["token"]["f1"]*100:.3f} | {trial["entity"]["f1"]*100:.3f} |')
    report += ['', '## HMM 序列约束', '',
               '每对比较使用同一组监督估计参数；下表非法率在修复前计算，两级 F1 均按统一规范化后的主协议计算。', '',
               '| alpha | min count | 普通非法率 (%) | 约束非法率 (%) | 普通 Token F1 (%) | 约束 Token F1 (%) | 普通实体 F1 (%) | 约束实体 F1 (%) |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    for constrained in summary['trials']:
        if constrained['family'] != 'hmm' or not constrained['parameters']['constrained']:
            continue
        plain = next(t for t in summary['trials'] if t['name'] == constrained['name'].replace('_bio', '_plain'))
        values = [plain['rates']['illegal_bio_transition_rate_before_repair'] * 100,
                  constrained['rates']['illegal_bio_transition_rate_before_repair'] * 100,
                  plain['token']['f1'] * 100, constrained['token']['f1'] * 100,
                  plain['entity']['f1'] * 100, constrained['entity']['f1'] * 100]
        report.append('| ' + ' | '.join([str(constrained['parameters']['alpha']), str(constrained['parameters']['min_char_count']), *[f'{value:.4f}' for value in values]]) + ' |')
    report += ['', '若普通解码已经合法，则约束未必改变结果；不能将没有观察到的增益写成实验结论。', '',
               '## 两级指标与选参差异', '',
               'Dev 选出的 HMM Token F1 高于词典，但实体 F1 低于词典，说明字符标签的部分正确不保证完整实体边界正确。',
               'CRF 的 (c1,c2)=(0.1,1.0) 配置 Token F1 略高于 (0.1,0.1)，但实体 F1 更低；按预先固定的实体标准仍选择 (0.1,0.1)。', '',
               '## 一对一错误归因', '',
               '先移除严格匹配。候选仅限相交跨度，依次优先相同边界、相同类别、较大 IoU，再按金标准及预测的起止位置与类别排序；贪心配对且每条实体最多使用一次。',
               '边界与类别同时错误单列；未配对金标准为漏抽，未配对预测为误抽。配对错误各贡献一个严格 FP 和 FN，程序核对总数守恒。', '',
               '| 方法 | 漏抽 | 误抽 | 边界 | 类别 | 边界及类别 |', '|---|---:|---:|---:|---:|---:|']
    for item in aggregates:
        report.append('| ' + ' | '.join([item['family'], *[str(item[k]) for k in ('missed', 'spurious', 'boundary', 'type', 'boundary_and_type')]]) + ' |')
    report += ['', '字符级错误按规范化后 BIO 另外统计：', '',
               '| 方法 | 漏标字符 | 多标字符 | B/I 混淆字符 | 类别混淆字符 |',
               '|---|---:|---:|---:|---:|']
    for item in aggregates:
        report.append('| ' + ' | '.join([item['family'], *[str(item[k]) for k in ('token_missed', 'token_spurious', 'token_bi_confusion', 'token_type_confusion')]]) + ' |')
    report += ['', '## 六个真实预测案例', '',
               '下列展示的是单个错误对；Token/实体计数是整句的 TP/FP/FN，不能用一个局部错误解释整句全部错误。跨度为原文 Unicode 字符 [start,end)。', '']
    def describe(span, text):
        if span is None:
            return '无'
        start, end, typ = span
        return f'`{text[start:end]}` [{start},{end}) {typ}'
    for number, case in enumerate(cases, 1):
        report += [f'### {number}. {case["family"]} / {case["id"]} / {case["category"]}', '',
                   case['text'], '',
                   f'金标准：{describe(case["gold"], case["text"])}；预测：{describe(case["prediction"], case["text"])}。', '',
                   f'整句 Token TP/FP/FN：`{case["sentence_token_counts"]}`；实体 TP/FP/FN：`{case["sentence_entity_counts"]}`。', '']
    training_audit = []
    for trial in summary['trials']:
        if trial['family'] == 'crf':
            model = joblib.load(root / trial['model_path'])
            iterations = model.model_.training_log_.iterations
            last = iterations[-1] if iterations else {}
            training_audit.append({'configuration': trial['name'],
                                   'iterations_completed': last.get('num', 0),
                                   'configured_limit': trial['parameters']['max_iterations'],
                                   'reached_iteration_limit': last.get('num', 0) >= trial['parameters']['max_iterations']})
    atomic_write(output / 'crf-training-audit.json', json_bytes(training_audit))
    report += ['这些案例由固定规则自动选择，仅供解释和后续核验，不代表错误分布的随机样本。',
               '金标准在这里指数据集的原始标注，未经过人工确认；例如姓名状文本被标为 position 的情况应留给标注质量复核，不自动改标签。', '',
               '## CRF 迭代上限', '',
               '原生训练日志的实际迭代数如下；达到 100 次上限不等于优化已经收敛，本阶段遵守预定上限，不按 Dev 分数临时追加迭代。', '']
    for audit in training_audit:
        report.append(f'- `{audit["configuration"]}`：完成 {audit["iterations_completed"]} 次；达到配置上限：{audit["reached_iteration_limit"]}。')
    report.append('')
    atomic_write(root / 'reports/traditional-dev-analysis.md', '\n'.join(report).encode('utf-8'))
    print('Saved automated Dev error audit and six actual prediction examples.')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
