"""Re-score 15 saved Test predictions with all three historical evaluators; offline only."""
import csv
import json
from pathlib import Path
from ner_experiment.common import ROOT, file_hash, load_json, project_data, read_jsonl, save_json, utc_now, write_jsonl
from ner_experiment.evaluation.metrics_v1 import evaluate as v1
from ner_experiment.evaluation.metrics_v2 import evaluate as v2
from ner_experiment.evaluation.metrics import evaluate as v3
from integrate_results import specifications, ordered_predictions

EVALUATORS = {'v1': v1, 'v2': v2, 'v3': v3}
REPORT = ROOT / 'reports/evaluation_versions/current_15'
LOCAL = ROOT / 'runs/evaluation_versions/current_15'


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_groups(specs, results, version, level, family=None):
    names = [s['method'] for s in specs if family is None or s['family'] == family]
    # Preserve ties as groups rather than manufacturing rank changes with rounding.
    groups = []
    for name in sorted(names, key=lambda n: (-results[n][version][level]['f1'], n)):
        score = results[name][version][level]['f1']
        if groups and score == groups[-1]['f1']:
            groups[-1]['methods'].append(name)
        else:
            groups.append({'f1': score, 'methods': [name]})
    return groups


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    specs = specifications()
    gold = project_data('test')
    assert len(specs) == 15 and len(gold) == 1209
    protected = set()
    for spec in specs:
        protected.add(spec['folder'] / 'predictions.jsonl')
        protected.add(spec['reference'])
        for filename in ['metrics.json', 'sentence_counts.jsonl', 'frozen_protocol.json', 'execution_protocol_v2.json']:
            path = spec['folder'] / filename
            if path.exists(): protected.add(path)
        base = spec.get('base', spec['folder'])
        for filename in ['ledger.json', 'frozen_protocol.json', 'execution_protocol_v2.json', 'summary.json']:
            path = base / filename
            if path.exists(): protected.add(path)
    protected.update((ROOT / 'data/processed/cluener2020_project').glob('*.jsonl'))
    protected.update((ROOT / 'configs').glob('*.json'))
    protected.update((ROOT / 'reports/tables/integrated_v3').glob('*.csv'))
    hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(protected)}
    results, affected, rows, differences = {}, [], [], []
    for spec in specs:
        name = spec['method']
        predictions = ordered_predictions(gold, read_jsonl(spec['folder'] / 'predictions.jsonl'))
        by_version, counts_by_version = {}, {}
        for version, evaluate in EVALUATORS.items():
            metrics, counts, cases = evaluate(gold, predictions)
            by_version[version], counts_by_version[version] = metrics, counts
            dest = LOCAL / name / version
            save_json(dest / 'metrics.json', metrics)
            write_jsonl(dest / 'sentence_counts.jsonl', counts)
            write_jsonl(dest / 'errors.jsonl', cases)
            row = {'method': name, 'display': spec['display'], 'version': version, 'samples': len(gold)}
            for level in ['token', 'entity']:
                for key in ['tp', 'fp', 'fn', 'precision', 'recall', 'f1']:
                    row[f'{level}_{key}'] = metrics[level][key]
            row.update(failure_rate=metrics['failure_rate'],
                       invalid_entities=metrics['diagnostics'].get('invalid_entities', 0),
                       overlap_samples=by_version[version]['diagnostics'].get('overlap_samples', metrics['diagnostics'].get('overlap_projection_samples', 0)))
            rows.append(row)
        assert by_version['v3'] == load_json(spec['reference']), f'V3 saved replay mismatch: {name}'
        assert by_version['v1']['token'] == by_version['v2']['token'], f'V1/V2 token mismatch: {name}'
        assert by_version['v1']['entity'] == by_version['v3']['entity_with_invalid_fp'], f'V1/penalized-V3 mismatch: {name}'
        changed = 0
        for i, row in enumerate(gold):
            cs = {v: {level: counts_by_version[v][i][level] for level in ['token', 'entity']} for v in EVALUATORS}
            if cs['v1'] != cs['v2'] or cs['v1'] != cs['v3']:
                changed += 1
                affected.append({'method': name, 'id': row['id'], 'text': row['text'], 'gold': row['entities'],
                                 'prediction': predictions[i], 'counts_by_version': cs})
        delta = {'method': name, 'display': spec['display'], 'affected_sentences': changed,
                 'overlap_samples': by_version['v3']['diagnostics'].get('overlap_samples', 0),
                 'invalid_entities': by_version['v3']['diagnostics'].get('invalid_entities', 0)}
        for level in ['token', 'entity']:
            values = [by_version[v][level]['f1'] for v in EVALUATORS]
            delta[f'{level}_max_range_pp'] = 100 * (max(values) - min(values))
            for version in ['v2', 'v3']:
                delta[f'{level}_{version}_minus_v1_pp'] = 100 * (by_version[version][level]['f1'] - by_version['v1'][level]['f1'])
        differences.append(delta)
        results[name] = by_version
        print('Re-scored ' + name, flush=True)
    ranks = {level: {v: rank_groups(specs, results, v, level) for v in EVALUATORS} for level in ['token', 'entity']}
    shot_ranks = {family: {level: {v: rank_groups(specs, results, v, level, family) for v in EVALUATORS}
                          for level in ['token', 'entity']} for family in ['deepseek', 'glm', 'mimo']}
    rank_same = {level: all([[g['methods'] for g in ranks[level][v]] == [g['methods'] for g in ranks[level]['v1']] for v in EVALUATORS]) for level in ranks}
    comparisons = {v: {s['method']: {level: results[s['method']][v][level]['f1'] - results['crf'][v][level]['f1']
                                   for level in ['token', 'entity']} for s in specs} for v in EVALUATORS}
    max_delta = {level: max(d[f'{level}_max_range_pp'] for d in differences) for level in ['token', 'entity']}
    summary = {'methods': 15, 'evaluations': 45, 'test_samples': 1209,
               'rank_order_identical_across_versions': rank_same,
               'max_f1_range_pp': max_delta, 'ranks': ranks, 'shot_ranks': shot_ranks,
               'differences_from_crf': comparisons, 'new_api_calls': 0,
               'note': 'Same saved predictions; no model change or new generation. No report protocol changed.'}
    csv_write(REPORT / 'metrics.csv', rows)
    csv_write(REPORT / 'differences.csv', differences)
    save_json(REPORT / 'all_metrics.json', results)
    save_json(REPORT / 'summary.json', summary)
    write_jsonl(LOCAL / 'affected_samples.jsonl', affected)
    unchanged = all(file_hash(ROOT / path) == digest for path, digest in hashes.items())
    assert unchanged, 'Protected input changed'
    save_json(LOCAL / 'manifest.json', {'created_at': utc_now(), 'methods': 15, 'evaluations': 45,
        'new_api_calls': 0, 'original_files_unchanged': unchanged, 'protected_files': hashes,
        'checks': {'all_v3_exact_replay': True, 'v1_v2_token_equal': True, 'v1_entity_equals_v3_invalid_penalized': True},
        'code_hashes': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in [Path(__file__),
            ROOT/'scripts/integrate_results.py', ROOT/'src/ner_experiment/common.py',
            ROOT/'src/ner_experiment/evaluation/metrics_v1.py', ROOT/'src/ner_experiment/evaluation/metrics_v2.py',
            ROOT/'src/ner_experiment/evaluation/metrics.py']}})
    lines = ['# 当前15组预测的V1/V2/V3离线重评', '',
             '固定同一1209句Test与保存预测，只改变评估器；不训练、不调用API、不改原指标或报告。', '',
             '| 版本 | Token口径 | 实体口径 |', '|---|---|---|',
             '| V1 | 固定去重叠后单标签BIO | 原始去重跨度，无效项计FP |',
             '| V2 | 固定去重叠后单标签BIO | 去重叠合法跨度 |',
             '| V3 | 合法跨度展开为字符位置+BIO集合，保留重叠 | 全部合法去重跨度，无效项另报 |', '',
             '下表数值单位均为百分比。不是模型重新生成后的结果。', '',
             '| 方法 | Token V1 | Token V2 | Token V3 | 实体 V1 | 实体 V2 | 实体 V3 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for spec in specs:
        r = results[spec['method']]
        values = [100*r[v][level]['f1'] for level in ['token', 'entity'] for v in EVALUATORS]
        lines.append('| ' + spec['display'] + ' | ' + ' | '.join(f'{x:.4f}' for x in values) + ' |')
    lines += ['', '## 差异与排序', '',
              f"- 三版本最大Token F1跨度：{max_delta['token']:.6f}个百分点。",
              f"- 三版本最大实体F1跨度：{max_delta['entity']:.6f}个百分点。",
              f"- Token排序完全相同：{rank_same['token']}；实体排序完全相同：{rank_same['entity']}。", '',
              '| 方法 | 受影响句数 | 重叠句数 | 无效项 | V3−V1 Token百分点 | V2−V1 实体百分点 | V3−V1 实体百分点 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for d in differences:
        lines.append(f"| {d['display']} | {d['affected_sentences']} | {d['overlap_samples']} | {d['invalid_entities']} | {d['token_v3_minus_v1_pp']:+.6f} | {d['entity_v2_minus_v1_pp']:+.6f} | {d['entity_v3_minus_v1_pp']:+.6f} |")
    lines += ['', '## 复核与复现', '',
              '15组V3重算与现存metrics.json逐项完全一致。全部V1实体P/R/F1与V3已有无效项计FP辅助指标完全相同；V1与V2的Token指标完全相同。',
              '源预测、数据、原指标、配置和原整合CSV均经过前后SHA256检查，未变化。受影响原文仅保存于本地runs，公开表不包含原始句子。',
              '完整P/R/F1与TP/FP/FN见metrics.csv；逐模型差值见differences.csv；排名和Shot排序见summary.json。',
              '重评保存于runs/evaluation_versions/current_15。尚未切换默认评估器或修改正式报告。', '',
              '复现：`.\\.venv\\Scripts\\python.exe scripts/compare_current_evaluation_versions.py`。']
    (REPORT / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'max_f1_range_pp': max_delta, 'rank_same': rank_same,
                      'affected_method_sentence_pairs': len(affected), 'original_files_unchanged': unchanged}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
