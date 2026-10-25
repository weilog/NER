"""Paired sentence-level contrasts for the preregistered index experiment."""
from collections import Counter
from pathlib import Path
import json
import numpy as np

from ner_experiment.data.splitting import read_jsonl, json_bytes, sha
from ner_experiment.evaluation.metrics import prf
from ner_experiment.experiments.traditional import atomic_write, csv_bytes

NAMES = ['A_pairs_brief', 'B_columns_brief', 'C_pairs_explicit', 'D_columns_explicit']
CONTRASTS = {'B-A': [-1,1,0,0], 'C-A': [-1,0,1,0], 'D-B': [0,-1,0,1],
             'D-C': [0,0,-1,1], '(D-C)-(B-A)': [1,-1,-1,1]}


def f1(counts):
    counts = np.asarray(counts, dtype=float)
    numerator = 2 * counts[..., 0]
    denominator = numerator + counts[..., 1] + counts[..., 2]
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)


def paired_contrasts(arms, seed, repeats=2000):
    ids = [row['id'] for row in arms[NAMES[0]]]
    if not ids or any([row['id'] for row in arms[name]] != ids for name in NAMES):
        raise ValueError('All arms must have identical nonempty ordered IDs')
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(repeats, len(ids)))
    endpoints = {}
    for endpoint in ('token', 'entity', 'surface'):
        counts = np.array([[row[endpoint] for row in arms[name]] for name in NAMES])
        endpoints[endpoint + '_f1'] = (f1(counts.sum(axis=1)), f1(counts[:, draws, :].sum(axis=2)))
    positions = np.array([[row['position_error'] for row in arms[name]] for name in NAMES])
    endpoints['position_error_rate'] = (positions.mean(axis=1), positions[:, draws].mean(axis=2))
    output = []
    for name, weights in CONTRASTS.items():
        weight = np.array(weights)
        for endpoint, (observed, bootstrapped) in endpoints.items():
            distribution = weight @ bootstrapped
            output.append({'contrast': name, 'endpoint': endpoint,
                           'difference': float(weight @ observed),
                           'ci95_low': float(np.quantile(distribution, .025)),
                           'ci95_high': float(np.quantile(distribution, .975)),
                           'seed': seed, 'repeats': repeats,
                           'interpretation': 'lower is better' if endpoint == 'position_error_rate' else 'higher is better'})
    return output


def main(root):
    root = Path(root)
    config = json.loads((root / 'configs/index_factorial_v1.json').read_text(encoding='utf-8'))
    directory = root / config['run_directory']
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if not summary['complete']:
        raise ValueError('Cannot analyze an incomplete experiment as completed')
    context = json.loads((directory / 'context.json').read_text(encoding='utf-8'))
    if sha(json_bytes(context)) != summary['context_sha256']:
        raise ValueError('Context checksum changed')
    for name, digest in context['source_sha256'].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError('Inference/scoring source changed: ' + name)
    records = read_jsonl(directory / 'records.jsonl')
    arm_data, rows, classes, examples = {}, [], [], []
    for name in NAMES:
        for filename, digest in summary['arms'][name]['output_sha256'].items():
            if sha((directory / name / filename).read_bytes()) != digest:
                raise ValueError('Output checksum changed: ' + name + '/' + filename)
        metrics = json.loads((directory / name / 'metrics.json').read_text(encoding='utf-8'))
        diagnostic = json.loads((directory / name / 'diagnostics.json').read_text(encoding='utf-8'))
        by_id = {row['id']: row for row in diagnostic['sentence_counts']}
        arm_data[name] = [{**row, 'surface': by_id[row['id']]['surface'],
                           'position_error': by_id[row['id']]['position_error']}
                          for row in metrics['sentence_counts']]
        own = [record for record in records if record['arm'] == name]
        durations = [record['network_seconds'] for record in own]
        usage = Counter()
        for record in own:
            for attempt in record['attempts']:
                detail = json.loads(attempt['cost_detail_json'] or '{}')
                for field in ('prompt_tokens','completion_tokens','cache_hit_tokens','cache_miss_tokens'):
                    usage[field] += detail.get(field,0)
        d = diagnostic['counts']; md = metrics['diagnostics']
        rows.append({'arm': name, **{f'{level}_{key}':metrics[level][key] for level in ('token','entity')
                                      for key in ('precision','recall','f1','tp','fp','fn')},
                     'surface_type_bag_f1': diagnostic['surface_type_bag']['f1'],
                     'position_error_samples': d.get('position_error_samples',0),
                     'position_error_sample_rate': diagnostic['position_error_sample_rate'],
                     'inclusive_end_signature_entities': d.get('inclusive_end_signature_entities',0),
                     'unique_surface_relocatable_entities': d.get('unique_surface_relocatable_entities',0),
                     'raw_entity_entries': d.get('raw_entity_entries',0),
                     'invalid_entities': md.get('invalid_entities',0),
                     'overlap_removed': md.get('overlap_entities_removed',0),
                     'duplicate_entities': md.get('duplicate_entities',0),
                     'failure_rate': metrics['rates']['failure_rate'],
                     'raw_span_entity_f1': metrics['sensitivity']['raw_span_entity']['f1'],
                     'median_network_seconds': float(np.median(durations)),
                     'p95_network_seconds': float(np.percentile(durations,95)),
                     'accounted_cny': sum(record['accounted_cny'] for record in own),
                     **usage})
        for label, score in metrics['per_type_entity'].items():
            classes.append({'arm':name,'type':label,**score})
        for event in diagnostic['position_events']:
            examples.append({'arm':name,**event})
    contrasts = paired_contrasts(arm_data, config['bootstrap_seed'], config['bootstrap_repeats'])
    atomic_write(root / 'reports/index-factorial-results.csv', csv_bytes(rows))
    atomic_write(root / 'reports/index-factorial-per-type.csv', csv_bytes(classes))
    atomic_write(root / 'reports/index-factorial-contrasts.csv', csv_bytes(contrasts))
    atomic_write(directory / 'analysis/contrasts.json', json_bytes(contrasts))
    atomic_write(directory / 'analysis/position-cases.json', json_bytes(examples))
    labels = {name:name[0] for name in NAMES}
    lines = ['# 字符位置表示与终点解释：四组真实 Dev 对照', '',
             '本分支的独立对照实验，区别于 main 的 0/2/4/8-shot 方案比较。固定模型、四条 Train 示例及 200 条 Dev，只改变索引表示和终点解释。', '',
             'A=普通索引+简短说明；B=三列边界索引+简短说明；C=普通索引+明确终点解释；D=三列边界索引+明确终点解释。', '',
             '样本排除了已有试跑和调参子集，但它们仍属于 Dev，且传统模型此前在完整 Dev 选参；本报告不是新的独立 Test 成绩。', '',
             '| 组 | Token P (%) | Token R (%) | Token F1 (%) | 实体 P (%) | 实体 R (%) | 实体 F1 (%) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [f'{row[f"{level}_{key}"]*100:.2f}' for level in ('token','entity') for key in ('precision','recall','f1')]
        lines.append('| ' + ' | '.join([labels[row['arm']],*values]) + ' |')
    lines += ['', '## 定位与输出质量', '',
              '| 组 | 坐标错误句子/200 | 文字+类别 bag F1 (%) | 原始跨度 F1 (%) | 无效条目 | 重叠删除 | 失败率 (%) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {labels[row["arm"]]} | {row["position_error_samples"]} | {row["surface_type_bag_f1"]*100:.2f} | {row["raw_span_entity_f1"]*100:.2f} | {row["invalid_entities"]} | {row["overlap_removed"]} | {row["failure_rate"]*100:.2f} |')
    lines += ['', '主坐标错误率按全部句子计，不剔除调用失败；必须与失败率、预测条目数和召回一起阅读。所有组同一规范化器，主指标不自动修坐标。',
              '文字+类别 bag 忽略坐标，只提供诊断；同名实体多次出现无法据此区分具体位置，不能代替严格实体成绩。两种 F1 的差不是严格的定位错误百分比。', '',
              '## 预定配对比较', '',
              '差值均以百分点表示。坐标错误率差值越低越好，F1 差值越高越好。2000 次句子级配对重采样，每次汇总计数重算 micro F1。', '',
              '| 比较 | 坐标错误率差 [95%区间] | 实体 F1 差 [95%区间] |', '|---|---:|---:|']
    for comparison in CONTRASTS:
        selected = {item['endpoint']:item for item in contrasts if item['contrast']==comparison}
        format_difference = lambda item:f'{item["difference"]*100:+.2f} [{item["ci95_low"]*100:+.2f}, {item["ci95_high"]*100:+.2f}]'
        lines.append('| ' + ' | '.join([comparison,format_difference(selected['position_error_rate']),format_difference(selected['entity_f1'])]) + ' |')
    lines += ['', '这些是探索性、未做多重比较校正的描述性区间。没有覆盖重新生成、服务版本变化或标注不确定性，不能将区间跨零解释成“没有任何作用”，也不能把不跨零升级为普遍因果结论。', '',
              '## 成本与延迟', '', '| 组 | 估算费用(元) | 输入 tokens | 输出 tokens | 网络中位数/秒 | P95/秒 |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {labels[row["arm"]]} | {row["accounted_cny"]:.6f} | {row["prompt_tokens"]} | {row["completion_tokens"]} | {row["median_network_seconds"]:.3f} | {row["p95_network_seconds"]:.3f} |')
    models = sorted({r.get('response_model','unknown') for r in records})
    fingerprints = sorted({str(r.get('system_fingerprint')) for r in records})
    lines += ['', f'实际 API 返回模型标识：`{models}`；返回 system_fingerprint 集合：`{fingerprints}`。',
              f'本阶段累计估算 {summary["stage_accounted_cny"]:.6f} 元，全实验累计 {summary["accounting"]["accounted_cny"]:.6f} 元；上限分别 12/100 元，不是供应商账单。',
              '延迟沿用首次真实请求的网络秒数，包含必要重试，不含退避等待、排队和读缓存；四并发是吞吐配置。',
              '价格按 2026-10-25 官方人民币快照记录，见 configs/deepseek_pricing_2026-10-25.json。', '',
              '## 真实坐标错误案例', '']
    if not examples:
        lines += ['四组均未检测到实体文字与坐标不一致或越界；这不等于实体均正确，需要继续检查语义边界、类别、漏抽及重叠。', '']
    else:
        for i,event in enumerate(examples[:6],1):
            lines += [f'### {i}. {labels[event["arm"]]} / {event["id"]}', '',event['text'],'',
                      '模型原始实体：`'+json.dumps(event['entity'],ensure_ascii=False)+'`。',
                      f'实体文字在原文出现位置：`{event["surface_occurrences"]}`；闭区间终点特征：{event["inclusive_end_signature"]}。',
                      '只做诊断，没有据此修正评分；案例尚未经过两名人工独立核验。','']
    lines += ['## 复现与边界', '',
              f'实验上下文指纹：`{summary["context_sha256"]}`。预注册提交与样本清单早于新增真实请求；完整请求、响应、逐句计数、源码快照和账本保留本地。',
              '运行 `scripts/run_index_factorial.py --offline` 必须命中全部缓存，不允许网络调用，并要求四组预测、指标、规范化和诊断文件逐字节一致。',
              '本轮不读取项目 Test；不按结果换样本、扩参数或修改提示。是否推广到 Test 留待后续方案冻结，不能用本轮 Dev 成绩宣称最终泛化提升。', '']
    atomic_write(root / 'reports/index-factorial-results.md','\n'.join(lines).encode('utf-8'))
    print(json.dumps({'arms':rows,'stage_cost_cny':summary['stage_accounted_cny'],'position_cases':len(examples)},indent=2))


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[3])
