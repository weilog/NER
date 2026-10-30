"""Offline integration of 15 frozen methods. No API client imports, no .env access."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np
from ner_experiment.common import ROOT, LABELS, file_hash, load_json, object_hash, project_data, read_jsonl, save_json, write_jsonl
from ner_experiment.evaluation.metrics_v1 import evaluate
from ner_experiment.evaluation.metrics import evaluate as historical_evaluate
from analyze_current_results import csv_write, shared_bootstrap


SHOTS = [0, 2, 4, 8]
FAMILIES = ['deepseek', 'glm', 'mimo']
NAMES = {'dictionary': 'Dictionary', 'hmm': 'HMM', 'crf': 'CRF',
         'deepseek': 'DeepSeek', 'glm': 'GLM', 'mimo': 'MiMo'}
TABLES = ROOT / 'reports/tables/integrated_v1'
LOCAL = ROOT / 'runs/integrated_v1_analysis'


def specifications(root=ROOT):
    specs = []
    for name in ['dictionary', 'hmm', 'crf']:
        specs.append({'method': name, 'family': name, 'shots': '', 'display': NAMES[name],
                      'folder': root / 'runs/final' / name,
                      'reference': root / 'runs/evaluation_versions/v3/final' / name / 'metrics.json'})
    folders = {'deepseek': 'llm_text_test_v1', 'glm': 'glm_flashx_test_v1', 'mimo': 'mimo_flash_test_v1'}
    for family, base in folders.items():
        for shot in SHOTS:
            folder = root / 'runs' / base / f'shot_{shot}'
            specs.append({'method': f'{family}_{shot}', 'family': family, 'shots': shot,
                          'display': f'{NAMES[family]} {shot}-shot', 'folder': folder,
                          'reference': folder / 'metrics.json', 'base': root / 'runs' / base})
    return specs


def comparison_pairs():
    pairs = [(f'{f}_{s}', 'crf') for f in FAMILIES for s in SHOTS]
    pairs += [(f'{a}_{s}', f'{b}_{s}') for a, b in [('deepseek', 'glm'), ('deepseek', 'mimo'), ('glm', 'mimo')] for s in SHOTS]
    pairs += [(f'{f}_{a}', f'{f}_{b}') for f in FAMILIES for a, b in [(2, 0), (4, 0), (8, 0), (4, 2), (8, 2), (8, 4)]]
    return pairs


def ordered_predictions(gold, predictions):
    ids = [r['id'] for r in gold]
    lookup = {r['id']: r for r in predictions}
    if len(lookup) != len(predictions) or set(lookup) != set(ids):
        raise ValueError('Missing, duplicated or extra prediction IDs')
    return [lookup[i] for i in ids]


def failure_kind(prediction):
    if prediction['status'] == 'ok':
        return ''
    if prediction['status'] == 'format_error':
        return 'format_or_structure'
    error = prediction.get('error') or {}
    if error.get('http_status') is None and error.get('exception') in {
            'URLError', 'TimeoutError', 'ConnectionError', 'ConnectionRefusedError',
            'RemoteDisconnected', 'ConnectionResetError', 'OSError'}:
        return 'connection'
    if error.get('http_status') == 421 or error.get('provider_error_code') == '1301':
        return 'provider_content_refusal'
    return 'other_api_error'


def network_subset_ids(gold, predictions_by_method):
    removed = {p['id'] for predictions in predictions_by_method.values()
               for p in predictions if failure_kind(p) == 'connection'}
    return [r['id'] for r in gold if r['id'] not in removed], [r['id'] for r in gold if r['id'] in removed]


def cost_for_predictions(predictions, attempts):
    keys = {p['cache_key'] for p in predictions}
    relevant = [a for a in attempts if a['key'] in keys]
    known = [a for a in relevant if a['status'] == 'response']
    def total_tokens(field):
        return sum((a.get('usage') or {}).get(field, 0) for a in known)
    hit = sum(((a.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens',
              (a.get('usage') or {}).get('prompt_cache_hit_tokens', 0)) for a in known)
    prompt = total_tokens('prompt_tokens')
    return {'attempts_including_retries': len(relevant), 'returned_attempts': len(known),
            'known_usage_estimate_cny': sum(a['charged_upper_cny'] for a in known),
            'unknown_reservations_cny': sum(a['charged_upper_cny'] for a in relevant if a['status'] != 'response'),
            'gross_ledger_upper_cny': sum(a['charged_upper_cny'] for a in relevant),
            'prompt_tokens': prompt, 'completion_tokens': total_tokens('completion_tokens'),
            'cached_input_tokens': hit, 'cache_hit_fraction': hit / prompt if prompt else 0,
            'reasoning_tokens_if_reported': sum(((a.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens', 0) for a in known)}


def percentile(values, q):
    return float(np.quantile(values, q)) if values else ''


def choose_cases(gold, all_cases, seed):
    gold_map = {r['id']: r for r in gold}
    chosen, used = [], set()
    preferences = {'boundary': ['mimo_4', 'glm_8', 'mimo_8'],
                   'type': ['glm_8', 'mimo_4', 'mimo_8'],
                   'missed': ['mimo_8', 'glm_8', 'mimo_4']}
    for kind, names in preferences.items():
        candidates = []
        for rank, method in enumerate(names):
            for case in all_cases[method]:
                if case['status'] != 'ok' or case['id'] in used:
                    continue
                for error in case['errors']:
                    if error['kind'] != kind:
                        continue
                    if kind == 'boundary' and case['token']['f1'] <= case['entity']['f1']:
                        continue
                    gold_row = gold_map[case['id']]
                    score = (0 if len(gold_row['entities']) == 1 else 1,
                             0 if len(case['errors']) == 1 else 1,
                             rank, len(case['text']), object_hash([seed, method, case['id']]))
                    candidates.append((score, method, case, error))
        if not candidates:
            raise RuntimeError('No real representative error available: ' + kind)
        _, method, case, error = min(candidates, key=lambda x: x[0])
        chosen.append({'kind': kind, 'method': method, 'id': case['id'], 'text': case['text'],
                       'matched_error': error, 'gold_entities': gold_map[case['id']]['entities'],
                       'raw_prediction': case['raw_prediction'], 'token': case['token'], 'entity': case['entity'],
                       'review': 'AI-assisted source-grounded illustration; not independent human review'})
        used.add(case['id'])
    return chosen


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    specs = specifications()
    protected = set()
    for spec in specs:
        protected.update(p for p in spec['folder'].glob('*.json*') if p.is_file())
        protected.add(spec['reference'])
    protected.update((ROOT / 'data/processed/cluener2020_project').glob('*.jsonl'))
    protected.update((ROOT / 'runs').rglob('ledger.json'))
    protected.update((ROOT / 'data/splits/cluener2020').glob('*.json'))
    protected.add(ROOT / 'configs/split_seed.json')
    for family in FAMILIES:
        base = next(s['base'] for s in specs if s['family'] == family)
        protected.update(base.glob('*protocol*.json'))
        protected.add(base / 'summary.json')
    before = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in protected}
    def tracked(path):
        before.setdefault(path.relative_to(ROOT).as_posix(), file_hash(path))
        return load_json(path)
    manifest = tracked(ROOT / 'data/splits/cluener2020/manifest.json')
    for split, info in manifest['files'].items():
        if file_hash(ROOT / info['path']) != info['sha256']:
            raise RuntimeError('Fixed dataset hash mismatch: ' + split)
    gold = project_data('test')
    dev = project_data('dev')
    if len(gold) != 1209 or len(dev) != 1343:
        raise RuntimeError('Unexpected fixed data sizes')
    seed = tracked(ROOT / 'configs/split_seed.json')['seed']
    protocol_by_family, ledgers = {}, {}
    for family in FAMILIES:
        base = next(s['base'] for s in specs if s['family'] == family)
        protocol_by_family[family] = tracked(base / 'frozen_protocol.json')['protocol']
        ledgers[family] = tracked(base / 'ledger.json')['attempts']
    systems = {object_hash(p['system']) for p in protocol_by_family.values()}
    example_hashes = {object_hash(p['examples']) for p in protocol_by_family.values()}
    if len(systems) != 1 or len(example_hashes) != 1:
        raise RuntimeError('LLM prompts/examples are not identical across providers')
    metrics, counts, cases, predictions = {}, {}, {}, {}
    diagnostics, costs, protocol_rows = [], [], []
    by_gold = {r['id']: r for r in gold}
    for spec in specs:
        name, folder = spec['method'], spec['folder']
        ordered = ordered_predictions(gold, read_jsonl(folder / 'predictions.jsonl'))
        m, c, e = evaluate(gold, ordered)
        if historical_evaluate(gold, ordered)[0] != load_json(spec['reference']):
            raise RuntimeError('Saved V3 metrics differ: ' + name)
        prediction_by_id = {p['id']: p for p in ordered}
        for case in e:
            case['raw_prediction'] = prediction_by_id[case['id']]
        metrics[name], counts[name], cases[name], predictions[name] = m, c, e, ordered
        save_json(LOCAL / name / 'metrics.json', m)
        write_jsonl(LOCAL / name / 'sentence_counts.jsonl', c)
        write_jsonl(LOCAL / name / 'errors.jsonl', e)
        failures = Counter(failure_kind(p) for p in ordered if p['status'] != 'ok')
        failed_gold = sum(len(by_gold[p['id']]['entities']) for p in ordered if p['status'] != 'ok')
        network_gold = sum(len(by_gold[p['id']]['entities']) for p in ordered if failure_kind(p) == 'connection')
        detail = Counter()
        mismatches = valid_spans = 0
        for row, p in zip(gold, ordered):
            if p['status'] == 'format_error' and 'base' in spec:
                raw = tracked(spec['base'] / 'cache' / (p['cache_key'] + '.json')).get('raw_response') or {}
                choice = (raw.get('choices') or [{}])[0]
                if choice.get('finish_reason') != 'stop':
                    detail['finish_reason:' + str(choice.get('finish_reason'))] += 1
                else:
                    try:
                        parsed = json.loads((choice.get('message') or {}).get('content', ''))
                        detail['schema_error'] += 1
                    except (ValueError, TypeError):
                        detail['invalid_json'] += 1
            if p['status'] == 'ok':
                for entity in p['entities']:
                    start, end = entity.get('start'), entity.get('end')
                    if type(start) is int and type(end) is int and 0 <= start < end <= len(row['text']) and entity.get('type') in LABELS:
                        valid_spans += 1
                        if 'text' in entity:
                            mismatches += entity['text'] != row['text'][start:end]
        diagnostics.append({'method': name, 'samples': len(gold),
            'api_error_count': sum(p['status'] == 'api_error' for p in ordered),
            'format_error_count': sum(p['status'] == 'format_error' for p in ordered),
            'network_error_count': failures['connection'], 'provider_refusal_count': failures['provider_content_refusal'],
            'other_api_error_count': failures['other_api_error'], 'failure_rate': m['failure_rate'],
            'failed_gold_entities': failed_gold, 'failed_entity_fn_share': failed_gold / m['entity']['fn'] if m['entity']['fn'] else 0,
            'network_failure_gold_entities': network_gold,
            'format_details': json.dumps(detail, ensure_ascii=False, sort_keys=True),
            'overlap_samples': m['diagnostics'].get('overlap_projection_samples', 0),
            'invalid_entities': m['diagnostics'].get('invalid_entities', 0),
            'duplicate_entities': m['diagnostics'].get('duplicate_entities', 0),
            'illegal_bio_rate': m['illegal_bio_rate'],
            'text_span_mismatches': mismatches if 'base' in spec else '',
            'valid_output_spans': valid_spans, 'text_mismatch_rate': mismatches / valid_spans if valid_spans else 0})
        latency = [p['latency_seconds'] for p in ordered if 'latency_seconds' in p]
        single_ok = [p['latency_seconds'] for p in ordered if p['status'] == 'ok' and p.get('attempts') == 1]
        if 'base' in spec:
            cost = cost_for_predictions(ordered, ledgers[spec['family']])
            params = protocol_by_family[spec['family']]['parameters']
            workers = Counter(str(p.get('execution_workers', 8 if spec['family'] == 'deepseek' else 1)) for p in ordered)
            relevant = [a for a in ledgers[spec['family']] if a['key'] in {p['cache_key'] for p in ordered}]
            dates = sorted({a['started_at'][:10] for a in relevant})
            models = sorted({p.get('actual_model') or 'unknown' for p in ordered})
            protocol_rows.append({'family': spec['family'], 'shots': spec['shots'], 'requested_model': params['model'],
                'returned_models': ','.join(models), 'thinking': params.get('thinking', {}).get('type', ''),
                'reasoning_effort': params.get('reasoning_effort', ''), 'temperature': params.get('temperature'),
                'output_limit': params.get('max_tokens', params.get('max_completion_tokens')),
                'worker_counts': json.dumps(workers, sort_keys=True),
                'global_rpm': 90 if spec['family'] == 'mimo' else '', 'utc_call_dates': ','.join(dates)})
            http = [p['http_seconds'] for p in ordered if p['status'] == 'ok' and p.get('attempts') == 1 and 'http_seconds' in p]
            cost.update(http_only_median_seconds=percentile(http, .5), http_only_p95_seconds=percentile(http, .95))
        else:
            run = tracked(folder / 'run.json')
            latency = []
            cost = {k: '' for k in ['attempts_including_retries', 'returned_attempts', 'prompt_tokens', 'completion_tokens',
                     'cached_input_tokens', 'cache_hit_fraction', 'reasoning_tokens_if_reported', 'http_only_median_seconds', 'http_only_p95_seconds']}
            cost.update(known_usage_estimate_cny=0, unknown_reservations_cny=0, gross_ledger_upper_cny=0)
        cost.update(method=name, family=spec['family'], shots=spec['shots'], samples=len(gold),
            latency_median_seconds=percentile(latency, .5) if latency else run['latency_median_seconds'],
            latency_p95_seconds=percentile(latency, .95) if latency else run['latency_p95_seconds'],
            single_attempt_ok_median_seconds=percentile(single_ok, .5),
            single_attempt_ok_p95_seconds=percentile(single_ok, .95),
            latency_scope='includes client throttle/retries; original timings retained' if 'base' in spec else 'CPU per-sentence inference; not API or total training cost')
        costs.append(cost)
        print('VERIFIED', name, f'entity F1={m["entity"]["f1"]:.6f}', flush=True)
    kept, removed = network_subset_ids(gold, predictions)
    kept_set = set(kept)
    gold_subset = [r for r in gold if r['id'] in kept_set]
    if not gold_subset:
        raise RuntimeError('Empty shared network diagnostic subset')
    subset_metrics, subset_counts, subset_rows = {}, {}, []
    for spec in specs:
        name = spec['method']
        ps = [p for p in predictions[name] if p['id'] in kept_set]
        m, c, _ = evaluate(gold_subset, ps)
        subset_metrics[name], subset_counts[name] = m, c
        subset_rows.append({'method': name, 'family': spec['family'], 'shots': spec['shots'], 'samples': len(kept),
            'full_token_f1': metrics[name]['token']['f1'], 'full_entity_f1': metrics[name]['entity']['f1'],
            'subset_token_f1': m['token']['f1'], 'subset_entity_f1': m['entity']['f1'],
            'subset_token_precision': m['token']['precision'], 'subset_token_recall': m['token']['recall'],
            'subset_entity_precision': m['entity']['precision'], 'subset_entity_recall': m['entity']['recall']})
    pairs = comparison_pairs()
    bootstrap = shared_bootstrap(counts, seed, repeats=2000, pairs=pairs)
    subset_bootstrap = shared_bootstrap(subset_counts, seed, repeats=2000, pairs=pairs)
    save_json(TABLES / 'bootstrap.json', bootstrap)
    save_json(TABLES / 'network_subset_bootstrap.json', subset_bootstrap)
    main_rows, per_class, token_labels, error_rows = [], [], [], []
    for spec in specs:
        name, m = spec['method'], metrics[spec['method']]
        row = {'method': name, 'display': spec['display'], 'family': spec['family'], 'shots': spec['shots'],
            'samples': len(gold), 'evaluation_protocol': 'v1', 'failure_rate': m['failure_rate'],
            'entity_macro_f1': m['entity_macro_f1'], 'character_accuracy': m['character_accuracy'],
            'entity_with_invalid_fp_f1': m['entity']['f1'],
            'invalid_sample_rate': m['invalid_sample_rate'],
            'selection': 'Historical Dev-selected traditional config' if 'base' not in spec else 'Fixed shot setting; no new Dev selection; never selected by Test ranking'}
        for level in ['token', 'entity']:
            row.update({f'{level}_{k}': m[level][k] for k in ['precision', 'recall', 'f1', 'tp', 'fp', 'fn']})
            row[f'{level}_f1_ci_low'], row[f'{level}_f1_ci_high'] = bootstrap['methods'][name][level]['ci95']
        main_rows.append(row)
        per_class.extend({'method': name, 'category': k, **v} for k, v in m['entity_per_type'].items())
        token_labels.extend({'method': name, 'bio_label': k, **v} for k, v in m['token_per_bio_label'].items())
        error_rows.append({'method': name, **{k: m['error_counts'].get(k, 0) for k in ['boundary', 'type', 'boundary_and_type', 'missed', 'spurious']}})
    comparison_rows = []
    for pair in bootstrap['comparisons']:
        for level in ['token', 'entity']:
            d = pair[level]
            comparison_rows.append({'a': pair['a'], 'b': pair['b'], 'metric': level,
                'difference_pp': d['difference_a_minus_b'] * 100, 'ci_low_pp': d['ci95'][0] * 100,
                'ci_high_pp': d['ci95'][1] * 100, 'contains_zero': d['ci95'][0] <= 0 <= d['ci95'][1],
                'scope': 'exploratory fixed-prediction sentence pairs; no multiple-comparison correction'})
    for row in subset_rows:
        for level in ['token', 'entity']:
            row[f'subset_{level}_f1_ci_low'], row[f'subset_{level}_f1_ci_high'] = subset_bootstrap['methods'][row['method']][level]['ci95']
    for name, rows in [('main_results', main_rows), ('entity_per_type', per_class), ('token_bio_per_label', token_labels),
                       ('error_counts', error_rows), ('cost_latency', costs), ('diagnostics', diagnostics),
                       ('api_protocols', protocol_rows), ('bootstrap_comparisons', comparison_rows), ('network_subset', subset_rows)]:
        csv_write(TABLES / (name + '.csv'), rows)
    subset_record = {'full_samples': len(gold), 'kept_samples': len(kept), 'removed_samples': len(removed),
        'kept_ids': kept, 'removed_ids': removed,
        'rule': 'Remove union of explicit connection-failed sentence IDs across every compared method; all 15 methods use identical kept IDs. Keep format/content/unknown-HTTP failures.',
        'limitation': 'Post hoc, nonrandom conditional diagnostic, not repaired/full-Test performance or unbiased population estimate'}
    save_json(TABLES / 'network_subset_ids.json', subset_record)
    # Re-evaluate historical Dev parameter predictions under V3 without changing selection.
    selection = tracked(ROOT / 'runs/baselines/selection.json')
    parameters = []
    for run in selection['runs']:
        folder = ROOT / 'runs/baselines' / run['name']
        path = folder / 'predictions.jsonl'
        before[path.relative_to(ROOT).as_posix()] = file_hash(path)
        m, _, _ = evaluate(dev, ordered_predictions(dev, read_jsonl(path)))
        row = {'name': run['name'], 'method': run['method'], 'config': json.dumps(run.get('config', {}), sort_keys=True),
            'selected': run['name'] in selection['selected'].values(), 'historical_dev_entity_f1': run['entity_f1'],
            'v3_dev_entity_f1': m['entity']['f1'], 'v3_dev_token_f1': m['token']['f1'],
            'fit_seconds': run['fit_seconds'], 'predict_seconds': run['predict_seconds'],
            'vocabulary_size': run.get('vocabulary_size', ''), 'dev_unknown_char_rate': run.get('dev_unknown_char_rate', ''),
            'unconstrained_entity_f1': '', 'unconstrained_token_f1': '', 'unconstrained_illegal_bio_rate': ''}
        if run['method'] == 'hmm':
            path = folder / 'unconstrained/predictions.jsonl'
            before[path.relative_to(ROOT).as_posix()] = file_hash(path)
            free, _, _ = evaluate(dev, ordered_predictions(dev, read_jsonl(path)))
            row.update(unconstrained_entity_f1=free['entity']['f1'], unconstrained_token_f1=free['token']['f1'],
                       unconstrained_illegal_bio_rate=free['illegal_bio_rate'])
        parameters.append(row)
    csv_write(TABLES / 'baseline_dev_parameters.csv', parameters)
    selected_cases = choose_cases(gold, cases, seed)
    save_json(LOCAL / 'representative_cases.json', selected_cases)
    all_ledgers = {}
    for path in (ROOT / 'runs').rglob('ledger.json'):
        attempts = tracked(path)['attempts']
        all_ledgers[path.relative_to(ROOT).as_posix()] = {
            'attempts': len(attempts),
            'known_usage_estimate_cny': sum(a['charged_upper_cny'] for a in attempts if a['status'] == 'response'),
            'unknown_reservations_cny': sum(a['charged_upper_cny'] for a in attempts if a['status'] != 'response'),
            'gross_upper_cny': sum(a['charged_upper_cny'] for a in attempts)}
    project_cost = {k: sum(v[k] for v in all_ledgers.values()) for k in ['known_usage_estimate_cny', 'unknown_reservations_cny', 'gross_upper_cny']}
    project_cost.update(ledgers=all_ledgers, note='Recorded historical price estimates, not invoices; includes older prompts/Dev/pilots. Raw unknown GLM reservations retained.')
    save_json(TABLES / 'project_cost.json', project_cost)
    for path, digest in before.items():
        if file_hash(ROOT / path) != digest:
            raise RuntimeError('Protected input changed during integration: ' + path)
    verification = {'checked_at': now, 'methods': [s['method'] for s in specs], 'samples': len(gold),
        'all_fifteen_metrics_replayed_exactly': True, 'shared_llm_system_and_examples': True,
        'input_files_unchanged': True, 'source_hashes': before, 'new_api_calls': 0,
        'bootstrap_repeats': 2000, 'seed': seed, 'comparison_pairs': pairs,
        'network_subset_samples': len(kept), 'network_removed_samples': len(removed),
        'analysis_code_sha256': file_hash(Path(__file__).resolve()),
        'analysis_source_hashes': {p: file_hash(ROOT / p) for p in [
            'scripts/integrate_v1_results.py', 'scripts/write_v1_integrated_report.py',
            'scripts/analyze_current_results.py', 'src/ner_experiment/evaluation/metrics.py', 'src/ner_experiment/evaluation/metrics_v1.py',
            'src/ner_experiment/common.py']}}
    save_json(LOCAL / 'verification.json', verification)
    save_json(TABLES / 'verification.json', verification)
    from write_v1_integrated_report import write_report
    write_report(specs, metrics, diagnostics, costs, protocol_rows, bootstrap, subset_rows, subset_record,
                 parameters, selection, selected_cases, project_cost, now)
    print(f'INTEGRATION COMPLETE: {len(specs)} methods, {len(pairs)} paired comparisons, subset n={len(kept)}, API calls=0', flush=True)


if __name__ == '__main__':
    main()
