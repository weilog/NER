"""Train baseline grids, select on Dev entity F1, and save offline artifacts."""
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from ner_experiment.data.splitting import read_jsonl, json_bytes, jsonl_bytes, sha
from ner_experiment.evaluation.metrics import evaluate, bio_to_spans, entities_list
from ner_experiment.models.dictionary import DictionaryNER
from ner_experiment.models.hmm import CharacterHMM
from ner_experiment.models.crf import CharacterCRF, FEATURE_VERSION, sentence_features


def atomic_write(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(body)
    temporary.replace(path)


def choose_best(trials):
    if not trials:
        raise ValueError('No completed candidates')
    return min(trials, key=lambda trial: (-trial['entity']['f1'], trial['fit_seconds'], trial['configuration_order']))


def predict_rows(model, rows, family, constrained=True):
    predictions = []
    for row in rows:
        started = time.perf_counter()
        output = model.predict(row['text'], constrained=constrained) if family == 'hmm' else model.predict(row['text'])
        if family == 'dictionary':
            prediction = {'id': row['id'], 'status': 'ok', 'entities': output}
        else:
            spans, _, _ = bio_to_spans(output)
            prediction = {'id': row['id'], 'status': 'ok', 'raw_bio_tags': output,
                          'entities': entities_list(spans, row['text'])}
        prediction['elapsed_seconds'] = time.perf_counter() - started
        predictions.append(prediction)
    return predictions


def context_for(root, config):
    split = json.loads((root / config['dataset']['split_manifest']).read_text(encoding='utf-8'))
    data_paths = {name: root / config['dataset'][name + '_file'] for name in ('train', 'dev')}
    expected = {'train': split['output_sha256']['train_valid.jsonl'], 'dev': split['output_sha256']['dev.jsonl']}
    for name, path in data_paths.items():
        if sha(path.read_bytes()) != expected[name]:
            raise ValueError(f'{name} data hash differs from frozen manifest')
    train, dev = (read_jsonl(data_paths[name]) for name in ('train', 'dev'))
    if len(train) != 9538 or len(dev) != 1343:
        raise ValueError('Unexpected dataset counts')
    if set(row['id'] for row in train) != set(split['ids']['train']) - set(split['excluded_training_ids']):
        raise ValueError('Training IDs differ from frozen split')
    if any(row['bio_compatible'] is not True for row in train + dev):
        raise ValueError('Overlapping or unlabeled gold is not allowed here')
    if {row['text'] for row in train} & {row['text'] for row in dev}:
        raise ValueError('Train/Dev text leakage')
    package_versions = {name: importlib.metadata.version(name) for name in
                        ('numpy', 'scipy', 'scikit-learn', 'sklearn-crfsuite', 'python-crfsuite', 'joblib')}
    source_paths = sorted((root / 'src/ner_experiment').rglob('*.py'))
    context = {'protocol': config['protocol'], 'seed': split['seed'],
               'train_sha256': expected['train'], 'dev_sha256': expected['dev'],
               'split_manifest_sha256': sha((root / config['dataset']['split_manifest']).read_bytes()),
               'baseline_configuration': config['baselines'], 'selection': config['selection'],
               'evaluation_configuration': config['evaluation'],
               'source_sha256': {path.relative_to(root).as_posix(): sha(path.read_bytes()) for path in source_paths},
               'packages': package_versions, 'python': platform.python_version(),
               'feature_version': FEATURE_VERSION, 'threadpool_limit': 1,
               'prediction_timing': 'per sentence: features/decoder and prediction adapter; excludes IO/evaluation',
               'test_accessed': False}
    return context, train, dev


def completed_trial(folder, context_hash):
    path = folder / 'trial.json'
    if not path.exists():
        return None
    trial = json.loads(path.read_text(encoding='utf-8'))
    if trial['context_sha256'] != context_hash:
        raise ValueError(f'Different experiment context: {folder}')
    for name in ('predictions.jsonl', 'metrics.json', 'normalized.jsonl'):
        if sha((folder / name).read_bytes()) != trial['output_sha256'][name]:
            raise ValueError(f'Corrupted cached trial {folder / name}')
    return trial


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def run_development(root, run_directory='runs/traditional_dev'):
    root = Path(root)
    config = json.loads((root / 'configs/experiment_v2.json').read_text(encoding='utf-8'))
    context, train, dev = context_for(root, config)
    context_hash = sha(json_bytes(context))
    run = root / run_directory
    run.mkdir(parents=True, exist_ok=True)
    context_path = run / 'context.json'
    if context_path.exists() and context_path.read_bytes() != json_bytes(context):
        raise ValueError('Run context changed; use a new --run-dir, preserving old experiments')
    atomic_write(context_path, json_bytes(context))
    model_directory = root / 'artifacts/models' / run.name / context_hash[:12]
    model_directory.mkdir(parents=True, exist_ok=True)
    trials = []
    shared_feature_seconds = 0.0
    crf_features = None
    trained = {}

    def execute(name, family, parameters, order, model_key, factory, constrained=True):
        nonlocal crf_features, shared_feature_seconds
        folder = run / name
        trial = completed_trial(folder, context_hash)
        if trial is not None:
            model_path = root / trial['model_path']
            if sha(model_path.read_bytes()) != trial['model_sha256']:
                raise ValueError(f'Corrupted cached model {model_path}')
            # Offline recomputation validates that serialized predictions score identically.
            recalculated = evaluate(dev, read_jsonl(folder / 'predictions.jsonl'))
            recalculated.pop('normalized_predictions')
            cached = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
            if recalculated != cached:
                raise ValueError(f'Offline metrics differ: {folder}')
            trials.append(trial)
            print(f'RESUME {name}: entity F1={trial["entity"]["f1"]:.6f}', flush=True)
            return
        print(f'START {name}: {parameters}', flush=True)
        model_path = model_directory / f'{model_key}.joblib'
        # HMM constrained/unconstrained trials share exactly the same fitted arrays.
        if model_key in trained:
            model, fit_seconds, metadata = trained[model_key]
        elif model_path.exists() and (run / model_key / 'fit.json').exists():
            metadata = json.loads((run / model_key / 'fit.json').read_text(encoding='utf-8'))
            if metadata['model_sha256'] != sha(model_path.read_bytes()):
                raise ValueError('Cached model hash changed')
            model, fit_seconds = joblib.load(model_path), metadata['fit_seconds']
        else:
            if family == 'crf' and crf_features is None:
                started = time.perf_counter()
                crf_features = [sentence_features(row['text']) for row in train]
                shared_feature_seconds = time.perf_counter() - started
                print(f'CRF shared feature extraction: {shared_feature_seconds:.2f}s', flush=True)
            model = factory()
            started = time.perf_counter()
            if family == 'crf':
                model.fit_features(crf_features, [row['bio_tags'] for row in train])
            else:
                model.fit(train)
            fit_seconds = time.perf_counter() - started
            joblib.dump(model, model_path, compress=3)
            metadata = {'fit_seconds': fit_seconds, 'model_sha256': sha(model_path.read_bytes()),
                        'context_sha256': context_hash}
            atomic_write(run / model_key / 'fit.json', json_bytes(metadata))
        trained[model_key] = model, fit_seconds, metadata
        predictions = predict_rows(model, dev, family, constrained=constrained)
        predictions_body = jsonl_bytes(predictions)
        atomic_write(folder / 'predictions.jsonl', predictions_body)
        # Always score predictions read back from the persisted file.
        persisted = read_jsonl(folder / 'predictions.jsonl')
        metrics = evaluate(dev, persisted)
        normalized = metrics.pop('normalized_predictions')
        atomic_write(folder / 'metrics.json', json_bytes(metrics))
        atomic_write(folder / 'normalized.jsonl', jsonl_bytes(normalized))
        durations = [row['elapsed_seconds'] for row in predictions]
        trial = {'name': name, 'family': family, 'parameters': parameters,
                 'configuration_order': order, 'context_sha256': context_hash,
                 'fit_seconds': fit_seconds, 'model_path': model_path.relative_to(root).as_posix(),
                 'model_sha256': metadata['model_sha256'],
                 'token': metrics['token'], 'entity': metrics['entity'],
                 'character_accuracy_including_O': metrics['character_accuracy_including_O'],
                 'entity_macro_f1_fixed_10': metrics['entity_macro_f1_fixed_10'],
                 'rates': metrics['rates'],
                 'dev_prediction_seconds': sum(durations), 'latency_median_ms': float(np.median(durations) * 1000),
                 'latency_p95_ms': float(np.percentile(durations, 95) * 1000),
                 'vocabulary': model.vocabulary_statistics(row['text'] for row in dev) if family == 'hmm' else None,
                 'dictionary_surfaces': len(model.lexicon_) if family == 'dictionary' else None,
                 'output_sha256': {filename: sha((folder / filename).read_bytes()) for filename in
                                   ('predictions.jsonl', 'metrics.json', 'normalized.jsonl')}}
        atomic_write(folder / 'trial.json', json_bytes(trial))
        trials.append(trial)
        print(f'DONE {name}: token F1={trial["token"]["f1"]:.6f}, entity F1={trial["entity"]["f1"]:.6f}, fit={fit_seconds:.2f}s', flush=True)

    with threadpool_limits(limits=1):
        execute('dictionary', 'dictionary', config['baselines']['dictionary'], 0, 'dictionary', DictionaryNER)
        order = 0
        for alpha in config['baselines']['hmm']['alpha']:
            for threshold in config['baselines']['hmm']['min_char_count']:
                base = f'hmm_a{alpha:g}_min{threshold}'
                parameters = {'alpha': alpha, 'min_char_count': threshold}
                for constrained in (True, False):
                    execute(base + ('_bio' if constrained else '_plain'), 'hmm',
                            {**parameters, 'constrained': constrained}, order, base,
                            lambda p=parameters: CharacterHMM(**p), constrained=constrained)
                # Only one HMM model needs to stay resident after each paired comparison.
                trained.clear()
                order += 1
        crf_config = config['baselines']['crf']
        for order, (c1, c2) in enumerate(crf_config['c1_c2']):
            name = f'crf_c1_{c1:g}_c2_{c2:g}'
            parameters = {'c1': c1, 'c2': c2, 'max_iterations': crf_config['max_iterations']}
            execute(name, 'crf', parameters, order, name, lambda p=parameters: CharacterCRF(**p))
            trained.clear()
    selected = {}
    for family in ('dictionary', 'hmm', 'crf'):
        candidates = [trial for trial in trials if trial['family'] == family
                      and (family != 'hmm' or trial['parameters']['constrained'])]
        selected[family] = choose_best(candidates)
    best = choose_best(list(selected.values()))
    summary = {'stage': 'traditional_dev', 'date_time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'context_sha256': context_hash, 'run_directory': run.relative_to(root).as_posix(),
               'selection': config['selection'], 'selected': selected,
               'best_traditional_by_dev': best['name'], 'trials': trials,
               'shared_crf_feature_seconds_this_invocation': shared_feature_seconds,
               'test_evaluated': False, 'api_calls': 0, 'api_cost_cny': 0}
    atomic_write(run / 'summary.json', json_bytes(summary))
    selected_config = {'status': 'traditional_dev_selected; awaiting_llm_and_global_test_freeze',
                       'context_sha256': context_hash, 'run_directory': run.relative_to(root).as_posix(),
                       'selection': config['selection'], 'best_traditional_by_dev': best['name'],
                       'selected': {family: {key: trial[key] for key in ('name', 'parameters', 'model_path', 'model_sha256')}
                                    for family, trial in selected.items()}, 'test_evaluated': False}
    atomic_write(root / 'configs/traditional_selected.json', json_bytes(selected_config))
    export_reports(root, summary)
    return summary


def export_reports(root, summary):
    flat = []
    for trial in summary['trials']:
        flat.append({'configuration': trial['name'], 'family': trial['family'],
                     **{f'{level}_{key}': trial[level][key] for level in ('token', 'entity') for key in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn')},
                     'fit_seconds': trial['fit_seconds'], 'dev_prediction_seconds': trial['dev_prediction_seconds'],
                     'latency_median_ms': trial['latency_median_ms'], 'latency_p95_ms': trial['latency_p95_ms'],
                     'illegal_bio_rate_before_repair': trial['rates']['illegal_bio_transition_rate_before_repair'],
                     'vocabulary_characters': trial['vocabulary']['vocabulary_characters'] if trial['vocabulary'] else '',
                     'dev_unk_rate': trial['vocabulary']['dev_unk_rate'] if trial['vocabulary'] else ''})
    atomic_write(root / 'reports/traditional-dev-grid.csv', csv_bytes(flat))
    per_type = []
    for family, trial in summary['selected'].items():
        metrics = json.loads((root / summary['run_directory'] / trial['name'] / 'metrics.json').read_text(encoding='utf-8'))
        for label, values in metrics['per_type_entity'].items():
            per_type.append({'family': family, 'configuration': trial['name'], 'type': label, **values})
    atomic_write(root / 'reports/traditional-dev-per-type.csv', csv_bytes(per_type))
    def percent(value):
        return f'{value * 100:.2f}'
    lines = ['# 传统模型 Dev 实验', '',
             '仅使用固定有效 Train 9,538 条训练、官方 Dev 1,343 条选参。正式 Test 未预测或评分，尚不能报告最终泛化性能。', '',
             f'实验指纹：`{summary["context_sha256"]}`。按 Dev 实体 micro F1 选择，同分比较训练耗时，再按配置顺序。', '',
             '| 方法 | Token P (%) | Token R (%) | Token F1 (%) | 实体 P (%) | 实体 R (%) | 实体 F1 (%) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for family, trial in summary['selected'].items():
        values = [percent(trial[level][key]) for level in ('token', 'entity') for key in ('precision', 'recall', 'f1')]
        lines.append('| ' + ' | '.join([family, *values]) + ' |')
    lines += ['', '## 选定配置', '']
    for family, trial in summary['selected'].items():
        lines.append(f'- {family}: `{trial["parameters"]}`；训练 {trial["fit_seconds"]:.3f}s，单句推理中位数/P95 为 {trial["latency_median_ms"]:.3f}/{trial["latency_p95_ms"]:.3f}ms。')
    lines += ['', f'Dev 最佳传统方案：`{summary["best_traditional_by_dev"]}`。', '',
              '完整 22 组结果（词典 1、HMM 18、CRF 3）见 `reports/traditional-dev-grid.csv`；每类结果见 `reports/traditional-dev-per-type.csv`。', '',
              'HMM 9 组参数各比较普通与 BIO 约束解码，配对共享同一模型参数和训练耗时；主排名只使用约束解码。',
              '词表仅来自有效 Train，始终保留一个 UNK 观测；低频及未见字符映射到 UNK。所有概率使用加性平滑，不使用 EOS 概率。',
              'HMM 约束屏蔽非法路径，不重新归一化概率；普通解码仍使用统一 BIO 修复后的主指标，修复前非法率另外列出。',
              'CRF 使用字符及前后两位上下文、局部二元字符、数字/字母/标点及 Unicode 类别特征，CPU L-BFGS 最多 100 次迭代，未添加额外解码约束。', '',
              '训练耗时不含共用的 Python CRF 特征提取；每句推理计时包含特征、解码及预测接口转换，排除文件 IO 和评分；无批量推理。',
              '全部 trial 保留原始 BIO（词典直接保存跨度）、预测 JSONL、规范化诊断和逐句 TP/FP/FN。模型本地保存，不提交 Git。',
              '恢复运行会检查上下文和输出哈希，并对保存的 Dev 预测离线重算，要求指标完全一致；改代码后必须使用新运行目录。', '',
              '没有付费 API 调用。LLM、正式 Test、人工错误复核和 bootstrap 尚待后续阶段完成。', '']
    atomic_write(root / 'reports/traditional-dev.md', '\n'.join(lines).encode('utf-8'))
