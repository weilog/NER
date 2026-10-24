"""Fixed 20-row zero-shot Dev pilot, with offline plan and real-call paths."""
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import dotenv_values

from ner_experiment.data.splitting import read_jsonl, json_bytes, jsonl_bytes, sha
from ner_experiment.evaluation.metrics import evaluate
from ner_experiment.experiments.traditional import atomic_write
from ner_experiment.llm.budget import Ledger, reserve_cost
from ner_experiment.llm.client import DeepSeekTransport, execute_request
from ner_experiment.llm.prompts import PROMPT_VERSION, PARSER_VERSION, request_for, request_key


def prepare(root):
    root = Path(root)
    experiment = json.loads((root / 'configs/experiment_v2.json').read_text(encoding='utf-8'))
    config = json.loads((root / 'configs/llm_pilot.json').read_text(encoding='utf-8'))
    if config['shot_count'] != 0 or config['concurrency'] != 1 or config['max_retries'] > 2:
        raise ValueError('The fixed pilot is zero-shot, serial, at most two retries')
    if config['thinking'] != {'type': 'disabled'} or config['model'] != 'deepseek-flash':
        raise ValueError('Unexpected model or thinking mode')
    pricing = json.loads((root / config['pricing_file']).read_text(encoding='utf-8'))
    manifest = json.loads((root / experiment['dataset']['split_manifest']).read_text(encoding='utf-8'))
    path = root / experiment['llm']['pilot_file']
    if sha(path.read_bytes()) != manifest['output_sha256']['dev_pilot.jsonl']:
        raise ValueError('Pilot data differs from frozen manifest')
    rows = read_jsonl(path)
    if len(rows) != 20 or [row['id'] for row in rows] != manifest['dev_pilot_ids']:
        raise ValueError('Pilot IDs differ from frozen 20 Dev samples')
    if any(not row['id'].startswith('dev:') or row['bio_compatible'] is not True for row in rows):
        raise ValueError('Pilot is not compatible labeled Dev data')
    dotenv = dotenv_values(root / '.env') if (root / '.env').exists() else {}
    credential = os.getenv('DEEPSEEK_API_KEY') or dotenv.get('DEEPSEEK_API_KEY') or ''
    endpoint = (os.getenv('DEEPSEEK_BASE_URL') or dotenv.get('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com').rstrip('/')
    requests = [{'id': row['id'], 'request': request_for(row['text'], config)} for row in rows]
    # No API credential, target annotations, or backend headers in the run context.
    sources = list((root / 'src/ner_experiment/llm').glob('*.py')) + [root / 'src/ner_experiment/evaluation/metrics.py']
    context = {'stage': config['stage'], 'config': config, 'pricing_snapshot': pricing,
               'endpoint': endpoint, 'seed': manifest['seed'], 'ids': manifest['dev_pilot_ids'],
               'pilot_sha256': sha(path.read_bytes()),
               'prompt_version': PROMPT_VERSION, 'parser_version': PARSER_VERSION,
               'source_sha256': {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(sources)},
               'test_accessed': False}
    return config, pricing, rows, requests, context, credential


def run_pilot(root, dry_run=False):
    root = Path(root)
    config, pricing, rows, requests, context, credential = prepare(root)
    directory = root / config['run_directory']
    directory.mkdir(parents=True, exist_ok=True)
    context_path = directory / 'context.json'
    if context_path.exists() and context_path.read_bytes() != json_bytes(context):
        raise ValueError('Pilot context changed; preserve old run and choose another directory')
    atomic_write(context_path, json_bytes(context))
    atomic_write(directory / 'requests.jsonl', jsonl_bytes(requests))
    readiness = {'stage': config['stage'], 'samples': len(rows), 'shot_count': 0,
                 'dry_run': dry_run, 'credential_configured': bool(credential.strip()),
                 'api_calls_this_invocation': 0, 'pilot_per_call_reservation_upper_cny': max(reserve_cost(item['request'], pricing) for item in requests),
                 'pilot_without_retry_reservation_sum_cny': sum(reserve_cost(item['request'], pricing) for item in requests),
                 'pilot_budget_cny': config['pilot_budget_cny'], 'total_budget_cny': config['budget_cny'],
                 'pricing_source': pricing['source'], 'pricing_verified_date': pricing['verified_date_shanghai'],
                 'test_accessed': False}
    atomic_write(directory / 'readiness.json', json_bytes(readiness))
    if dry_run:
        print(json.dumps(readiness, ensure_ascii=False, indent=2))
        return readiness
    if not credential.strip():
        raise ValueError('DEEPSEEK_API_KEY is not configured in environment or local .env')
    if datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat() != pricing['verified_date_shanghai']:
        raise ValueError('Reverify official pricing and save today\'s snapshot before paid calls')
    lock_path = directory / 'run.lock'
    # Exclusive file prevents overlapping exports; a stale lock after a process
    # crash must be removed only after confirming the old process has ended.
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode('ascii'))
    os.close(descriptor)
    ledger, transport = None, None
    try:
        ledger = Ledger(root / config['ledger_path'], config['budget_cny'])
        initial_attempts = ledger.summary()['attempt_count']
        transport = DeepSeekTransport(credential, context['endpoint'], config['timeout_seconds'])
        records, predictions = [], []
        stop_reason, consecutive_failures = None, 0
        for i, item in enumerate(requests):
            if stop_reason:
                prediction = {'id': item['id'], 'status': 'budget_stopped' if stop_reason == 'budget' else 'error',
                              'entities': [], 'error_code': 'not_attempted_after_' + stop_reason}
                predictions.append(prediction)
                continue
            started = time.perf_counter()
            record, cached = execute_request(item['id'], item['request'], context['endpoint'], ledger,
                 transport, pricing, config['stage'], config['pilot_budget_cny'],
                 max_retries=config['max_retries'], backoff=config['retry_backoff_seconds'])
            records.append(record)
            prediction = {**record['prediction'], 'request_key': record['request_key'],
                          'cache_hit': cached, 'elapsed_seconds_this_invocation': time.perf_counter() - started,
                          'network_seconds': record.get('network_seconds', 0)}
            predictions.append(prediction)
            consecutive_failures = consecutive_failures + 1 if prediction['status'] != 'ok' else 0
            if prediction['status'] == 'budget_stopped':
                stop_reason = 'budget'
            elif record['fatal']:
                stop_reason = 'fatal_api_or_accounting_issue'
            elif consecutive_failures >= 3:
                stop_reason = 'three_consecutive_failures'
            atomic_write(directory / 'records.jsonl', jsonl_bytes(records))
            atomic_write(directory / 'predictions.partial.jsonl', jsonl_bytes(predictions))
            print(f'{i+1}/20 {item["id"]} {prediction["status"]} cache={cached} accounted_total_cny={ledger.spent():.6f}', flush=True)
        atomic_write(directory / 'predictions.jsonl', jsonl_bytes(predictions))
        # The shared evaluator receives only saved predictions, never a separate LLM metric path.
        metrics = evaluate(rows, read_jsonl(directory / 'predictions.jsonl'))
        normalized = metrics.pop('normalized_predictions')
        atomic_write(directory / 'metrics.json', json_bytes(metrics))
        atomic_write(directory / 'normalized.jsonl', jsonl_bytes(normalized))
        models = sorted({r['response_model'] for r in records if isinstance(r.get('response_model'), str)})
        times = [r['network_seconds'] for r in records if 'raw_response' in r]
        tokens = Counter()
        for record in records:
            for attempt in record.get('attempts', []):
                detail = json.loads(attempt['cost_detail_json'] or '{}')
                tokens.update({key: detail.get(key, 0) for key in ('prompt_tokens', 'completion_tokens', 'cache_hit_tokens', 'cache_miss_tokens')})
        accounting = ledger.summary()
        ready = (not stop_reason and len(records) == 20 and all(p['status'] == 'ok' for p in predictions)
                 and not metrics['diagnostics'].get('invalid_entities', 0)
                 and not metrics['diagnostics'].get('overlap_entities_removed', 0)
                 and accounting['unknown_or_pending_attempts'] == 0 and len(models) == 1)
        summary = {'stage': config['stage'], 'context_sha256': sha(json_bytes(context)),
                   'completed_date_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                   'samples': 20, 'attempted_samples': len(records),
                   'api_calls_this_invocation': accounting['attempt_count'] - initial_attempts,
                   'cached_samples_this_invocation': sum(p.get('cache_hit', False) for p in predictions),
                   'status_counts': dict(Counter(p['status'] for p in predictions)),
                   'requested_model': config['model'], 'reported_response_models': models,
                   'documented_model_version': pricing['documented_model_version'],
                   'usage': dict(tokens), 'accounting': accounting,
                   'latency_median_seconds_including_retries_excluding_backoff': float(np.median(times)) if times else None,
                   'latency_p95_seconds_including_retries_excluding_backoff': float(np.percentile(times, 95)) if times else None,
                   'token': metrics['token'], 'entity': metrics['entity'], 'rates': metrics['rates'],
                   'runtime_freeze_ready': ready, 'stop_reason': stop_reason,
                   'test_accessed': False,
                   'output_sha256': {name: sha((directory / name).read_bytes()) for name in ('predictions.jsonl', 'records.jsonl', 'metrics.json', 'normalized.jsonl')}}
        atomic_write(directory / 'summary.json', json_bytes(summary))
        if ready:
            frozen = {'status': 'pilot_format_and_position_checks_passed; not_a_test_authorization',
                      'pilot_context_sha256': summary['context_sha256'],
                      'requested_model': config['model'], 'reported_response_models': models,
                      'documented_model_version': pricing['documented_model_version'],
                      'prompt_version': PROMPT_VERSION, 'parser_version': PARSER_VERSION,
                      'parameters': {key: config[key] for key in ('temperature', 'thinking', 'response_format', 'max_tokens', 'timeout_seconds', 'concurrency', 'max_retries')},
                      'pricing_snapshot': pricing,
                      'pilot_summary_sha256': sha((directory / 'summary.json').read_bytes())}
            atomic_write(root / 'configs/llm_runtime_frozen.json', json_bytes(frozen))
        write_report(root, summary, config, pricing)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if transport is not None:
            transport.close()
        if ledger is not None:
            ledger.close()
        lock_path.unlink(missing_ok=True)


def write_report(root, summary, config, pricing):
    lines = ['# DeepSeek 固定 20 条 Dev 试跑', '',
             '仅为格式、位置、费用和延迟试跑；20 条 Dev 分数不用于最终方法排名。没有读取或评分项目 Test。', '',
             f'请求模型：`{summary["requested_model"]}`；API 返回模型：`{summary["reported_response_models"]}`。',
             f'官方价格页记录版本：`{summary["documented_model_version"]}`；该版本来自文档，不伪称 API 返回了具体版本。',
             f'实际运行日期：{summary["completed_date_shanghai"]}。', '',
             f'固定参数：关闭 Thinking；temperature=0；JSON 对象；max_tokens={config["max_tokens"]}；超时 {config["timeout_seconds"]} 秒；串行；网络/限流/服务暂时错误最多重试两次；SDK 自带重试关闭。', '',
             f'状态统计：`{summary["status_counts"]}`；运行参数冻结检查通过：{summary["runtime_freeze_ready"]}。',
             f'原始预测无效实体率：{summary["rates"]["invalid_entity_rate"]:.4%}；失败率：{summary["rates"]["failure_rate"]:.4%}。', '',
             f'Usage：`{summary["usage"]}`。',
             f'保守累计记账（元）：{summary["accounting"]["accounted_cny"]:.6f}；包含未知费用预留，不等同于供应商账单。',
             f'全项目预算：{config["budget_cny"]} 元；本次试跑阶段上限：{config["pilot_budget_cny"]} 元。',
             f'网络耗时（含重试、不含退避等待）中位数/P95：{summary["latency_median_seconds_including_retries_excluding_backoff"]} / {summary["latency_p95_seconds_including_retries_excluding_backoff"]} 秒。', '',
             f'价格来源：[DeepSeek 官方人民币价格]({pricing["source"]})；核验日期 {pricing["verified_date_shanghai"]}。',
             '按请求开始和结束时段的较高价格记账；周末用空闲价，工作日法定假日若未额外核验则保守按高峰价。缓存用量缺失时按未命中价。', '',
             '| 层级 | P (%) | R (%) | F1 (%) |', '|---|---:|---:|---:|']
    for name in ('token', 'entity'):
        lines.append('| ' + ' | '.join([name, *[f'{summary[name][key]*100:.2f}' for key in ('precision', 'recall', 'f1')]]) + ' |')
    lines += ['', '所有原始请求、响应、解析结果、usage、模型标识、错误及哈希仅存本地 runs/llm；密钥仅从环境变量或本地 .env 读取。',
              '无自动位置纠正，无隐藏格式修复调用；失败按空预测，共用主评估器。冻结检查未通过时先审核诊断，不自动扩大调用或改变固定 Test。', '']
    atomic_write(root / 'reports/llm-pilot.md', '\n'.join(lines).encode('utf-8'))
