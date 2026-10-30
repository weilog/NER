"""Save a private, verified checkpoint of the paused GLM run. No network or API calls."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(content):
    return hashlib.sha256(content).hexdigest()


def main():
    base = ROOT / 'runs/glm_flashx_test_v1'
    if (ROOT / 'runs/llm/process.lock').exists():
        raise RuntimeError('Only checkpoint after the LLM process stopped; do not snapshot an active ledger')
    attempts = load(base / 'ledger.json')['attempts']
    if any(a['status'] == 'reserved' for a in attempts):
        raise RuntimeError('Unresolved in-flight reservations; audit before checkpointing')
    gold = [json.loads(line) for line in (ROOT / 'data/processed/cluener2020_project/test.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    gold_ids = [r['id'] for r in gold]
    by_shot = defaultdict(dict)
    for path in (base / 'cache').glob('*.json'):
        record = load(path)
        prediction = record['prediction']
        shot = len(record['example_ids'])
        if prediction['id'] in by_shot[shot]:
            raise RuntimeError('Duplicate cached sample ID; checkpoint needs manual review')
        by_shot[shot][prediction['id']] = prediction
    groups = []
    for shot in [0, 2, 4, 8]:
        predictions = by_shot[shot]
        if not set(predictions) <= set(gold_ids):
            raise RuntimeError('Cache contains non-Test ID')
        status = {'shots': shot, 'saved': len(predictions), 'expected': len(gold_ids),
                  'remaining_ids': [i for i in gold_ids if i not in predictions],
                  'statuses': dict(Counter(p['status'] for p in predictions.values()))}
        metrics = base / f'shot_{shot}/metrics.json'
        if metrics.exists():
            result = load(metrics)
            status.update(token=result['token'], entity=result['entity'])
        elif predictions:
            export = base / 'partial_exports' / f'shot_{shot}.predictions.partial.jsonl'
            export.parent.mkdir(exist_ok=True)
            export.write_text(''.join(json.dumps(predictions[i], ensure_ascii=False) + '\n'
                                      for i in gold_ids if i in predictions), encoding='utf-8')
            save(export.with_suffix('.metadata.json'), {'partial': True, 'not_a_full_test_result': True,
                 'saved_samples': len(predictions), 'expected_samples': len(gold_ids), 'shots': shot})
            status['partial_export'] = export.relative_to(ROOT).as_posix()
        groups.append(status)
    created = datetime.now(timezone(timedelta(hours=8)))
    status = {'saved_at': created.isoformat(), 'state': 'paused',
              'reason': 'BigModel HTTP429 business code1113: insufficient balance or no resource package',
              'groups': groups, 'saved_total': sum(g['saved'] for g in groups),
              'returned_usage_estimate_cny': sum(a['charged_upper_cny'] for a in attempts if a['status'] == 'response'),
              'unknown_failed_reservations_cny': sum(a['charged_upper_cny'] for a in attempts if a['status'] != 'response'),
              'gross_ledger_upper_cny': sum(a['charged_upper_cny'] for a in attempts),
              'local_gross_budget_cny': 40, 'project_budget_cny': 100,
              'new_model_planned': 'mimo-v2.6-flash', 'new_model_api_calls': 0,
              'resume_notes': ['Keep original successes/failures and unknown fee reservations.',
                  'Top up/verify BigModel balance first; current cached fatal1113 records require an explicit audited resume decision.',
                  'Do not delete caches, reset ledger, or claim partial 8-shot as full Test.',
                  '4-shot includes network outage failures; performance is not a pure model-capability comparison.']}
    save(base / 'paused_status.json', status)
    tag = 'glm_flashx_paused_' + created.strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'runs/checkpoints' / tag
    out.mkdir(parents=True, exist_ok=False)
    files = {p.relative_to(ROOT).as_posix(): p for p in base.rglob('*') if p.is_file()}
    for folder in ['src', 'scripts', 'tests', 'configs', 'data/splits']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}:
                files[path.relative_to(ROOT).as_posix()] = path
    for name in ['pyproject.toml', 'requirements-lock.txt', '.python-version', '.env.example',
                 'docs/run_glm_flashx.md', 'data/processed/cluener2020_project/train.jsonl',
                 'data/processed/cluener2020_project/dev.jsonl', 'data/processed/cluener2020_project/test.jsonl']:
        files[name] = ROOT / name
    secrets = [v for k, v in os.environ.items() if k.endswith('API_KEY') and len(v) >= 12]
    env = ROOT / '.env'
    if env.exists():
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            key, sep, value = line.partition('=')
            value = value.strip().strip('"\'')
            if sep and key.strip().endswith('API_KEY') and len(value) >= 12:
                secrets.append(value)
    hashes = {}
    archive_path = out / (tag + '.zip')
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            if path.name == '.env':
                raise RuntimeError('Refuse to include actual .env')
            content = path.read_bytes()
            if any(secret.encode('utf-8') in content for secret in secrets):
                raise RuntimeError('Credential bytes found in snapshot candidate; do not share archive')
            hashes[name] = sha(content)
            archive.writestr(name, content)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert all(sha(archive.read(name)) == digest for name, digest in hashes.items())
    save(out / 'files_sha256.json', hashes)
    save(out / 'status.json', status)
    save(out / 'verification.json', {'verified_at': datetime.now(timezone.utc).isoformat(),
         'archive': archive_path.name, 'archive_sha256': sha(archive_path.read_bytes()),
         'files': len(hashes), 'zip_crc_passed': True, 'all_file_hashes_passed': True,
         'actual_env_excluded': True, 'known_credentials_excluded': True, 'api_calls': 0,
         'scope': 'Private local checkpoint includes corpus and raw predictions; not for public GitHub'})
    summary = ['# GLM 暂停检查点', '', f'保存时间：{created.isoformat()}', '',
        '| Shot | 已保存 / Test总量 | 状态 |', '|---|---:|---|']
    for group in groups:
        summary.append(f'| {group["shots"]} | {group["saved"]} / {group["expected"]} | ' +
                       ('完整评分已保存' if 'entity' in group else '部分预测，仅用于恢复') + ' |')
    summary += ['', f'已返回usage估算约 ¥{status["returned_usage_estimate_cny"]:.4f}；含未知预留的保守账本 ¥{status["gross_ledger_upper_cny"]:.4f}。不是平台实扣账单。',
        '', '停止原因：智谱余额/资源包不足（1113），没有自动充值。',
        '8-shot部分预测按原Test顺序另行导出，但不冒充完整测试成绩。余额恢复后，原余额不足fatal缓存也需要明确审核恢复规则，不能直接删缓存。',
        '4-shot保留断网导致的失败，分析时须区分网络故障与NER能力。',
        '', 'ZIP包含当前运行目录、源码/配置/划分、固定处理数据；逐文件SHA256和ZIP校验已通过。不含真实.env或已知API Key。仅组内本地保存，不上传公共GitHub。',
        '', '下一模型：mimo-v2.6-flash（小米官方按量API）；本次保存没有调用新模型。']
    (out / 'README.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print(json.dumps({'checkpoint_dir': str(out), 'archive': str(archive_path),
         'saved_total': status['saved_total'], 'file_count': len(hashes),
         'archive_bytes': archive_path.stat().st_size, 'verified': True, 'api_calls': 0}, ensure_ascii=True))


if __name__ == '__main__':
    main()
