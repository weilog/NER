"""GLM billing/cache/failure tests use fake transports only; never real API calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import urllib.error

import pytest
from ner_experiment.common import ROOT, load_json, save_json
from ner_experiment.evaluation.metrics import evaluate
from ner_experiment.models.glm import GLMClient, InterfaceStop, parse_glm_response, settings, usage_cost
from ner_experiment.models.llm import BudgetStop

ROW = {'id': 'x', 'text': '张三', 'tokens': list('张三'),
       'entities': [{'start': 0, 'end': 2, 'type': 'name'}]}


def messages(row, examples):
    return [{'role': 'user', 'content': row['text']}]


def response(content='{"entities":[]}', usage=None, finish='stop'):
    return {'model': 'glm-5.3-flashx', 'choices': [{'finish_reason': finish,
            'message': {'content': content, 'reasoning_content': 'mock reasoning'}}],
            'usage': usage if usage is not None else {'prompt_tokens': 100, 'completion_tokens': 20,
                                                     'prompt_tokens_details': {'cached_tokens': 40}}}


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('GLM_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4/')
    monkeypatch.setenv('GLM_MODEL', 'glm-5.3-flashx')
    return load_json(ROOT / 'configs/glm_flashx.json')


def client(tmp_path, config, transport, budget=1):
    return GLMClient(tmp_path, config, messages, budget, transport=transport, sleep=lambda _: None)


def test_usage_includes_cache_discount_without_reasoning_double_count(config):
    usage = {'prompt_tokens': 100, 'completion_tokens': 20,
             'prompt_tokens_details': {'cached_tokens': 40},
             'completion_tokens_details': {'reasoning_tokens': 10}}
    assert usage_cost(usage, config['pricing']) == pytest.approx((60 * 2 + 40 * .57 + 20 * 7) / 1e6)
    for invalid in [{}, {'prompt_tokens': 1, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 3}}]:
        with pytest.raises(ValueError):
            usage_cost(invalid, config['pricing'])


def test_cache_restart_no_second_request_and_corruption_stops(tmp_path, config):
    calls = []
    def fake(payload):
        calls.append(payload)
        return response()
    c = client(tmp_path, config, fake)
    first = c.predict(ROW, [])
    assert first['status'] == 'ok'
    assert client(tmp_path, config, fake).predict(ROW, []) == first
    assert len(calls) == 1
    assert calls[0]['model'] == 'glm-5.3-flashx'
    assert calls[0]['thinking']['type'] == 'enabled'
    assert calls[0]['reasoning_effort'] == 'low'
    path = c.cache / (first['cache_key'] + '.json')
    record = load_json(path)
    record['raw_response']['usage']['completion_tokens'] = 999
    save_json(path, record)
    with pytest.raises(ValueError):
        c.predict(ROW, [])
    assert len(calls) == 1


def test_offline_replay_four_groups_without_a_client(tmp_path, config, runner):
    from ner_experiment.common import object_hash
    protocol = {'test_ids': ['x']}
    runner.freeze_once(tmp_path / 'run', protocol)
    c = client(tmp_path / 'client', config, lambda _: response())
    for shots in [0, 2, 4, 8]:
        runner.run_group(c, [ROW], [], tmp_path / f'run/shot_{shots}', object_hash(protocol))
    runner.replay(tmp_path / 'run', [ROW])
    with pytest.raises(RuntimeError, match='ID sequence'):
        runner.replay(tmp_path / 'run', [{**ROW, 'id': 'other'}])


def test_budget_checked_before_call(tmp_path, config):
    def forbidden(payload):
        raise AssertionError('API call attempted')
    with pytest.raises(BudgetStop):
        client(tmp_path, config, forbidden, budget=0).predict(ROW, [])


def test_network_three_attempts_and_failure_cached(tmp_path, config):
    calls = []
    def fail(payload):
        calls.append(1)
        raise urllib.error.URLError('mock failure')
    c = client(tmp_path, config, fail)
    result = c.predict(ROW, [])
    assert result['status'] == 'api_error' and len(calls) == 3
    assert client(tmp_path, config, fail).predict(ROW, []) == result
    assert len(calls) == 3 and c.total() > 0


def test_retry_limit_survives_budget_interruption(tmp_path, config):
    calls = []
    def fail(payload):
        calls.append(1)
        raise TimeoutError()
    c = client(tmp_path, config, fail)
    payload = {**c.parameters, 'messages': messages(ROW, [])}
    c.budget = c.estimate_upper(payload) * 1.1
    with pytest.raises(BudgetStop):
        c.predict(ROW, [])
    assert len(calls) == 1
    result = client(tmp_path, config, fail).predict(ROW, [])
    assert result['status'] == 'api_error' and result['attempts'] == 3 and len(calls) == 3


@pytest.mark.parametrize('code', [400, 401, 403, 402])
def test_fatal_http_stops_batch_and_cached_failure_cannot_be_retried(tmp_path, config, code):
    calls = []
    def fail(payload):
        calls.append(1)
        raise urllib.error.HTTPError('url', code, 'fake', None, None)
    c = client(tmp_path, config, fail)
    with pytest.raises(InterfaceStop):
        c.predict(ROW, [])
    with pytest.raises(InterfaceStop):
        client(tmp_path, config, fail).predict(ROW, [])
    assert len(calls) == 1


def test_unknown_usage_keeps_reservation_and_stops(tmp_path, config):
    c = client(tmp_path, config, lambda _: response(usage={}))
    with pytest.raises(InterfaceStop):
        c.predict(ROW, [])
    assert c.total() > 0
    assert next(c.cache.glob('*.json')).is_file()


def test_charged_request_missing_cache_requires_audit(tmp_path, config):
    c = client(tmp_path, config, lambda _: response())
    result = c.predict(ROW, [])
    (c.cache / (result['cache_key'] + '.json')).unlink()
    with pytest.raises(InterfaceStop, match='manual audit'):
        c.predict(ROW, [])


def test_bad_json_no_repair_and_invalid_spans_keep_v3_policy(tmp_path, config):
    assert client(tmp_path / 'bad', config, lambda _: response('oops')).predict(ROW, [])['status'] == 'format_error'
    body = '{"entities":[{"text":"张","start":0,"end":1,"type":"name"},{"text":"越界","start":0,"end":99,"type":"name"}]}'
    prediction = client(tmp_path / 'invalid', config, lambda _: response(body)).predict(ROW, [])
    assert prediction['status'] == 'ok' and len(prediction['entities']) == 2
    metrics, _, _ = evaluate([ROW], [prediction])
    assert metrics['entity']['fp'] == 1
    assert metrics['entity_with_invalid_fp']['fp'] == 2
    assert metrics['diagnostics']['invalid_entities'] == 1


def test_truncation_and_missing_text_are_format_failures():
    assert parse_glm_response(response(finish='length')) == ('format_error', [])
    assert parse_glm_response(response('{"entities":[{"start":0,"end":1,"type":"name"}]}')) == ('format_error', [])


def test_unapproved_endpoint_rejected(config, monkeypatch, tmp_path):
    monkeypatch.setenv('GLM_BASE_URL', 'https://other.invalid/v1')
    with pytest.raises(ValueError, match='official'):
        settings(config, tmp_path / 'missing.env')


@pytest.fixture
def runner(monkeypatch):
    scripts = ROOT / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('glm_test_runner', scripts / 'run_glm_test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_context_matches_frozen_reference(runner, config):
    manifest, examples, test = runner.build_context(config)
    assert len(test) == 1209 and manifest['seed'] == 718853259
    assert [r['id'] for r in examples] == config['example_ids']
    wrong = deepcopy(config)
    wrong['example_ids'].reverse()
    with pytest.raises(RuntimeError, match='examples differ'):
        runner.build_context(wrong)


def test_default_preflight_never_constructs_api_client(runner, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Offline preflight constructed a paid client')
    monkeypatch.setattr(runner, 'GLMClient', forbidden)
    monkeypatch.setattr(sys, 'argv', ['run_glm_test.py', '--check-config'])
    runner.main()


def test_freeze_stable_and_changed_protocol_rejected(tmp_path, runner):
    protocol = {'model': 'glm-5.3-flashx'}
    runner.freeze_once(tmp_path / 'run', protocol)
    before = (tmp_path / 'run/frozen_protocol.json').read_bytes()
    runner.freeze_once(tmp_path / 'run', protocol)
    with pytest.raises(RuntimeError, match='changed'):
        runner.freeze_once(tmp_path / 'run', {'model': 'changed'})
    assert (tmp_path / 'run/frozen_protocol.json').read_bytes() == before


def test_group_uses_same_evaluator_and_saved_replay(tmp_path, config, runner):
    calls = []
    def fake(payload):
        calls.append(1)
        return response('{"entities":[{"text":"张三","start":0,"end":2,"type":"name"}]}')
    c = client(tmp_path / 'client', config, fake)
    folder = tmp_path / 'shot_0'
    diag = runner.run_group(c, [ROW], [], folder, 'fixed')
    assert diag['entity']['f1'] == 1 and diag['token']['f1'] == 1
    assert runner.run_group(c, [ROW], [], folder, 'fixed') == diag and len(calls) == 1
    (folder / 'predictions.jsonl').write_text('damaged', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Damaged'):
        runner.run_group(c, [ROW], [], folder, 'fixed')
    assert len(calls) == 1


def test_four_workers_actually_overlap_and_keep_input_order(tmp_path, config, runner):
    import threading
    import time
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=5)
    active = peak = 0
    def fake(payload):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()
        time.sleep(.005)
        with lock:
            active -= 1
        return response()
    c = client(tmp_path, config, fake)
    rows = [{**ROW, 'id': str(i), 'text': f'张三{i}', 'tokens': list(f'张三{i}')} for i in range(4)]
    results = runner.predict_rows(c, rows, [], 4)
    assert peak == 4 and [p['id'] for p in results] == [r['id'] for r in rows]
    assert len(load_json(c.ledger_path)['attempts']) == 4


def test_parallel_budget_stop_drains_current_response_without_overspend(tmp_path, config, runner):
    import time
    calls = []
    def fake(payload):
        calls.append(1)
        time.sleep(.1)
        return response()
    c = client(tmp_path, config, fake)
    rows = [{**ROW, 'id': str(i), 'text': f'张三{i}', 'tokens': list(f'张三{i}')} for i in range(4)]
    payload = {**c.parameters, 'messages': messages(rows[0], [])}
    c.budget = c.estimate_upper(payload) * 1.1
    with pytest.raises((BudgetStop, InterfaceStop)):
        runner.predict_rows(c, rows, [], 4)
    assert len(calls) == 1 and c.total() <= c.budget
    assert len(list(c.cache.glob('*.json'))) == 1
    assert not any(a['status'] == 'reserved' for a in c.ledger['attempts'])


def test_audited_historical_400_preserved_without_retry(tmp_path, config):
    from ner_experiment.common import file_hash
    def fail(payload):
        raise urllib.error.HTTPError('url', 400, 'test', None, None)
    c = client(tmp_path, config, fail)
    with pytest.raises(InterfaceStop):
        c.predict(ROW, [])
    path = next(c.cache.glob('*.json'))
    before = file_hash(path)
    def forbidden(payload):
        raise AssertionError('Historical failure must not be retried')
    restored = GLMClient(tmp_path, config, messages, 1, transport=forbidden,
                         accepted_failure_keys=[path.stem], workers=4)
    assert restored.predict(ROW, [])['status'] == 'api_error'
    assert file_hash(path) == before and len(restored.ledger['attempts']) == 1


def test_provider_refusal_is_scored_failure_and_diagnostics_redact_key(tmp_path, config, monkeypatch):
    import io
    import json
    monkeypatch.setenv('GLM_API_KEY', 'dummy-key-only-for-test')
    calls = []
    def fail(payload):
        calls.append(1)
        body = json.dumps({'error': {'code': '1301', 'message': 'refused dummy-key-only-for-test'}}).encode()
        raise urllib.error.HTTPError('url', 400, 'test', None, io.BytesIO(body))
    c = client(tmp_path, config, fail)
    result = c.predict(ROW, [])
    assert result['status'] == 'api_error' and not result.get('fatal_error')
    assert result['error']['provider_error_code'] == '1301'
    assert 'dummy-key-only-for-test' not in next(c.cache.glob('*.json')).read_text(encoding='utf-8')
    assert len(calls) == 1 and c.predict(ROW, []) == result


@pytest.mark.parametrize('source_name', ['scripts/run_glm_test.py', 'scripts\\run_glm_test.py'])
def test_migration_accepts_only_scheduling_and_preserves_original_freeze(tmp_path, runner, source_name):
    from ner_experiment.common import file_hash
    base = tmp_path / 'run'
    audit = base / 'scheduling_migration_v1'
    old = {'parameters': {'model': 'fixed'}, 'source_sha256': {source_name: 'old'}}
    new = {**old, 'workers': 4, 'source_sha256': {source_name: 'new'}}
    runner.freeze_once(base, old)
    original_hash = file_hash(base / 'frozen_protocol.json')
    save_json(base / 'execution_protocol_v2.json', {'protocol': new})
    cache_path = base / 'cache/key.json'
    save_json(cache_path, {'prediction': {'status': 'api_error', 'error': {'http_status': 400}}})
    cache_name = 'cache\\key.json' if '\\' in source_name else 'cache/key.json'
    save_json(audit / 'before_manifest.json', {'files': {cache_name: file_hash(cache_path)}})
    migration = {'original_freeze_sha256': original_hash,
                 'active_freeze_sha256': file_hash(base / 'execution_protocol_v2.json'),
                 'before_manifest_sha256': file_hash(audit / 'before_manifest.json'),
                 'retained_http400_failure_keys': ['key']}
    save_json(audit / 'migration.json', migration)
    assert runner.freeze_once(base, new)['protocol'] == new
    assert runner.accepted_failures(base) == ['key']
    assert file_hash(base / 'frozen_protocol.json') == original_hash
    before_cache = cache_path.read_bytes()
    save_json(cache_path, {'changed': True})
    with pytest.raises(RuntimeError, match='serial cache changed'):
        runner.migrated_freeze(base)
    cache_path.write_bytes(before_cache)
    changed = {**new, 'parameters': {'model': 'changed'}}
    save_json(base / 'execution_protocol_v2.json', {'protocol': changed})
    save_json(audit / 'migration.json', {**migration, 'active_freeze_sha256': file_hash(base / 'execution_protocol_v2.json')})
    with pytest.raises(RuntimeError, match='model/data/prompt/budget'):
        runner.freeze_once(base, changed)
