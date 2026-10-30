"""MiMo tests use fake transports/time only, never paid API calls."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import socket
import sys
import threading
import urllib.error

import pytest
from ner_experiment.common import ROOT, file_hash, load_json, save_json
from ner_experiment.models.glm import InterfaceStop
from ner_experiment.models.llm import BudgetStop
from ner_experiment.models.mimo import MiMoClient, RateLimiter, settings


ROW = {'id': 'x', 'text': '张三', 'tokens': list('张三'), 'entities': []}


class NoWait:
    def __init__(self):
        self.calls = 0

    def acquire(self, stop):
        if stop.is_set():
            raise InterfaceStop('Stopped')
        self.calls += 1
        return 0


class Clock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def messages(row, examples):
    return [{'role': 'user', 'content': row['text']}]


def response(content='{"entities":[]}', usage=None, finish='stop'):
    return {'model': 'mimo-v2.6-flash', 'choices': [{'finish_reason': finish,
            'message': {'content': content, 'reasoning_content': ''}}],
            'usage': usage if usage is not None else {'prompt_tokens': 100, 'completion_tokens': 20,
                        'prompt_tokens_details': {'cached_tokens': 40}}}


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('MIMO_API_KEY', 'dummy-mimo-key-for-tests')
    monkeypatch.setenv('MIMO_MODEL', 'mimo-v2.6-flash')
    monkeypatch.setenv('MIMO_BASE_URL', 'https://api.xiaomimimo.com/v1')
    return load_json(ROOT / 'configs/mimo_flash.json')


def client(folder, config, transport, budget=20, limiter=None):
    return MiMoClient(folder, config, messages, budget, transport=transport,
                       limiter=limiter or NoWait(), sleep=lambda _: None)


def test_global_pacing_and_rolling_minute_limit_with_fake_clock():
    clock = Clock()
    gate = RateLimiter(90, clock=clock.time, sleeper=clock.sleep)
    stop = threading.Event()
    stamps = []
    for _ in range(181):
        gate.acquire(stop)
        stamps.append(clock.now)
    assert all(b - a >= 60 / 90 - 1e-8 for a, b in zip(stamps, stamps[1:]))
    assert all(stamps[i + 90] - stamps[i] >= 60 - 1e-8 for i in range(91))
    stop.set()
    with pytest.raises(InterfaceStop):
        gate.acquire(stop)


def test_limiter_seeds_previous_process_requests():
    clock = Clock()
    now = datetime.now(timezone.utc)
    stamps = [(now - timedelta(seconds=i / 90)).isoformat() for i in range(90)]
    gate = RateLimiter(90, clock=clock.time, sleeper=clock.sleep, recent_starts=stamps)
    gate.acquire(threading.Event())
    assert clock.now > 58  # Restart does not reset a full minute of account requests.


def test_cache_restart_does_not_repeat_request_or_consume_rpm_slot(tmp_path, config):
    calls = []
    def fake(payload):
        calls.append(payload)
        return response()
    gate = NoWait()
    c = client(tmp_path, config, fake, limiter=gate)
    first = c.predict(ROW, [])
    restored = client(tmp_path, config, fake, limiter=gate)
    assert restored.predict(ROW, []) == first and len(calls) == 1 and gate.calls == 1
    assert calls[0]['thinking'] == {'type': 'disabled'}
    assert calls[0]['max_completion_tokens'] == 2048 and 'max_tokens' not in calls[0]
    assert calls[0]['temperature'] == 0 and calls[0]['response_format']['type'] == 'json_object'
    assert c.total() == pytest.approx((60 * 1 + 40 * .02 + 20 * 2) / 1e6)


def test_official_http_auth_header_and_endpoint(tmp_path, config, monkeypatch):
    seen = []
    def open_fake(request, timeout):
        seen.append(request)
        return io.BytesIO(json.dumps(response()).encode())
    monkeypatch.setattr('ner_experiment.models.mimo.urllib.request.urlopen', open_fake)
    c = client(tmp_path, config, lambda _: response())
    assert c.http({'model': 'mimo-v2.6-flash'})['model'] == 'mimo-v2.6-flash'
    assert seen[0].full_url == 'https://api.xiaomimimo.com/v1/chat/completions'
    assert seen[0].get_header('Api-key') == 'dummy-mimo-key-for-tests'


def test_budget_checked_before_transport(tmp_path, config):
    def forbidden(payload):
        raise AssertionError('Budget must stop before API')
    with pytest.raises(BudgetStop):
        client(tmp_path, config, forbidden, budget=0).predict(ROW, [])


@pytest.mark.parametrize('code', [400, 401, 402, 403])
def test_interface_or_balance_error_stops_after_one_attempt(tmp_path, config, code):
    calls = []
    def fail(payload):
        calls.append(1)
        raise urllib.error.HTTPError('url', code, 'fake', None, io.BytesIO(b'{"error":{"code":"bad"}}'))
    c = client(tmp_path, config, fail)
    with pytest.raises(InterfaceStop):
        c.predict(ROW, [])
    with pytest.raises(InterfaceStop):
        client(tmp_path, config, fail).predict(ROW, [])
    assert len(calls) == 1


def test_documented_content_filter_is_failed_row_not_batch_stop(tmp_path, config):
    calls = []
    def fail(payload):
        calls.append(1)
        body = b'{"error":{"message":"dummy-mimo-key-for-tests filtered"}}'
        raise urllib.error.HTTPError('url', 421, 'fake', None, io.BytesIO(body))
    c = client(tmp_path, config, fail)
    result = c.predict(ROW, [])
    assert result['status'] == 'api_error' and not result.get('fatal_error')
    assert len(calls) == 1 and c.predict(ROW, []) == result
    assert 'dummy-mimo-key-for-tests' not in next(c.cache.glob('*.json')).read_text(encoding='utf-8')


def test_network_retries_share_rate_limiter_and_stop_after_eight(tmp_path, config):
    gate = NoWait()
    def fail(payload):
        raise TimeoutError()
    c = client(tmp_path, config, fail, limiter=gate)
    for i in range(2):
        assert c.predict({**ROW, 'id': str(i), 'text': str(i)}, [])['status'] == 'api_error'
    with pytest.raises(InterfaceStop):
        c.predict({**ROW, 'id': '2', 'text': '2'}, [])
    assert len(c.ledger['attempts']) == 8 and gate.calls == 8
    assert c.total() > 0 and c.stop_event.is_set()


def test_dns_failure_releases_only_proven_unsent_reservation(tmp_path, config):
    def fail(payload):
        raise urllib.error.URLError(socket.gaierror(11001, 'mock DNS'))
    c = client(tmp_path, config, fail)
    with pytest.raises(InterfaceStop):
        c.predict(ROW, [])
    assert c.total() == 0 and len(c.ledger['attempts']) == 1
    assert c.ledger['attempts'][0]['verified_not_sent']


@pytest.mark.parametrize('content,finish', [('oops', 'stop'), ('{"entities":[]}', 'length')])
def test_format_or_truncation_failure_not_repaired(tmp_path, config, content, finish):
    calls = []
    def fake(payload):
        calls.append(1)
        return response(content, finish=finish)
    c = client(tmp_path, config, fake)
    assert c.predict(ROW, [])['status'] == 'format_error' and len(calls) == 1


def test_missing_usage_preserves_reservation_and_raw_response(tmp_path, config):
    c = client(tmp_path, config, lambda _: response(usage={}))
    with pytest.raises(InterfaceStop):
        c.predict(ROW, [])
    assert c.total() > 0
    record = load_json(next(c.cache.glob('*.json')))
    assert record['raw_response']['usage'] == {}


def test_corrupt_cache_cannot_trigger_new_paid_request(tmp_path, config):
    calls = []
    def fake(payload):
        calls.append(1)
        return response()
    c = client(tmp_path, config, fake)
    result = c.predict(ROW, [])
    path = c.cache / (result['cache_key'] + '.json')
    record = load_json(path)
    record['raw_response']['model'] = 'changed'
    save_json(path, record)
    with pytest.raises(ValueError):
        c.predict(ROW, [])
    assert len(calls) == 1


def test_four_concurrent_responses_have_one_durable_ledger(tmp_path, config, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import run_glm_test
    barrier = threading.Barrier(4, timeout=5)
    def fake(payload):
        barrier.wait()
        return response()
    c = client(tmp_path, config, fake)
    rows = [{**ROW, 'id': str(i), 'text': str(i)} for i in range(4)]
    results = run_glm_test.predict_rows(c, rows, [], 4)
    assert [r['id'] for r in results] == [r['id'] for r in rows]
    assert len(load_json(c.ledger_path)['attempts']) == 4
    assert len(list(c.cache.glob('*.json'))) == 4


def test_official_endpoint_only(config, monkeypatch, tmp_path):
    monkeypatch.setenv('MIMO_BASE_URL', 'https://other.invalid/v1')
    with pytest.raises(ValueError):
        settings(config, tmp_path / 'none.env')


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('mimo_test_runner', ROOT / 'scripts/run_mimo_test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_check_does_not_construct_client(runner, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Offline check constructed API client')
    monkeypatch.setattr(runner, 'MiMoClient', forbidden)
    monkeypatch.setattr(sys, 'argv', ['run_mimo_test.py', '--check-config'])
    runner.main()


def test_freeze_rejects_prompt_or_rate_changes(tmp_path, runner):
    original = {'system': 'fixed', 'rpm': 90}
    runner.freeze_once(tmp_path, original)
    before = file_hash(tmp_path / 'frozen_protocol.json')
    with pytest.raises(RuntimeError):
        runner.freeze_once(tmp_path, {**original, 'rpm': 80})
    assert file_hash(tmp_path / 'frozen_protocol.json') == before
