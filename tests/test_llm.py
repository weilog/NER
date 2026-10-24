from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ner_experiment.llm.budget import BudgetStopped, Ledger, reserve_cost, usage_cost
from ner_experiment.llm.client import TransportError, execute_request
from ner_experiment.llm.prompts import messages_for, parse_response, request_for, request_key
from ner_experiment.evaluation.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
PRICING = json.loads((ROOT / 'configs/deepseek_pricing_2026-10-24.json').read_text(encoding='utf-8'))
CONFIG = json.loads((ROOT / 'configs/llm_pilot.json').read_text(encoding='utf-8'))
ENDPOINT = 'https://api.deepseek.com'


def response(content='{"entities":[]}', finish='stop', usage=True):
    result = {'model': 'deepseek-flash', 'id': 'synthetic-response',
              'choices': [{'finish_reason': finish, 'message': {'content': content}}]}
    if usage:
        result['usage'] = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
                           'prompt_cache_hit_tokens': 40, 'prompt_cache_miss_tokens': 60}
    return result


class Transport:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def send(self, request):
        self.calls += 1
        if not self.results:
            raise AssertionError('Unexpected extra API call')
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def ledger(tmp_path):
    value = Ledger(tmp_path / 'budget.sqlite3')
    yield value
    value.close()


def execute(ledger, transport, **kwargs):
    return execute_request('dev:0', request_for('北京大学', CONFIG), ENDPOINT, ledger,
                           transport, PRICING, 'pilot', kwargs.pop('stage_limit', 5),
                           sleep=lambda seconds: None, **kwargs)


def test_prompt_is_character_indexed_zero_shot_and_hides_gold():
    messages = messages_for('𠮷😀北')
    assert len(messages) == 2
    input_data = json.loads(messages[-1]['content'])
    assert input_data == {'text': '𠮷😀北', 'character_count': 3, 'characters': [[0, '𠮷'], [1, '😀'], [2, '北']]}
    assert 'entities' not in input_data
    examples = [{'id': 'train:0', 'text': '张三', 'entities': [{'start': 0, 'end': 2, 'type': 'name', 'text': '张三'}], 'bio_compatible': True}]
    assert len(messages_for('北京', examples)) == 4
    with pytest.raises(ValueError):
        messages_for('北京', [{**examples[0], 'bio_compatible': False}])


def test_request_cache_covers_parameters_prompt_examples_endpoint_and_id():
    request = request_for('北京', CONFIG)
    key = request_key('dev:0', request, ENDPOINT)
    assert key == request_key('dev:0', dict(reversed(list(request.items()))), ENDPOINT)
    for changed in ({**request, 'temperature': 1}, {**request, 'max_tokens': 1024},
                    {**request, 'messages': request['messages'] + [{'role': 'user', 'content': 'x'}]}):
        assert key != request_key('dev:0', changed, ENDPOINT)
    assert key != request_key('dev:1', request, ENDPOINT)
    assert key != request_key('dev:0', request, ENDPOINT + '/v1')


@pytest.mark.parametrize('content,finish', [
    ('', 'stop'), ('{}', 'stop'), ('[]', 'stop'), ('```json\n{"entities":[]}\n```', 'stop'),
    ('{"entities":[],"entities":[]}', 'stop'), ('{"entities":[],"explanation":"x"}', 'stop'),
    ('{"entities":[]}', 'length'), ('{"entities":[NaN]}', 'stop')])
def test_no_silent_json_repair(content, finish):
    with pytest.raises(ValueError):
        parse_response(response(content, finish), 'dev:0')


def test_invalid_spans_unknown_types_missing_text_and_duplicates_stay_diagnostic():
    entities = [{'start': 0, 'end': 2, 'type': 'name', 'text': '张三'},
                {'start': 0, 'end': 2, 'type': 'name', 'text': '张三'},
                {'start': 0, 'end': 99, 'type': 'name', 'text': '张三'},
                {'start': 0, 'end': 2, 'type': 'UNKNOWN', 'text': '张三'},
                {'start': 0, 'end': 2, 'type': 'name', 'text': '李四'},
                {'start': 0, 'end': 2, 'type': 'name'}]
    parsed = parse_response(response(json.dumps({'entities': entities})), 'dev:0')
    gold = {'id': 'dev:0', 'text': '张三', 'tokens': ['张', '三'], 'has_labels': True,
            'bio_compatible': True, 'entities': [entities[0]], 'bio_tags': ['B-name', 'I-name']}
    score = evaluate([gold], [parsed])
    assert score['entity']['f1'] == 1
    assert score['diagnostics']['invalid_entities'] == 4
    assert score['diagnostics']['duplicate_entities'] == 2


def test_cny_cost_uses_cache_hits_and_weekend_offpeak():
    cost, detail = usage_cost(response(), PRICING, '2026-10-24T11:00:00+08:00', '2026-10-24T11:00:01+08:00')
    assert cost == pytest.approx((40 * .02 + 60 + 20 * 4) / 1_000_000)
    assert detail['tariff'] == 'off_peak'
    peak_cost, _ = usage_cost(response(), PRICING, '2026-10-23T11:59:59+08:00', '2026-10-23T12:00:01+08:00')
    assert peak_cost == pytest.approx(2 * cost)
    raw = response(); raw['usage'].pop('prompt_cache_hit_tokens'); raw['usage'].pop('prompt_cache_miss_tokens')
    cost_unknown, detail = usage_cost(raw, PRICING, '2026-10-24T11:00:00+08:00', '2026-10-24T11:00:01+08:00')
    assert cost_unknown >= cost and not detail['cache_hit_usage_known']


def test_budget_stop_sends_no_call_and_is_resumable(ledger):
    transport = Transport(response())
    record, cached = execute(ledger, transport, stage_limit=0)
    assert record['status'] == 'budget_stopped' and not cached and transport.calls == 0
    assert ledger.spent() == 0
    record, cached = execute(ledger, transport)
    assert record['status'] == 'ok' and transport.calls == 1


def test_success_cache_survives_reopen_and_no_additional_charges(tmp_path):
    path = tmp_path / 'ledger.sqlite3'
    first = Ledger(path)
    transport = Transport(response())
    expected, cached = execute(first, transport)
    amount = first.spent(); first.close()
    second = Ledger(path)
    actual, cached = execute(second, Transport())
    assert actual == expected and cached and second.spent() == amount
    second.close()


def test_rate_limit_retries_exactly_twice_and_no_sdk_hidden_retries(ledger):
    rate_limit = TransportError('http_error', status=429, retryable=True, rejected=True)
    transport = Transport(rate_limit, rate_limit, response())
    record, _ = execute(ledger, transport)
    assert record['status'] == 'ok' and transport.calls == 3
    assert len(record['attempts']) == 3
    assert record['attempts'][0]['accounted_cny'] == 0
    again, cached = execute(ledger, Transport())
    assert cached and again == record


def test_timeout_exhaustion_keeps_three_unknown_reservations(ledger):
    timeout = TransportError('timeout', retryable=True)
    transport = Transport(timeout, timeout, timeout)
    record, _ = execute(ledger, transport)
    assert record['status'] == 'api_error' and transport.calls == 3
    assert ledger.spent() == pytest.approx(3 * reserve_cost(request_for('北京大学', CONFIG), PRICING))
    assert ledger.summary()['unknown_or_pending_attempts'] == 3


def test_auth_error_stops_without_retry(ledger):
    transport = Transport(TransportError('http_error', status=401, rejected=True))
    record, _ = execute(ledger, transport)
    assert record['status'] == 'api_error' and record['fatal']
    assert transport.calls == 1 and ledger.spent() == 0


def test_parse_failure_has_no_hidden_repair_call(ledger):
    transport = Transport(response('not json'))
    record, _ = execute(ledger, transport)
    assert record['status'] == 'parse_error' and transport.calls == 1
    assert ledger.spent() > 0


def test_raw_response_recovery_without_new_network_call(ledger):
    request = request_for('北京大学', CONFIG)
    key = request_key('dev:0', request, ENDPOINT)
    attempt = ledger.reserve(key, 1, 'pilot', reserve_cost(request, PRICING), 5)
    ledger.finish(attempt, response=response(), cost=0.001, detail={'elapsed_seconds': 1})
    record, cached = execute(ledger, Transport())
    assert not cached and record['status'] == 'ok'
    assert ledger.summary()['attempt_count'] == 1


def test_ambiguous_crash_is_not_automatically_resent(ledger):
    request = request_for('北京大学', CONFIG)
    key = request_key('dev:0', request, ENDPOINT)
    ledger.reserve(key, 1, 'pilot', reserve_cost(request, PRICING), 5)
    record, _ = execute(ledger, Transport())
    assert record['status'] == 'api_error' and record['fatal']
    assert ledger.summary()['unknown_or_pending_attempts'] == 1


def test_global_budget_applies_across_stages(ledger):
    ledger.reserve('first-stage', 1, 'first', 99.99, 100)
    with pytest.raises(BudgetStopped):
        ledger.reserve('second-stage', 1, 'second', .02, 100)
    assert ledger.spent() == 99.99


def test_invalid_usage_holds_reservation_and_stops(ledger):
    record, _ = execute(ledger, Transport(response(usage=False)))
    assert record['status'] == 'ok' and record['fatal']
    assert ledger.spent() == reserve_cost(request_for('北京大学', CONFIG), PRICING)
    assert ledger.summary()['unknown_or_pending_attempts'] == 1


def test_cache_checksum_detects_corruption(ledger):
    execute(ledger, Transport(response()))
    ledger.db.execute('UPDATE cache SET record_json=?', ('{}',)); ledger.db.commit()
    with pytest.raises(ValueError, match='Cache checksum mismatch'):
        execute(ledger, Transport())


def test_sdk_retries_disabled_and_only_official_endpoint(monkeypatch):
    import openai
    from ner_experiment.llm.client import DeepSeekTransport
    captured = {}
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: captured.update(kwargs))
    DeepSeekTransport('synthetic-test-placeholder', ENDPOINT, 45)
    assert captured['max_retries'] == 0 and captured['timeout'] == 45
    with pytest.raises(ValueError, match='official HTTPS'):
        DeepSeekTransport('synthetic-test-placeholder', 'https://other.invalid', 45)


def test_full_mock_pilot_offline_recompute_freeze_and_cached_resume(tmp_path, monkeypatch):
    from ner_experiment.llm import pilot
    config = deepcopy(CONFIG)
    pricing = deepcopy(PRICING)
    pricing['verified_date_shanghai'] = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    entity = {'start': 0, 'end': 4, 'type': 'organization', 'text': '北京大学'}
    rows = [{'id': f'dev:{i}', 'text': '北京大学', 'tokens': list('北京大学'),
             'has_labels': True, 'bio_compatible': True, 'entities': [entity],
             'bio_tags': ['B-organization'] + ['I-organization'] * 3} for i in range(20)]
    requests = [{'id': row['id'], 'request': request_for(row['text'], config)} for row in rows]
    context = {'config': config, 'endpoint': ENDPOINT, 'no_credential_here': True}
    monkeypatch.setattr(pilot, 'prepare', lambda root: (config, pricing, rows, requests, context, 'synthetic-test-placeholder'))
    fake = Transport(*[response(json.dumps({'entities': [entity]})) for _ in rows])
    fake.close = lambda: None
    monkeypatch.setattr(pilot, 'DeepSeekTransport', lambda *args: fake)
    summary = pilot.run_pilot(tmp_path)
    assert summary['runtime_freeze_ready'] and fake.calls == 20
    assert summary['entity']['f1'] == summary['token']['f1'] == 1
    assert (tmp_path / 'configs/llm_runtime_frozen.json').exists()
    directory = tmp_path / config['run_directory']
    saved_metrics = (directory / 'metrics.json').read_bytes()
    resumed = pilot.run_pilot(tmp_path)
    assert resumed['api_calls_this_invocation'] == 0
    assert resumed['cached_samples_this_invocation'] == 20
    assert fake.calls == 20 and (directory / 'metrics.json').read_bytes() == saved_metrics
    assert not (directory / 'run.lock').exists()
