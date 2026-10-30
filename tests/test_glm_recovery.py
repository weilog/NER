"""Offline network/billing recovery checks. No real transport or credentials used."""
from copy import deepcopy
import importlib.util
import socket
import ssl
import sys
import urllib.error

import pytest
from ner_experiment.common import ROOT, file_hash, load_json, object_hash, save_json
from ner_experiment.models.glm import InterfaceStop
from ner_experiment.models.glm_recovery import RecoveryGLMClient, reconciled_total, validate_reconciliation


ROW = {'id': 'x', 'text': '张三', 'tokens': list('张三'), 'entities': []}


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('GLM_API_KEY', 'test-key-not-a-real-secret')
    monkeypatch.setenv('GLM_MODEL', 'glm-5.3-flashx')
    monkeypatch.setenv('GLM_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4/')
    return load_json(ROOT / 'configs/glm_flashx.json')


def messages(row, examples):
    return [{'role': 'user', 'content': row['text']}]


def response(payload):
    return {'model': 'mock', 'choices': [{'finish_reason': 'stop', 'message': {'content': '{"entities":[]}'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}


def record_for(ledger, charge):
    return {'prefix_attempts': len(ledger['attempts']), 'prefix_sha256': object_hash(ledger['attempts']),
            'confirmed_charge_upper_cny': charge}


def test_reconciliation_keeps_original_ledger_and_accounts_new_unknown_attempts():
    ledger = {'attempts': [{'status': 'response', 'charged_upper_cny': 3.83},
                           {'status': 'network_error', 'charged_upper_cny': 26.11}]}
    before = deepcopy(ledger)
    record = record_for(ledger, 3.84)
    validate_reconciliation(ledger, record)
    assert reconciled_total(ledger, record) == 3.84 and ledger == before
    ledger['attempts'].append({'status': 'network_error', 'charged_upper_cny': .07})
    validate_reconciliation(ledger, record)
    assert reconciled_total(ledger, record) == pytest.approx(3.91)
    ledger['attempts'][0]['charged_upper_cny'] = 0
    with pytest.raises(ValueError, match='prefix changed'):
        validate_reconciliation(ledger, record)


@pytest.mark.parametrize('charge', [0, float('nan'), float('inf'), True])
def test_reconciliation_rejects_under_count_or_nonfinite_charge(charge):
    ledger = {'attempts': [{'status': 'response', 'charged_upper_cny': 3.83}]}
    with pytest.raises(ValueError):
        validate_reconciliation(ledger, record_for(ledger, charge))


@pytest.mark.parametrize('reason', [socket.gaierror(11001, 'mock DNS'),
                                  ConnectionRefusedError(10061, 'mock refused'),
                                  ssl.SSLCertVerificationError(1, 'mock certificate')])
def test_proven_pre_send_error_stops_before_retry_and_releases_only_new_reservation(tmp_path, config, reason):
    calls = []
    def fail(payload):
        calls.append(1)
        raise urllib.error.URLError(reason)
    c = RecoveryGLMClient(tmp_path, config, messages, 30, transport=fail, sleep=lambda _: None)
    with pytest.raises(InterfaceStop):
        c.predict(ROW, [])
    assert len(calls) == 1
    entry = c.ledger['attempts'][0]
    assert entry['verified_not_sent'] and entry['charged_upper_cny'] == 0
    assert entry['reservation_released_cny'] > 0
    assert c.total() == 0 and c.stop_event.is_set()
    assert entry['network_diagnostic']['reason_type'] == type(reason).__name__


def test_unknown_timeout_retains_reservation_and_eight_failures_stop(tmp_path, config):
    c = RecoveryGLMClient(tmp_path, config, messages, 30,
                          transport=lambda _: (_ for _ in ()).throw(TimeoutError()), sleep=lambda _: None)
    for i in range(2):
        assert c.predict({**ROW, 'id': str(i), 'text': str(i)}, [])['status'] == 'api_error'
    with pytest.raises(InterfaceStop):
        c.predict({**ROW, 'id': '2', 'text': '2'}, [])
    assert len(c.ledger['attempts']) == 8 and c.total() > 0
    assert c.stop_event.is_set()
    assert all(not a.get('verified_not_sent') for a in c.ledger['attempts'])


def test_success_resets_consecutive_network_counter(tmp_path, config):
    calls = []
    def fake(payload):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError()
        return response(payload)
    c = RecoveryGLMClient(tmp_path, config, messages, 30, transport=fake, sleep=lambda _: None)
    assert c.predict(ROW, [])['status'] == 'ok'
    assert c.consecutive_connection_failures == 0 and not c.stop_event.is_set()


def test_cached_failed_prediction_never_re_requested(tmp_path, config):
    def fail(payload):
        raise TimeoutError()
    c = RecoveryGLMClient(tmp_path, config, messages, 30, transport=fail, sleep=lambda _: None)
    expected = c.predict(ROW, [])
    path = c.cache / (expected['cache_key'] + '.json')
    before = file_hash(path)
    def forbidden(payload):
        raise AssertionError('Existing failure must not be re-requested')
    restored = RecoveryGLMClient(tmp_path, config, messages, 30, transport=forbidden)
    assert restored.predict(ROW, []) == expected and file_hash(path) == before


def test_network_diagnostic_redacts_key(tmp_path, config):
    def fail(payload):
        raise urllib.error.URLError(ConnectionRefusedError(1, 'test-key-not-a-real-secret'))
    c = RecoveryGLMClient(tmp_path, config, messages, 30, transport=fail, sleep=lambda _: None)
    with pytest.raises(InterfaceStop):
        c.predict(ROW, [])
    assert 'test-key-not-a-real-secret' not in c.ledger_path.read_text(encoding='utf-8')


def test_resume_without_verified_bill_cannot_construct_client(monkeypatch, capsys):
    scripts = ROOT / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('glm_network_resume', scripts / 'resume_glm_test.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if (ROOT / 'runs/glm_flashx_test_v1/billing_reconciliation.json').exists():
        pytest.skip('Real bill already reconciled; do not alter it')
    def forbidden(*a, **kw):
        raise AssertionError('No verified bill: client must not be constructed')
    monkeypatch.setattr(runner, 'RecoveryGLMClient', forbidden)
    monkeypatch.setattr(sys, 'argv', ['resume_glm_test.py', '--run'])
    runner.main()
    assert 'No requests sent' in capsys.readouterr().out


def test_recovery_uses_canonical_frozen_numbers_in_hashes(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('glm_resume_canonical', ROOT / 'scripts/resume_glm_test.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    frozen = {'budget_cny': 30.0}
    candidate = {'budget_cny': 30}
    assert frozen == candidate and object_hash(frozen) != object_hash(candidate)
    monkeypatch.setattr(runner.original, 'freeze_once', lambda *a: {'protocol': frozen})
    assert object_hash(runner.canonical_protocol(tmp_path, candidate)) == object_hash(frozen)
