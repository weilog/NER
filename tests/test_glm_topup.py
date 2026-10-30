"""Top-up recovery preserves records; fake responses only, no API calls."""
import importlib.util
import io
from pathlib import Path
import sys
import urllib.error

import pytest
from ner_experiment.common import ROOT, file_hash, load_json
from ner_experiment.models.glm import InterfaceStop
from ner_experiment.models.glm_recovery import RecoveryGLMClient


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('glm_topup_runner', ROOT / 'scripts/resume_glm_after_topup.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_model_reserved_budgets_fit_project_cap(runner):
    assert runner.allocation_limit(12.23) == pytest.approx(72.23)
    with pytest.raises(RuntimeError):
        runner.allocation_limit(50)
    with pytest.raises(ValueError):
        runner.allocation_limit(float('nan'))


def test_provider_lock_does_not_touch_mimo_lock(tmp_path, runner):
    base = tmp_path / 'glm'
    base.mkdir()
    shared = tmp_path / 'process.lock'
    shared.write_text('77')
    with runner.provider_lock(base, shared, 77):
        assert shared.read_text() == '77'
        assert (base / 'provider.lock').exists()
        with pytest.raises(FileExistsError):
            with runner.provider_lock(base, shared, 77):
                pass
    assert shared.read_text() == '77' and not (base / 'provider.lock').exists()
    with pytest.raises(RuntimeError, match='owner changed'):
        with runner.provider_lock(base, shared, 88):
            pass


def test_balance_failure_archived_then_retried_without_overwriting_history(tmp_path, runner, monkeypatch):
    monkeypatch.setenv('GLM_API_KEY', 'fake-key')
    monkeypatch.setenv('GLM_MODEL', 'glm-5.3-flashx')
    monkeypatch.setenv('GLM_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4/')
    config = load_json(ROOT / 'configs/glm_flashx.json')
    row = {'id': 'x', 'text': '张三', 'tokens': list('张三')}
    messages = lambda r, e: [{'role': 'user', 'content': r['text']}]
    def fail(payload):
        raise urllib.error.HTTPError('url', 429, 'fake', None,
            io.BytesIO(b'{"error":{"code":"1113","message":"insufficient balance"}}'))
    c = RecoveryGLMClient(tmp_path, config, messages, 40, transport=fail, sleep=lambda _: None)
    with pytest.raises(InterfaceStop):
        c.predict(row, [])
    old = next(c.cache.glob('*.json'))
    old_hash = file_hash(old)
    candidates = runner.quota_candidates(tmp_path)
    assert len(candidates) == 1 and candidates[0]['previous_attempts'] == 1
    runner.archive_quota(tmp_path, candidates)
    archived = tmp_path / 'quota_recovery_v1/prior_cache' / old.name
    assert not old.exists() and file_hash(archived) == old_hash
    def success(payload):
        return {'model': 'glm-5.3-flashx', 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"entities":[]}'}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}
    restored = RecoveryGLMClient(tmp_path, config, messages, 40, transport=success)
    result = restored.predict(row, [])
    assert result['status'] == 'ok' and result['attempts'] == 2
    assert file_hash(archived) == old_hash and len(restored.ledger['attempts']) == 2
    assert restored.ledger['attempts'][0] == c.ledger['attempts'][0]
    assert not runner.quota_candidates(tmp_path)  # successful answer can never be retried as quota recovery


def test_archive_refuses_changed_cache(tmp_path, runner):
    base = tmp_path
    (base / 'cache').mkdir()
    key = 'a' * 64
    path = base / 'cache' / (key + '.json')
    path.write_text('original')
    candidates = [{'key': key, 'sha256': file_hash(path)}]
    path.write_text('changed')
    with pytest.raises(RuntimeError):
        runner.archive_quota(base, candidates)
    assert path.read_text() == 'changed'
