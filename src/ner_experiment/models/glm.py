"""Official BigModel adapter. No API call on import or construction."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from dotenv import dotenv_values
from ner_experiment.common import ROOT, load_json, object_hash, save_json, utc_now
from ner_experiment.models.llm import BudgetStop, parse_response, validate_cache


class InterfaceStop(RuntimeError):
    """Authentication, unsupported parameters, billing or unknown charging: stop batch."""


def settings(config, env_file=None):
    values = dotenv_values(env_file or ROOT / '.env')
    def value(name, default=''):
        return os.getenv(name) or values.get(name) or default
    base = value(config['base_url_env'], config['base_url']).rstrip('/')
    if base != 'https://open.bigmodel.cn/api/paas/v4':
        raise ValueError('GLM_BASE_URL must be the official BigModel standard API endpoint')
    model = value(config['model_env'], config['model'])
    if model != 'glm-5.3-flashx':
        raise ValueError('This frozen experiment requires GLM_MODEL=glm-5.3-flashx')
    return {'key': value(config['api_key_env']), 'endpoint': base + '/chat/completions', 'model': model}


def parse_glm_response(raw):
    status, entities = parse_response(raw)
    if status == 'ok' and any(not isinstance(e.get('text'), str) for e in entities):
        return 'format_error', []
    return status, entities


def usage_cost(usage, prices):
    """Completion usage is the API's total output count; do not add reasoning twice."""
    prompt, output = usage.get('prompt_tokens'), usage.get('completion_tokens')
    details = usage.get('prompt_tokens_details') or {}
    hit = details.get('cached_tokens', 0) if isinstance(details, dict) else None
    if any(type(n) is not int or n < 0 for n in (prompt, output, hit)) or hit > prompt:
        raise ValueError('Missing or inconsistent GLM usage; keep conservative reservation')
    return ((prompt - hit) * prices['input_cache_miss'] + hit * prices['input_cache_hit']
            + output * prices['output']) / 1e6


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    save_json(temp, value)
    temp.replace(path)


class GLMClient:
    def __init__(self, folder, config, message_builder, budget, *, transport=None, sleep=time.sleep,
                 workers=1, accepted_failure_keys=()):
        self.settings = settings(config)
        if transport is None and not self.settings['key']:
            raise RuntimeError('GLM_API_KEY is empty; set it in project .env or process environment')
        if not math.isfinite(budget) or budget < 0:
            raise ValueError('Budget must be finite and nonnegative')
        self.parameters = {'model': self.settings['model'], **deepcopy(config['parameters'])}
        self.prices = deepcopy(config['pricing'])
        self.folder = Path(folder)
        self.cache = self.folder / 'cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.folder / 'ledger.json'
        self.ledger = load_json(self.ledger_path) if self.ledger_path.exists() else {'attempts': []}
        self.budget, self.message_builder = budget, message_builder
        self.transport, self.sleep = transport or self.http, sleep
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.workers = workers
        self.accepted_failure_keys = set(accepted_failure_keys)

    def total(self):
        with self.lock:
            return sum(entry['charged_upper_cny'] for entry in self.ledger['attempts'])

    def persist(self):
        atomic_json(self.ledger_path, self.ledger)

    def estimate_upper(self, payload):
        # Same conservative byte/framing allowance as existing adapter; no cache discount.
        input_allowance = len(json.dumps(payload, ensure_ascii=False).encode('utf-8')) + 1024
        return (input_allowance * self.prices['input_cache_miss']
                + payload['max_tokens'] * self.prices['output']) / 1e6

    def reserve(self, key, payload):
        upper = self.estimate_upper(payload)
        with self.lock:
            if self.stop_event.is_set():
                raise InterfaceStop('Batch stopped before scheduling another request')
            if self.total() + upper > self.budget:
                raise BudgetStop('GLM budget reached before request; cached work is preserved')
            index = len(self.ledger['attempts'])
            self.ledger['attempts'].append({'key': key, 'started_at': utc_now(),
                                          'status': 'reserved', 'charged_upper_cny': upper})
            self.persist()
        return index

    def http(self, payload):
        request = urllib.request.Request(
            self.settings['endpoint'], json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={'Authorization': 'Bearer ' + self.settings['key'], 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    def predict(self, row, examples):
        payload = {**self.parameters, 'messages': self.message_builder(row, examples)}
        endpoint = self.settings['endpoint']
        key = object_hash({'endpoint': endpoint, 'payload': payload})
        path = self.cache / (key + '.json')
        if path.exists():
            record = load_json(path)
            result = validate_cache(record, payload, endpoint, parse_glm_response)
            accepted = (key in self.accepted_failure_keys and result['status'] == 'api_error'
                        and result.get('error', {}).get('http_status') == 400)
            if result.get('fatal_error') and not accepted:
                raise InterfaceStop('Previously stopped request is cached; inspect it before continuing')
            return {**result, 'id': row['id']}

        # A crash after a reservation could have incurred a charge; never silently resend it.
        with self.lock:
            previous = [e.copy() for e in self.ledger['attempts'] if e['key'] == key]
        if any(e['status'] in {'reserved', 'response'} for e in previous):
            raise InterfaceStop('Unfinished charged/reserved request without cache; manual audit required')
        start = time.perf_counter()
        raw, error, fatal = None, previous[-1].get('error') if previous else None, False
        attempts = len(previous)
        for attempt in range(attempts, 3):
            index = self.reserve(key, payload)
            attempts += 1
            try:
                raw = self.transport(payload)
                if not isinstance(raw, dict):
                    raw = {'error': {'reason': 'non_object_API_response'}}
                    raise ValueError('Non-object API response')
                usage = raw.get('usage') or {}
                with self.lock:
                    entry = self.ledger['attempts'][index]
                    entry.update(status='response', usage=usage, finished_at=utc_now())
                    try:
                        cost = usage_cost(usage, self.prices)
                        fatal = cost > entry['charged_upper_cny'] + 1e-12
                        entry['charged_upper_cny'] = cost
                        if fatal:
                            error = {'reason': 'usage_exceeded_pre_request_allowance'}
                    except (ValueError, TypeError):
                        error, fatal = {'reason': 'usage_missing_or_invalid'}, True
                    self.persist()
                if raw.get('error'):
                    error, fatal = {'reason': 'API_error_in_response'}, True
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                code = getattr(exc, 'code', None)
                # Never include exception text: transports may echo credentials/request headers.
                error = {'exception': type(exc).__name__, 'http_status': code}
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        body = exc.read(65536).decode('utf-8', errors='replace')
                        if self.settings['key']:
                            body = body.replace(self.settings['key'], '[REDACTED]')
                        detail = json.loads(body).get('error', {})
                        if isinstance(detail, dict):
                            error['provider_error_code'] = str(detail.get('code', ''))
                            error['provider_error_message'] = str(detail.get('message', ''))[:2000]
                    except (OSError, ValueError, AttributeError, TypeError):
                        pass
                retry = code is None or code == 429 or (isinstance(code, int) and code >= 500)
                if error.get('provider_error_code') in {'1113', '1308', '1309', '1310', '1311',
                                                        '1313', '1314', '1315', '1316', '1317',
                                                        '1318', '1319', '1320', '1321'}:
                    retry = False
                with self.lock:
                    self.ledger['attempts'][index].update(status='network_error', error=error, finished_at=utc_now())
                    self.persist()
                if not retry:
                    # A documented per-sample provider refusal is a scored failure, never retried.
                    fatal = not (code == 400 and error.get('provider_error_code') == '1301')
                    break
                if attempt < 2:
                    self.sleep(2 ** attempt)
            except (ValueError, TypeError):
                error, fatal = {'reason': 'malformed_API_response_unknown_usage'}, True
                with self.lock:
                    self.ledger['attempts'][index].update(status='network_error', error=error, finished_at=utc_now())
                    self.persist()
                break

        result = {'id': row['id'], 'status': 'api_error', 'entities': [],
                  'latency_seconds': time.perf_counter() - start, 'attempts': attempts,
                  'cache_key': key, 'created_at': utc_now(), 'execution_workers': self.workers}
        if raw is not None:
            result['status'], result['entities'] = parse_glm_response(raw)
            choices = raw.get('choices')
            first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            result.update(actual_model=raw.get('model'), usage=raw.get('usage', {}),
                          finish_reason=first.get('finish_reason'))
        if error:
            result['error'] = error
        if fatal:
            result['fatal_error'] = True
        record = {'cache_version': 2, 'endpoint': endpoint, 'payload': payload,
                  'payload_sha256': object_hash(payload), 'created_at': utc_now(),
                  'parameters': self.parameters, 'example_ids': [e['id'] for e in examples],
                  'prediction': result, 'raw_response': raw}
        record['record_sha256'] = object_hash(record)
        atomic_json(path, record)
        if fatal:
            raise InterfaceStop('GLM interface/usage check stopped the batch; inspect cached error (key omitted)')
        return result
