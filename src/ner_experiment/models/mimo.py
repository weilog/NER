"""MiMo official pay-as-you-go adapter: bounded budget, durable cache and global RPM."""
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request

from dotenv import dotenv_values
from ner_experiment.common import ROOT, load_json, object_hash, utc_now
from ner_experiment.models.glm import InterfaceStop, atomic_json, parse_glm_response, usage_cost
from ner_experiment.models.llm import BudgetStop, validate_cache


def settings(config, env_file=None):
    values = dotenv_values(env_file or ROOT / '.env')
    def value(name, default=''):
        return os.getenv(name) or values.get(name) or default
    base = value('MIMO_BASE_URL', config['base_url']).rstrip('/')
    model = value('MIMO_MODEL', config['model'])
    if base != 'https://api.xiaomimimo.com/v1' or model != 'mimo-v2.6-flash':
        raise ValueError('Use official MiMo pay-as-you-go endpoint and mimo-v2.6-flash')
    return {'key': value('MIMO_API_KEY'), 'endpoint': base + '/chat/completions', 'model': model}


class RateLimiter:
    """Paced starts plus a rolling-minute cap, shared by all workers and retries."""
    def __init__(self, rpm=90, *, clock=time.monotonic, sleeper=None, recent_starts=()):
        if type(rpm) is not int or not 1 <= rpm <= 90:
            raise ValueError('Use integer global RPM in 1..90')
        self.rpm, self.clock, self.sleeper = rpm, clock, sleeper
        self.history, self.next_at = deque(), 0.0
        self.lock = threading.Lock()
        now_wall, now_clock = datetime.now(timezone.utc), self.clock()
        for stamp in sorted(recent_starts):
            age = (now_wall - datetime.fromisoformat(stamp)).total_seconds()
            if 0 <= age < 60:
                self.history.append(now_clock - age)
        if self.history:
            self.next_at = self.history[-1] + 60 / self.rpm

    def acquire(self, stop):
        started = self.clock()
        while True:
            if stop.is_set():
                raise InterfaceStop('MiMo batch stopped while waiting for request slot')
            with self.lock:
                now = self.clock()
                while self.history and now - self.history[0] >= 60:
                    self.history.popleft()
                delay = max(0, self.next_at - now)
                if len(self.history) >= self.rpm:
                    delay = max(delay, self.history[0] + 60 - now)
                if delay <= 1e-9:
                    self.history.append(now)
                    self.next_at = now + 60 / self.rpm
                    return max(0, now - started)
            if self.sleeper:
                self.sleeper(delay)
            elif stop.wait(delay):
                raise InterfaceStop('MiMo batch stopped during rate-limit wait')


class MiMoClient:
    def __init__(self, folder, config, message_builder, budget, *, transport=None,
                 limiter=None, sleep=time.sleep, workers=4):
        self.settings = settings(config)
        if transport is None and not self.settings['key']:
            raise RuntimeError('MIMO_API_KEY is empty')
        if not math.isfinite(budget) or budget < 0:
            raise ValueError('Budget must be finite and nonnegative')
        self.parameters = {'model': self.settings['model'], **deepcopy(config['parameters'])}
        self.prices = deepcopy(config['pricing'])
        self.folder = Path(folder)
        self.cache = self.folder / 'cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.folder / 'ledger.json'
        self.ledger = load_json(self.ledger_path) if self.ledger_path.exists() else {'attempts': []}
        self.budget, self.message_builder, self.workers = budget, message_builder, workers
        self.transport, self.sleep = transport or self.http, sleep
        self.lock, self.stop_event = threading.RLock(), threading.Event()
        self.limiter = limiter or RateLimiter(config['rpm'], recent_starts=[
            a['started_at'] for a in self.ledger['attempts'] if a.get('started_at')])
        self.consecutive_connection_failures = 0
        self.consecutive_rate_failures = 0
        self.consecutive_transient_failures = 0

    def total(self):
        with self.lock:
            return sum(a['charged_upper_cny'] for a in self.ledger['attempts'])

    def persist(self):
        atomic_json(self.ledger_path, self.ledger)

    def estimate_upper(self, payload):
        allowance = len(json.dumps(payload, ensure_ascii=False).encode('utf-8')) + 1024
        return (allowance * self.prices['input_cache_miss']
                + payload['max_completion_tokens'] * self.prices['output']) / 1e6

    def reserve(self, key, payload):
        with self.lock:
            if self.stop_event.is_set():
                raise InterfaceStop('MiMo batch stopped before request')
            upper = self.estimate_upper(payload)
            if self.total() + upper > self.budget:
                raise BudgetStop('MiMo budget ceiling reached before request')
            index = len(self.ledger['attempts'])
            self.ledger['attempts'].append({'key': key, 'started_at': utc_now(),
                                          'status': 'reserved', 'charged_upper_cny': upper})
            self.persist()
            return index

    def http(self, payload):
        request = urllib.request.Request(self.settings['endpoint'],
            json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={'api-key': self.settings['key'], 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    def error_detail(self, exc):
        error = {'exception': type(exc).__name__, 'http_status': getattr(exc, 'code', None)}
        if isinstance(exc, urllib.error.HTTPError):
            try:
                body = exc.read(65536).decode('utf-8', errors='replace')
                if self.settings['key']:
                    body = body.replace(self.settings['key'], '[REDACTED]')
                detail = json.loads(body).get('error', {})
                if isinstance(detail, dict):
                    error.update(provider_code=str(detail.get('code', '')),
                                 provider_message=str(detail.get('message', ''))[:2000])
            except (ValueError, TypeError, AttributeError, OSError):
                pass
        else:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            message = str(reason)
            if self.settings['key']:
                message = message.replace(self.settings['key'], '[REDACTED]')
            error.update(reason_type=type(reason).__name__, errno=getattr(reason, 'errno', None),
                         winerror=getattr(reason, 'winerror', None), reason=message[:1000])
        return error

    def predict(self, row, examples):
        payload = {**self.parameters, 'messages': self.message_builder(row, examples)}
        endpoint = self.settings['endpoint']
        key = object_hash({'endpoint': endpoint, 'payload': payload})
        path = self.cache / (key + '.json')
        if path.exists():
            result = validate_cache(load_json(path), payload, endpoint, parse_glm_response)
            if result.get('fatal_error'):
                raise InterfaceStop('MiMo fatal interface failure cached; review before resuming')
            return {**result, 'id': row['id']}
        with self.lock:
            previous = [a.copy() for a in self.ledger['attempts'] if a['key'] == key]
        if any(a['status'] in {'reserved', 'response'} for a in previous):
            raise InterfaceStop('MiMo request may be charged but lacks cache; audit before resend')
        started = time.perf_counter()
        raw, error, fatal, batch_stop = None, None, False, False
        attempts, throttle = len(previous), 0.0
        http_seconds = 0.0
        for attempt in range(attempts, 3):
            throttle += self.limiter.acquire(self.stop_event)
            index = self.reserve(key, payload)
            attempts += 1
            call_started = time.perf_counter()
            try:
                raw = self.transport(payload)
                elapsed = time.perf_counter() - call_started
                http_seconds += elapsed
                if not isinstance(raw, dict):
                    raw = {'error': {'message': 'Non-object response'}}
                with self.lock:
                    self.consecutive_connection_failures = 0
                    self.consecutive_rate_failures = 0
                    self.consecutive_transient_failures = 0
                    entry = self.ledger['attempts'][index]
                    entry.update(status='response', usage=raw.get('usage') or {},
                                 http_seconds=elapsed, finished_at=utc_now())
                    try:
                        cost = usage_cost(entry['usage'], self.prices)
                        if cost > entry['charged_upper_cny'] + 1e-12:
                            fatal, error = True, {'reason': 'usage_exceeds_reservation'}
                        entry['charged_upper_cny'] = cost
                    except (ValueError, TypeError, AttributeError):
                        fatal, error = True, {'reason': 'missing_or_invalid_usage'}
                    self.persist()
                if raw.get('error'):
                    fatal, error = True, {'reason': 'error_in_HTTP200_response'}
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                elapsed = time.perf_counter() - call_started
                http_seconds += elapsed
                error = self.error_detail(exc)
                code = error['http_status']
                retry = code is None or code == 429 or (isinstance(code, int) and code >= 500)
                with self.lock:
                    entry = self.ledger['attempts'][index]
                    entry.update(status='network_error', error=error, http_seconds=elapsed, finished_at=utc_now())
                    if isinstance(exc, urllib.error.HTTPError):
                        self.consecutive_connection_failures = 0
                        self.consecutive_rate_failures = self.consecutive_rate_failures + 1 if code == 429 else 0
                        if self.consecutive_rate_failures >= 8:
                            batch_stop = True
                        self.consecutive_transient_failures = (self.consecutive_transient_failures + 1
                            if code == 429 or code >= 500 else 0)
                    else:
                        self.consecutive_connection_failures += 1
                        self.consecutive_transient_failures += 1
                        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                        if isinstance(reason, (socket.gaierror, ConnectionRefusedError, ssl.SSLCertVerificationError)):
                            entry['verified_not_sent'] = True
                            entry['reservation_released_cny'] = entry['charged_upper_cny']
                            entry['charged_upper_cny'] = 0
                            batch_stop = True
                        if self.consecutive_connection_failures >= 8:
                            batch_stop = True
                    if self.consecutive_transient_failures >= 8:
                        batch_stop = True
                    if batch_stop:
                        self.stop_event.set()
                        self.ledger['connection_stop'] = {'at': utc_now(), 'error': error,
                            'consecutive_failed_attempts': self.consecutive_connection_failures,
                            'consecutive_transient_failures': self.consecutive_transient_failures}
                    self.persist()
                if code == 421:
                    break  # documented content refusal: score empty, never re-request
                if not retry:
                    fatal = True
                    break
                if batch_stop:
                    break
                if attempt < 2:
                    self.sleep(2 ** attempt)
            except (ValueError, TypeError):
                error, fatal = {'reason': 'malformed_response_unknown_usage'}, True
                with self.lock:
                    self.ledger['attempts'][index].update(status='network_error', error=error, finished_at=utc_now())
                    self.persist()
                break
        result = {'id': row['id'], 'status': 'api_error', 'entities': [],
                  'latency_seconds': time.perf_counter() - started, 'throttle_seconds': throttle,
                  'http_seconds': http_seconds, 'attempts': attempts, 'cache_key': key,
                  'created_at': utc_now(), 'execution_workers': self.workers}
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
        if batch_stop:
            result['batch_stop'] = True
        record = {'cache_version': 2, 'endpoint': endpoint, 'payload': payload,
                  'payload_sha256': object_hash(payload), 'parameters': self.parameters,
                  'created_at': utc_now(), 'example_ids': [e['id'] for e in examples],
                  'prediction': result, 'raw_response': raw}
        record['record_sha256'] = object_hash(record)
        atomic_json(path, record)
        if fatal or batch_stop:
            self.stop_event.set()
            raise InterfaceStop('MiMo batch stopped; cached diagnostics preserved, no hidden repair')
        return result
