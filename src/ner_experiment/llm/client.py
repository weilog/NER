"""Explicit retries, persistent raw responses, and no model repair calls."""
import json
import time
from urllib.parse import urlparse

from ner_experiment.llm.budget import BudgetStopped, now, reserve_cost, usage_cost
from ner_experiment.llm.prompts import (PARSER_VERSION, PROMPT_VERSION,
                                        parse_response, request_key)


class TransportError(Exception):
    def __init__(self, kind, status=None, retryable=False, rejected=False):
        super().__init__(kind)
        self.kind, self.status = kind, status
        self.retryable, self.rejected = retryable, rejected


class DeepSeekTransport:
    def __init__(self, api_key, endpoint, timeout):
        from openai import OpenAI
        parsed = urlparse(endpoint)
        if parsed.scheme != 'https' or parsed.hostname != 'api.deepseek.com' or parsed.username or parsed.password:
            raise ValueError('This experiment uses only the official HTTPS DeepSeek endpoint')
        self.client = OpenAI(api_key=api_key, base_url=endpoint,
                             timeout=timeout, max_retries=0)

    def send(self, request):
        from openai import APIConnectionError, APITimeoutError, APIStatusError
        try:
            return self.client.chat.completions.create(**request).model_dump(mode='json')
        except APITimeoutError:
            raise TransportError('timeout', retryable=True) from None
        except APIConnectionError:
            raise TransportError('connection_error', retryable=True) from None
        except APIStatusError as exc:
            status = exc.status_code
            raise TransportError('http_error', status=status,
                                 retryable=status == 429 or status >= 500,
                                 rejected=400 <= status < 500) from None
        except Exception:
            # Never persist exception strings or headers which may contain secrets.
            raise TransportError('unexpected_sdk_error', retryable=False) from None

    def close(self):
        self.client.close()


def execute_request(sample_id, request, endpoint, ledger, transport, pricing,
                    stage, stage_limit, max_retries=2, backoff=(1, 2), sleep=time.sleep):
    if max_retries not in (0, 1, 2):
        raise ValueError('At most two explicit retries allowed')
    key = request_key(sample_id, request, endpoint)
    cached = ledger.cached(key)
    if cached is not None:
        return cached, True
    record = {'id': sample_id, 'request_key': key, 'request': request,
              'endpoint': endpoint, 'prompt_version': PROMPT_VERSION,
              'parser_version': PARSER_VERSION, 'pricing_snapshot': pricing,
              'fatal': False, 'operational_issues': []}
    amount = reserve_cost(request, pricing)
    attempts = ledger.attempts(key)
    # A previous completed response can be reparsed without another paid call.
    response_attempt = next((attempt for attempt in attempts if attempt['response_json'] is not None), None)
    interrupted = any(attempt['state'] == 'reserved' for attempt in attempts)
    if interrupted:
        prediction = {'id': sample_id, 'status': 'api_error', 'entities': [],
                      'error_code': 'ambiguous_interrupted_attempt_no_automatic_resend'}
        record['fatal'] = True
    else:
        prediction = None
    while response_attempt is None and prediction is None and len(attempts) < max_retries + 1:
        if attempts:
            error = json.loads(attempts[-1]['error_json'])
            if not error['retryable']:
                break
        try:
            attempt_id = ledger.reserve(key, len(attempts) + 1, stage, amount, stage_limit)
        except BudgetStopped:
            record.update(prediction={'id': sample_id, 'status': 'budget_stopped', 'entities': []},
                          attempts=attempts, status='budget_stopped', fatal=True)
            # No terminal cache for unsent requests; a later authorized stage may resume.
            return record, False
        started = now()
        timer = time.perf_counter()
        try:
            response = transport.send(request)
        except TransportError as exc:
            elapsed = time.perf_counter() - timer
            error = {'kind': exc.kind, 'http_status': exc.status,
                     'retryable': exc.retryable, 'elapsed_seconds': elapsed}
            ledger.finish(attempt_id, error=error, cost=0.0 if exc.rejected else None)
            attempts = ledger.attempts(key)
            if exc.retryable and len(attempts) < max_retries + 1:
                sleep(backoff[len(attempts) - 1])
            continue
        except Exception:
            # Unknown transport errors also retain the complete reservation.
            ledger.finish(attempt_id, error={'kind': 'unexpected_transport_error',
                          'retryable': False, 'elapsed_seconds': time.perf_counter() - timer})
            attempts = ledger.attempts(key)
            continue
        finished = now()
        elapsed = time.perf_counter() - timer
        try:
            cost, detail = usage_cost(response, pricing, started, finished)
            detail['elapsed_seconds'] = elapsed
        except (ValueError, TypeError, AttributeError):
            cost, detail = None, {'usage_invalid': True, 'elapsed_seconds': elapsed}
        ledger.finish(attempt_id, response=response, cost=cost, detail=detail)
        attempts = ledger.attempts(key)
        response_attempt = attempts[-1]
    if response_attempt is not None:
        response = json.loads(response_attempt['response_json'])
        record['raw_response'] = response
        record['response_model'] = response.get('model')
        record['system_fingerprint'] = response.get('system_fingerprint')
        cost_detail = json.loads(response_attempt['cost_detail_json'])
        if cost_detail.get('usage_invalid'):
            record['operational_issues'].append('missing_or_invalid_usage_full_reservation_retained')
            record['fatal'] = True
        if response_attempt['accounted_cny'] > response_attempt['reserved_cny'] + 1e-12:
            record['operational_issues'].append('actual_usage_exceeded_reservation_stop')
            record['fatal'] = True
        if not isinstance(response.get('model'), str) or not response['model']:
            record['operational_issues'].append('missing_response_model')
            record['fatal'] = True
        choices = response.get('choices') or []
        if choices and choices[0].get('message', {}).get('reasoning_content'):
            record['operational_issues'].append('unexpected_thinking_output')
            record['fatal'] = True
        try:
            prediction = parse_response(response, sample_id)
        except (ValueError, TypeError, KeyError, IndexError):
            prediction = {'id': sample_id, 'status': 'parse_error', 'entities': [],
                          'error_code': 'response_json_or_schema_or_finish_invalid'}
    elif prediction is None:
        last_error = json.loads(attempts[-1]['error_json']) if attempts else {}
        prediction = {'id': sample_id, 'status': 'api_error', 'entities': [],
                      'error_code': last_error.get('kind', 'no_response')}
        record['fatal'] = not last_error.get('retryable', False)
    record.update(prediction=prediction, status=prediction['status'], attempts=ledger.attempts(key),
                  recorded_at=now())
    record['accounted_cny'] = sum(attempt['accounted_cny'] for attempt in record['attempts'])
    record['network_seconds'] = sum((json.loads(attempt['cost_detail_json'] or attempt['error_json'] or '{}')).get('elapsed_seconds', 0)
                                    for attempt in record['attempts'])
    ledger.save_cache(key, record)
    return record, False
