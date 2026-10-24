"""Persistent SQLite accounting: every network attempt reserves before send."""
from datetime import datetime
from pathlib import Path
import json
import sqlite3
from zoneinfo import ZoneInfo
from ner_experiment.llm.prompts import canonical


class BudgetStopped(RuntimeError):
    pass


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('Invalid usage integer')
    return value


def usage_cost(response, pricing, started, finished):
    usage = response.get('usage')
    if not isinstance(usage, dict):
        raise ValueError('missing_usage')
    prompt = integer(usage.get('prompt_tokens'))
    output = integer(usage.get('completion_tokens'))
    if 'total_tokens' in usage and usage['total_tokens'] is not None:
        if integer(usage['total_tokens']) != prompt + output:
            raise ValueError('usage_total_mismatch')
    hit = usage.get('prompt_cache_hit_tokens')
    if hit is None:
        hit = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    hit_known = hit is not None
    hit = integer(hit) if hit_known else 0
    if hit > prompt:
        raise ValueError('cache_hit_exceeds_prompt')
    miss = usage.get('prompt_cache_miss_tokens')
    if miss is not None and integer(miss) != prompt - hit:
        raise ValueError('cache_usage_mismatch')
    def peak(timestamp):
        date = datetime.fromisoformat(timestamp).astimezone(ZoneInfo('Asia/Shanghai'))
        return date.weekday() < 5 and any(a <= date.hour < b for a, b in pricing['peak_hours_shanghai'])
    tier = 'peak' if peak(started) or peak(finished) else 'off_peak'
    rates = pricing['per_million_tokens'][tier]
    cost = (hit * rates['input_cache_hit'] + (prompt - hit) * rates['input_cache_miss'] + output * rates['output']) / 1_000_000
    return cost, {'prompt_tokens': prompt, 'completion_tokens': output,
                  'cache_hit_tokens': hit, 'cache_miss_tokens': prompt - hit,
                  'cache_hit_usage_known': hit_known, 'tariff': tier,
                  'cost_is_conservative_if_cache_usage_missing_or_weekday_holiday': not hit_known or tier == 'peak'}


def reserve_cost(request, pricing):
    # Byte-level BPE input bound plus a conservative allowance for chat framing.
    input_bound = len(canonical(request).encode('utf-8')) + 512
    rates = pricing['per_million_tokens']['peak']
    return (input_bound * rates['input_cache_miss'] + request['max_tokens'] * rates['output']) / 1_000_000


class Ledger:
    def __init__(self, path, total_limit=100):
        if not 0 < total_limit <= 100:
            raise ValueError('Total budget must be within authorized 100 CNY')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.limit = total_limit
        self.db.executescript('''
          PRAGMA journal_mode=WAL;
          PRAGMA synchronous=FULL;
          CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, request_key TEXT NOT NULL, attempt_index INTEGER NOT NULL,
            stage TEXT NOT NULL, started TEXT NOT NULL, finished TEXT,
            reserved_cny REAL NOT NULL, accounted_cny REAL NOT NULL,
            state TEXT NOT NULL, response_json TEXT, error_json TEXT, cost_detail_json TEXT,
            UNIQUE(request_key, attempt_index));
          CREATE TABLE IF NOT EXISTS cache (
            request_key TEXT PRIMARY KEY, record_json TEXT NOT NULL, sha256 TEXT NOT NULL);
        ''')

    def close(self):
        self.db.close()

    def spent(self, stage=None):
        sql = 'SELECT COALESCE(SUM(accounted_cny),0) FROM attempts'
        args = ()
        if stage is not None:
            sql += ' WHERE stage=?'
            args = (stage,)
        return float(self.db.execute(sql, args).fetchone()[0])

    def attempts(self, key):
        return [dict(row) for row in self.db.execute('SELECT * FROM attempts WHERE request_key=? ORDER BY attempt_index', (key,))]

    def reserve(self, key, index, stage, amount, stage_limit):
        if amount <= 0:
            raise ValueError('Reservation must be positive')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if self.spent() + amount > self.limit + 1e-12 or self.spent(stage) + amount > stage_limit + 1e-12:
                raise BudgetStopped('budget_limit_before_request')
            if len(self.attempts(key)) != index - 1:
                raise RuntimeError('Attempt already reserved; concurrent or interrupted request')
            cursor = self.db.execute('INSERT INTO attempts (request_key,attempt_index,stage,started,reserved_cny,accounted_cny,state) VALUES (?,?,?,?,?,?,?)',
                                     (key, index, stage, now(), amount, amount, 'reserved'))
            return cursor.lastrowid

    def finish(self, attempt_id, response=None, error=None, cost=None, detail=None):
        with self.db:
            row = self.db.execute('SELECT * FROM attempts WHERE id=?', (attempt_id,)).fetchone()
            if row is None or row['state'] != 'reserved':
                raise RuntimeError('Attempt is not pending')
            accounted = row['reserved_cny'] if cost is None else cost
            self.db.execute('UPDATE attempts SET finished=?,state=?,accounted_cny=?,response_json=?,error_json=?,cost_detail_json=? WHERE id=?',
                            (now(), 'response' if response is not None else 'error', accounted,
                             canonical(response) if response is not None else None,
                             canonical(error) if error is not None else None,
                             canonical(detail) if detail is not None else None, attempt_id))

    def cached(self, key):
        from hashlib import sha256
        row = self.db.execute('SELECT * FROM cache WHERE request_key=?', (key,)).fetchone()
        if row is None:
            return None
        if sha256(row['record_json'].encode('utf-8')).hexdigest() != row['sha256']:
            raise ValueError('Cache checksum mismatch')
        return json.loads(row['record_json'])

    def save_cache(self, key, record):
        from hashlib import sha256
        body = canonical(record)
        with self.db:
            self.db.execute('INSERT INTO cache VALUES (?,?,?)', (key, body, sha256(body.encode('utf-8')).hexdigest()))

    def summary(self):
        rows = [dict(row) for row in self.db.execute('SELECT * FROM attempts ORDER BY id')]
        def uncertain(row):
            return (row['state'] == 'reserved' or
                    bool(json.loads(row['cost_detail_json'] or '{}').get('usage_invalid')) or
                    row['response_json'] is None and row['accounted_cny'] > 0)
        return {'accounted_cny': self.spent(), 'budget_cny': self.limit,
                'attempt_count': len(rows),
                'unknown_or_pending_attempts': sum(uncertain(row) for row in rows),
                'unknown_or_pending_reserved_cny': sum(row['accounted_cny'] for row in rows if uncertain(row)),
                'successful_response_attempts': sum(row['response_json'] is not None for row in rows)}
