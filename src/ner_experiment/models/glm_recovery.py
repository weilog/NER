"""Audited network recovery. Preserve the frozen client, requests and raw ledger."""
import socket
import ssl
import urllib.error
import math

from ner_experiment.common import load_json, object_hash, utc_now
from ner_experiment.models.glm import GLMClient, InterfaceStop


def validate_reconciliation(ledger, record):
    count = record['prefix_attempts']
    if type(count) is not int or count < 0 or count > len(ledger['attempts']):
        raise ValueError('Invalid reconciled ledger prefix')
    prefix = ledger['attempts'][:count]
    if object_hash(prefix) != record['prefix_sha256']:
        raise ValueError('Reconciled ledger prefix changed')
    known = sum(a['charged_upper_cny'] for a in prefix if a['status'] == 'response')
    charge = record['confirmed_charge_upper_cny']
    if type(charge) not in (int, float) or not math.isfinite(charge) or charge < known:
        raise ValueError('Reconciliation cannot go below recorded known usage charges')
    return record


def reconciled_total(ledger, record):
    if record is None:
        return sum(a['charged_upper_cny'] for a in ledger['attempts'])
    return record['confirmed_charge_upper_cny'] + sum(
        a['charged_upper_cny'] for a in ledger['attempts'][record['prefix_attempts']:])


class RecoveryGLMClient(GLMClient):
    """Same payload/parser/cache; eight consecutive connection failures stop the batch."""
    def __init__(self, folder, config, message_builder, budget, **kwargs):
        super().__init__(folder, config, message_builder, budget, **kwargs)
        path = self.folder / 'billing_reconciliation.json'
        self.reconciliation = validate_reconciliation(self.ledger, load_json(path)) if path.exists() else None
        self.consecutive_connection_failures = 0
        original_transport = self.transport
        self.transport = lambda payload: self.checked_transport(original_transport, payload)

    def total(self):
        with self.lock:
            return reconciled_total(self.ledger, self.reconciliation)

    def persist(self):
        # Only newly diagnosed failures proven to precede HTTP sending can release a reservation.
        # Historical URLError entries have no such evidence and remain unchanged.
        for entry in self.ledger['attempts']:
            if entry.get('verified_not_sent') and entry.get('reservation_released_cny') is None:
                entry['reservation_released_cny'] = entry['charged_upper_cny']
                entry['charged_upper_cny'] = 0
        super().persist()

    def checked_transport(self, transport, payload):
        key = object_hash({'endpoint': self.settings['endpoint'], 'payload': payload})
        try:
            raw = transport(payload)
        except urllib.error.HTTPError:
            # HTTP proves a working connection; the existing client handles business errors/retries.
            with self.lock:
                self.consecutive_connection_failures = 0
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            safe_message = str(reason)
            if self.settings['key']:
                safe_message = safe_message.replace(self.settings['key'], '[REDACTED]')
            not_sent = isinstance(reason, (socket.gaierror, ConnectionRefusedError, ssl.SSLCertVerificationError))
            with self.lock:
                self.consecutive_connection_failures += 1
                pending = [a for a in self.ledger['attempts'] if a['key'] == key and a['status'] == 'reserved']
                if pending:
                    pending[-1]['network_diagnostic'] = {
                        'exception': type(exc).__name__, 'reason_type': type(reason).__name__,
                        'errno': getattr(reason, 'errno', None), 'winerror': getattr(reason, 'winerror', None),
                        'message': safe_message[:1000], 'observed_at': utc_now(),
                    }
                    if not_sent:
                        pending[-1]['verified_not_sent'] = True
                if not_sent or self.consecutive_connection_failures >= 8:
                    self.stop_event.set()
                    self.ledger['connection_stop'] = {
                        'at': utc_now(), 'consecutive_failed_attempts': self.consecutive_connection_failures,
                        'reason': 'pre_send_connection_error' if not_sent else 'eight_consecutive_connection_failures',
                    }
                self.persist()
            raise
        else:
            with self.lock:
                self.consecutive_connection_failures = 0
            return raw
