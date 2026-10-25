"""Coordinate-independent surface diagnostics, never a replacement main score."""
from collections import Counter
from ner_experiment.data.splitting import LABELS
from ner_experiment.evaluation.metrics import normalize_prediction, prf


def occurrences(text, surface):
    if not surface:
        return []
    starts, cursor = [], 0
    while True:
        found = text.find(surface, cursor)
        if found < 0:
            return starts
        starts.append(found)
        cursor = found + 1


def diagnose(gold_rows, predictions):
    mapping = {row['id']: row for row in predictions}
    if len(mapping) != len(predictions) or set(mapping) != {r['id'] for r in gold_rows}:
        raise ValueError('Diagnostic IDs must match exactly')
    totals, sentence_rows, events = Counter(), [], []
    surface_total = [0, 0, 0]
    for gold in gold_rows:
        prediction = mapping[gold['id']]
        norm = normalize_prediction(prediction, gold['text'])
        reasons = {event['reason'] for event in norm['events']}
        position_error = bool(reasons & {'out_of_bounds', 'text_position_mismatch'})
        totals['position_error_samples'] += position_error
        totals['samples'] += 1
        truth = Counter((e['text'], e['type']) for e in gold['entities'])
        proposed, seen = Counter(), set()
        for entry in prediction.get('entities', []) if prediction['status'] == 'ok' else []:
            totals['raw_entity_entries'] += 1
            if not isinstance(entry, dict) or not isinstance(entry.get('text'), str) or not entry['text'] or not isinstance(entry.get('type'), str):
                totals['surface_unscorable_entries'] += 1
                continue
            # Do not credit exact duplicate output, but retain multiplicity for
            # distinct coordinates of a repeated name. This bag diagnostic
            # deliberately cannot distinguish which occurrence was intended.
            key = (str(entry.get('start')), str(entry.get('end')), entry['type'], entry['text'])
            if key in seen:
                continue
            seen.add(key)
            proposed[(entry['text'], entry['type'])] += 1
            start, end, surface = entry.get('start'), entry.get('end'), entry['text']
            if type(start) is not int or type(end) is not int:
                continue
            invalid_position = not 0 <= start < end <= len(gold['text']) or gold['text'][start:end] != surface
            if invalid_position and entry['type'] in LABELS:
                candidates = occurrences(gold['text'], surface)
                inclusive_bug = (0 <= start <= end < len(gold['text']) and gold['text'][start:end + 1] == surface)
                totals['inclusive_end_signature_entities'] += inclusive_bug
                totals['unique_surface_relocatable_entities'] += len(candidates) == 1
                events.append({'id': gold['id'], 'text': gold['text'], 'entity': entry,
                               'surface_occurrences': candidates,
                               'inclusive_end_signature': inclusive_bug,
                               'unique_surface_relocatable': len(candidates) == 1})
        tp = sum((truth & proposed).values())
        triple = [tp, sum(proposed.values()) - tp, sum(truth.values()) - tp]
        for i, value in enumerate(triple):
            surface_total[i] += value
        sentence_rows.append({'id': gold['id'], 'position_error': int(position_error),
                              'surface': triple})
    n = totals['samples']
    return {'position_error_sample_rate': totals['position_error_samples'] / n if n else 0.0,
            'surface_type_bag': prf(*surface_total), 'counts': dict(totals),
            'sentence_counts': sentence_rows, 'position_events': events,
            'scope': 'Exact entity text+type with multiplicity, ignores coordinates; duplicates deduplicated by raw coordinates/text/type. Repeated mentions remain ambiguous; not a primary NER metric.'}
