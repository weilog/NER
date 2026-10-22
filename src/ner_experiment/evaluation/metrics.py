"""One canonical non-overlapping prediction set for both primary metrics.

Raw spans are scored only as a separate diagnostic. No gold-aware repair is
allowed. Character offsets count Python Unicode code points, without trimming
or Unicode normalization of the source string.
"""
from collections import Counter
from ner_experiment.data.splitting import LABELS

PROTOCOL = 'dual-evaluation-v2'
FAILURE_STATUSES = {'api_error', 'parse_error', 'budget_stopped', 'error'}
BIO_LABELS = {'O'} | {f'{prefix}-{label}' for prefix in ('B', 'I') for label in LABELS}


class PredictionFileError(ValueError):
    pass


def prf(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def span_set(entities):
    return {(e['start'], e['end'], e['type']) for e in entities}


def entities_list(spans, text):
    return [{'start': start, 'end': end, 'type': typ, 'text': text[start:end]}
            for start, end, typ in sorted(spans)]


def bio_to_spans(tags, repair=True):
    """Repair orphan/type-changing I as B; strict mode drops orphan I runs.

    In strict mode only a B opens an entity. Following I of the same type
    extends it. An orphan I never opens an entity, including mid-run.
    """
    if any(not isinstance(tag, str) or tag not in BIO_LABELS for tag in tags):
        raise PredictionFileError('Unknown or malformed BIO label')
    illegal = 0
    previous = 'O'
    normalized = []
    for tag in tags:
        bad = tag.startswith('I-') and previous not in ('B-' + tag[2:], 'I-' + tag[2:])
        illegal += int(bad)
        normalized.append('B-' + tag[2:] if bad and repair else tag)
        previous = tag
    spans = set()
    i = 0
    while i < len(normalized):
        tag = normalized[i]
        if not tag.startswith('B-'):
            i += 1
            continue
        j = i + 1
        while j < len(normalized) and normalized[j] == 'I-' + tag[2:]:
            j += 1
        spans.add((i, j, tag[2:]))
        i = j
    return spans, normalized, illegal


def spans_to_bio(spans, length):
    tags = ['O'] * length
    for start, end, typ in sorted(spans):
        if not (0 <= start < end <= length and typ in LABELS):
            raise ValueError('Cannot project invalid span')
        if any(tag != 'O' for tag in tags[start:end]):
            raise ValueError('Cannot project overlapping spans')
        tags[start] = 'B-' + typ
        tags[start + 1:end] = ['I-' + typ] * (end - start - 1)
    return tags


def normalize_prediction(prediction, text):
    status = prediction.get('status')
    if status not in {'ok'} | FAILURE_STATUSES:
        raise PredictionFileError('Explicit recognized status required')
    diagnostics = Counter()
    events = []
    if status != 'ok':
        diagnostics['failed_samples'] = 1
        return {'spans': set(), 'bio': ['O'] * len(text), 'raw_spans': set(),
                'raw_bio': None, 'strict_bio_spans': None,
                'diagnostics': diagnostics, 'events': [{'reason': status}]}
    raw_bio = prediction.get('raw_bio_tags')
    strict_spans = None
    if raw_bio is not None:
        if not isinstance(raw_bio, list) or len(raw_bio) != len(text):
            raise PredictionFileError('BIO sequence must match character count')
        derived, _, illegal = bio_to_spans(raw_bio)
        strict_spans, _, _ = bio_to_spans(raw_bio, repair=False)
        diagnostics.update(bio_samples=1, bio_positions=len(raw_bio), illegal_bio_transitions=illegal)
        if 'entities' in prediction:
            supplied = prediction['entities']
            if not isinstance(supplied, list):
                raise PredictionFileError('entities must be a list')
            try:
                agrees = span_set(supplied) == derived
            except (TypeError, KeyError):
                agrees = False
            if not agrees:
                raise PredictionFileError('Entities disagree with normalized raw BIO')
        entries = entities_list(derived, text)
    else:
        entries = prediction.get('entities')
        if not isinstance(entries, list):
            raise PredictionFileError('Successful prediction requires entities list or raw BIO')
    diagnostics['entity_entries'] = len(entries)
    raw_spans, seen, valid = set(), set(), set()
    for index, entry in enumerate(entries):
        reason = None
        if not isinstance(entry, dict) or type(entry.get('start')) is not int or type(entry.get('end')) is not int or not isinstance(entry.get('type'), str):
            reason = 'malformed_entity'
        else:
            start, end, typ = entry['start'], entry['end'], entry['type']
            span = (start, end, typ)
            raw_spans.add(span)  # Includes structural but invalid spans as raw FP.
            if span in seen:
                diagnostics['duplicate_entities'] += 1
                events.append({'index': index, 'reason': 'duplicate_entity', 'entity': entry})
            seen.add(span)
            if not 0 <= start < end <= len(text):
                reason = 'out_of_bounds'
            elif typ not in LABELS:
                reason = 'unknown_type'
            elif 'text' in entry and entry['text'] != text[start:end]:
                reason = 'text_position_mismatch'
            else:
                valid.add(span)
        if reason:
            diagnostics['invalid_entities'] += 1
            diagnostics[reason] += 1
            events.append({'index': index, 'reason': reason, 'entity': entry})
    kept, end_of_previous = set(), -1
    for start, end, typ in sorted(valid, key=lambda span: (span[0], -(span[1] - span[0]), span[2])):
        if start < end_of_previous:
            diagnostics['overlap_entities_removed'] += 1
            events.append({'reason': 'overlap_removed', 'entity': {'start': start, 'end': end, 'type': typ}})
        else:
            kept.add((start, end, typ))
            end_of_previous = end
    if diagnostics['invalid_entities']:
        diagnostics['invalid_samples'] = 1
    if diagnostics['overlap_entities_removed']:
        diagnostics['overlap_affected_samples'] = 1
    return {'spans': kept, 'bio': spans_to_bio(kept, len(text)),
            'raw_spans': raw_spans, 'raw_bio': raw_bio, 'strict_bio_spans': strict_spans,
            'diagnostics': diagnostics, 'events': events}


def counts(gold, predicted):
    return len(gold & predicted), len(predicted - gold), len(gold - predicted)


def token_counts(gold, predicted):
    if len(gold) != len(predicted):
        raise PredictionFileError('Character lengths differ')
    return (sum(g == p and g != 'O' for g, p in zip(gold, predicted)),
            sum(p != 'O' and g != p for g, p in zip(gold, predicted)),
            sum(g != 'O' and g != p for g, p in zip(gold, predicted)))


def evaluate(gold_rows, predictions):
    gold_rows, predictions = list(gold_rows), list(predictions)
    def index(rows, label):
        result = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('id'), str):
                raise PredictionFileError(f'{label}: string ID required')
            if row['id'] in result:
                raise PredictionFileError(f'{label}: duplicate ID {row["id"]}')
            result[row['id']] = row
        return result
    gold_index, pred_index = index(gold_rows, 'gold'), index(predictions, 'prediction')
    if gold_index.keys() != pred_index.keys():
        raise PredictionFileError(f'ID mismatch: missing={sorted(gold_index.keys()-pred_index.keys())[:10]}, extra={sorted(pred_index.keys()-gold_index.keys())[:10]}')
    totals = {name: [0, 0, 0] for name in ('token', 'entity', 'raw_entity', 'raw_bio_token', 'strict_bio_entity')}
    per_type = {label: [0, 0, 0] for label in LABELS}
    diagnostics = Counter(samples=len(gold_rows), characters=0, correct_characters=0)
    sentence_counts, normalized_rows = [], []
    for row in gold_rows:
        if row.get('has_labels') is not True or row.get('bio_compatible') is not True:
            raise ValueError(f'Gold is unlabeled or overlapping: {row["id"]}')
        gold_spans = span_set(row['entities'])
        gold_bio = spans_to_bio(gold_spans, len(row['text']))
        if row['tokens'] != list(row['text']) or row['bio_tags'] != gold_bio:
            raise ValueError('Gold tokens/BIO inconsistent with spans')
        for entity in row['entities']:
            if row['text'][entity['start']:entity['end']] != entity['text']:
                raise ValueError('Gold entity text mismatch')
        norm = normalize_prediction(pred_index[row['id']], row['text'])
        values = {'token': token_counts(gold_bio, norm['bio']),
                  'entity': counts(gold_spans, norm['spans']),
                  'raw_entity': counts(gold_spans, norm['raw_spans'])}
        if norm['raw_bio'] is not None:
            values['raw_bio_token'] = token_counts(gold_bio, norm['raw_bio'])
            values['strict_bio_entity'] = counts(gold_spans, norm['strict_bio_spans'])
        for name, triple in values.items():
            for i, value in enumerate(triple):
                totals[name][i] += value
        for label in LABELS:
            triple = counts({span for span in gold_spans if span[2] == label},
                            {span for span in norm['spans'] if span[2] == label})
            for i, value in enumerate(triple):
                per_type[label][i] += value
        diagnostics.update(norm['diagnostics'])
        diagnostics['characters'] += len(gold_bio)
        diagnostics['correct_characters'] += sum(g == p for g, p in zip(gold_bio, norm['bio']))
        sentence_counts.append({'id': row['id'], 'token': list(values['token']), 'entity': list(values['entity'])})
        normalized_rows.append({'id': row['id'], 'status': pred_index[row['id']]['status'],
                                'entities': entities_list(norm['spans'], row['text']), 'events': norm['events']})
    by_type = {label: prf(*triple) for label, triple in per_type.items()}
    def rate(numerator, denominator):
        return diagnostics[numerator] / diagnostics[denominator] if diagnostics[denominator] else 0.0
    return {'protocol': PROTOCOL, 'primary_prediction_set': 'validated-deduplicated-nonoverlapping',
            'token': prf(*totals['token']), 'entity': prf(*totals['entity']),
            'per_type_entity': by_type,
            'entity_macro_f1_fixed_10': sum(score['f1'] for score in by_type.values()) / len(LABELS),
            'character_accuracy_including_O': rate('correct_characters', 'characters'),
            'diagnostics': dict(diagnostics),
            'rates': {'failure_rate': rate('failed_samples', 'samples'),
                      'invalid_entity_rate': rate('invalid_entities', 'entity_entries'),
                      'invalid_sample_rate': rate('invalid_samples', 'samples'),
                      'illegal_bio_transition_rate_before_repair': rate('illegal_bio_transitions', 'bio_positions'),
                      'overlap_affected_sample_rate': rate('overlap_affected_samples', 'samples')},
            'sensitivity': {'raw_span_entity': prf(*totals['raw_entity']),
                            'raw_bio_token_on_bio_samples_only': prf(*totals['raw_bio_token']),
                            'strict_bio_entity_on_bio_samples_only': prf(*totals['strict_bio_entity']),
                            'bio_samples': diagnostics['bio_samples'],
                            'strict_rule': 'Only B opens; orphan I ignored; same-type I extends'},
            'sentence_counts': sentence_counts, 'normalized_predictions': normalized_rows}
