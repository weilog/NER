"""Strict span and non-O BIO micro metrics, including explicit failures."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from ner_experiment.common import LABELS, TAGS, bio_to_entities, spans_to_bio

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'precision': p, 'recall': r,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}

def checked_entities(raw, length):
    if not isinstance(raw, list): raise ValueError('entities must be a list')
    valid = set(); seen = set(); invalid = 0; duplicates = 0
    for item in raw:
        if not isinstance(item, dict): invalid += 1; continue
        start, end, category = item.get('start'), item.get('end'), item.get('type')
        # Missing/wrong-typed fields are invalid, not silently discarded.
        if type(start) is not int or type(end) is not int or not isinstance(category, str): invalid += 1; continue
        key = (start, end, category)
        if key in seen: duplicates += 1; continue
        seen.add(key)
        if not 0 <= start < end <= length or category not in LABELS: invalid += 1
        else: valid.add(key)
    return valid, invalid, duplicates

def error_pairs(gold, predicted):
    """Maximum-weight one-to-one: same bounds, then same-type overlap, then other overlap.

    Categories are exclusive. Both wrong boundary and type is a distinct class.
    Sorted spans and deterministic Hungarian solver make ties reproducible.
    """
    g, p = sorted(gold - predicted), sorted(predicted - gold)
    errors = []; used_g = set(); used_p = set()
    if g and p:
        weights = np.zeros((len(g), len(p)))
        base = max(len(g), len(p)) + 1
        for i, a in enumerate(g):
            for j, b in enumerate(p):
                overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
                if not overlap: continue
                tier = base ** 2 if a[:2] == b[:2] else base if a[2] == b[2] else 1
                weights[i, j] = tier + overlap / (max(a[1], b[1]) - min(a[0], b[0])) / base
        gi, pi = linear_sum_assignment(-weights)
        for i, j in zip(gi, pi):
            if weights[i, j] <= 0: continue
            used_g.add(int(i)); used_p.add(int(j))
            kind = 'type' if g[i][:2] == p[j][:2] else 'boundary' if g[i][2] == p[j][2] else 'boundary_and_type'
            errors.append({'kind': kind, 'gold': g[i], 'predicted': p[j]})
    errors += [{'kind': 'missed', 'gold': a, 'predicted': None} for i, a in enumerate(g) if i not in used_g]
    errors += [{'kind': 'spurious', 'gold': None, 'predicted': b} for j, b in enumerate(p) if j not in used_p]
    return errors

def evaluate(gold_rows, predictions):
    if len({r['id'] for r in gold_rows}) != len(gold_rows): raise ValueError('Duplicate gold IDs')
    if len({r['id'] for r in predictions}) != len(predictions): raise ValueError('Duplicate prediction IDs')
    if {r['id'] for r in predictions} != {r['id'] for r in gold_rows}: raise ValueError('Missing or extra prediction IDs')
    pred_map = {r['id']: r for r in predictions}
    entity = Counter(); token = Counter(); per_type = {c: Counter() for c in LABELS}
    per_tag = {c: Counter() for c in TAGS[1:]}; diagnostics = Counter(); cases = []; counts = []
    for row in gold_rows:
        if not row.get('has_labels', True) or row.get('entities') is None: raise ValueError('Unlabeled gold cannot be scored')
        n = len(row['tokens']); g, invalid, duplicate = checked_entities(row['entities'], n)
        if invalid or duplicate: raise ValueError('Invalid gold entities')
        gt, overlaps = spans_to_bio(g, n)
        if overlaps: raise ValueError('Nested gold needs an explicit token evaluation policy')
        record = pred_map[row['id']]; status = record.get('status', 'ok')
        if status not in {'ok', 'api_error', 'format_error'}: raise ValueError('Unknown prediction status')
        if status != 'ok':
            p, inv, dup = set(), 0, 0; diagnostics['failed_samples'] += 1
        else:
            raw = record.get('entities', [])
            if 'raw_bio' in record:
                if len(record['raw_bio']) != n: raise ValueError('BIO length mismatch')
                raw, illegal = bio_to_entities(record['raw_bio']); diagnostics['illegal_bio_positions'] += illegal
                diagnostics['raw_bio_positions'] += n
            p, inv, dup = checked_entities(raw, n)
        pt, projected = spans_to_bio(p, n)
        diagnostics['invalid_entities'] += inv; diagnostics['duplicate_entities'] += dup
        diagnostics['invalid_samples'] += bool(inv); diagnostics['overlap_projection_samples'] += bool(projected)
        diagnostics['projected_out_entities'] += projected
        tp, fp, fn = len(g & p), len(p - g) + inv, len(g - p)
        entity.update(tp=tp, fp=fp, fn=fn)
        for c in LABELS:
            gc, pc = {e for e in g if e[2] == c}, {e for e in p if e[2] == c}
            per_type[c].update(tp=len(gc & pc), fp=len(pc - gc), fn=len(gc - pc))
        # Invalid entities cannot be assigned consistently to the ten valid classes;
        # they are counted in overall strict micro FP and separately documented.
        stp = sfp = sfn = 0
        for a, b in zip(gt, pt):
            diagnostics['characters'] += 1; diagnostics['correct_characters'] += a == b
            if a == b and a != 'O': stp += 1; per_tag[a]['tp'] += 1
            if a != b:
                if b != 'O': sfp += 1; per_tag[b]['fp'] += 1
                if a != 'O': sfn += 1; per_tag[a]['fn'] += 1
        token.update(tp=stp, fp=sfp, fn=sfn)
        errors = error_pairs(g, p)
        errors += [{'kind': 'invalid', 'gold': None, 'predicted': None} for _ in range(inv)]
        counts.append({'id': row['id'], 'entity': [tp, fp, fn], 'token': [stp, sfp, sfn]})
        if errors or status != 'ok': cases.append({'id': row['id'], 'text': row['text'], 'status': status, 'errors': errors,
                                                 'entity': prf(tp, fp, fn), 'token': prf(stp, sfp, sfn)})
    def score(c): return prf(c['tp'], c['fp'], c['fn'])
    classes = {k: score(v) for k, v in per_type.items()}
    result = {'samples': len(gold_rows), 'entity': score(entity), 'token': score(token), 'entity_per_type': classes,
              'token_per_bio_label': {k: score(v) for k, v in per_tag.items()},
              'entity_macro_f1': sum(v['f1'] for v in classes.values()) / len(LABELS),
              'character_accuracy': diagnostics['correct_characters'] / diagnostics['characters'] if diagnostics['characters'] else 0,
              'diagnostics': dict(diagnostics), 'failure_rate': diagnostics['failed_samples'] / len(gold_rows) if gold_rows else 0,
              'invalid_sample_rate': diagnostics['invalid_samples'] / len(gold_rows) if gold_rows else 0,
              'illegal_bio_rate': diagnostics['illegal_bio_positions'] / diagnostics['raw_bio_positions'] if diagnostics['raw_bio_positions'] else 0,
              'error_counts': dict(Counter(e['kind'] for case in cases for e in case['errors'])),
              'protocol': 'normalized BIO, non-O 20-label token micro; exact span/type entity micro; invalid FP only in overall entity micro'}
    return result, counts, cases

def paired_bootstrap(a, b, seed, repeats=2000):
    if [x['id'] for x in a] != [x['id'] for x in b]: raise ValueError('Paired ID order differs')
    if not a: raise ValueError('No samples')
    rng = np.random.default_rng(seed); aa = np.array([x['entity'] for x in a]); bb = np.array([x['entity'] for x in b])
    differences = []
    for _ in range(repeats):
        idx = rng.integers(0, len(a), len(a)); at, bt = aa[idx].sum(0), bb[idx].sum(0)
        differences.append(prf(*at)['f1'] - prf(*bt)['f1'])
    return {'difference_a_minus_b': prf(*aa.sum(0))['f1'] - prf(*bb.sum(0))['f1'],
            'ci95': np.quantile(differences, [.025, .975]).tolist(), 'seed': seed, 'repeats': repeats}
