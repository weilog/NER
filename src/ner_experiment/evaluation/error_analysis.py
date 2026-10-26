"""Fixed one-to-one attribution, matching the preregistered Dev audit rule."""
from collections import Counter


def attribute(gold, predicted):
    gold_left, pred_left = set(gold - predicted), set(predicted - gold)
    candidates = []
    for g in sorted(gold_left):
        for p in sorted(pred_left):
            intersection = min(g[1], p[1]) - max(g[0], p[0])
            if intersection > 0:
                iou = intersection / (max(g[1], p[1]) - min(g[0], p[0]))
                candidates.append((-(g[:2] == p[:2]), -(g[2] == p[2]), -iou, g, p))
    errors = []
    for _, _, _, g, p in sorted(candidates):
        if g not in gold_left or p not in pred_left:
            continue
        kind = 'type' if g[:2] == p[:2] else 'boundary' if g[2] == p[2] else 'boundary_and_type'
        errors.append({'category': kind, 'gold': g, 'prediction': p})
        gold_left.remove(g); pred_left.remove(p)
    errors += [{'category': 'missed', 'gold': g, 'prediction': None} for g in sorted(gold_left)]
    errors += [{'category': 'spurious', 'gold': None, 'prediction': p} for p in sorted(pred_left)]
    return errors


def character_errors(gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError('Character sequence length mismatch')
    result = Counter()
    for g,p in zip(gold,predicted):
        if g == p:
            continue
        key = ('token_missed' if p == 'O' else 'token_spurious' if g == 'O'
               else 'token_bi_confusion' if g[2:] == p[2:] else 'token_type_confusion')
        result[key] += 1
    return result
