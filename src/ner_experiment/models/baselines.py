from collections import Counter, defaultdict
import math
import unicodedata
import numpy as np
import sklearn_crfsuite
from ner_experiment.common import TAGS, bio_to_entities

def prediction(row, tags):
    entities, _ = bio_to_entities(tags)
    return {'id': row['id'], 'status': 'ok', 'entities': entities, 'raw_bio': tags}

class Dictionary:
    def fit(self, rows):
        counts = defaultdict(Counter)
        for row in rows:
            for entity in row['entities']:
                counts[row['text'][entity['start']:entity['end']]][entity['type']] += 1
        self.words = {w: sorted(c, key=lambda label: (-c[label], label))[0] for w, c in counts.items()}
        self.trie = {}
        for word, category in self.words.items():
            node = self.trie
            for char in word: node = node.setdefault(char, {})
            node[None] = category
        return self

    def predict(self, row):
        tags = ['O'] * len(row['tokens']); text = row['text']; i = 0
        while i < len(text):
            node = self.trie; j = i; match = None
            while j < len(text) and text[j] in node:
                node = node[text[j]]; j += 1
                if None in node: match = (j, node[None])
            if match:
                end, category = match; tags[i] = 'B-' + category
                tags[i + 1:end] = ['I-' + category] * (end - i - 1); i = end
            else: i += 1
        return prediction(row, tags)

def legal_masks(tags):
    start = np.array([not t.startswith('I-') for t in tags])
    trans = np.array([[not b.startswith('I-') or a in ('B-' + b[2:], b) for b in tags] for a in tags])
    return start, trans

def viterbi(initial, transition, emissions, constrained=False, tags=TAGS):
    if len(emissions) == 0: return []
    initial, transition = initial.copy(), transition.copy()
    if constrained:
        s, t = legal_masks(tags); initial[~s] = -np.inf; transition[~t] = -np.inf
    score = initial + emissions[0]; back = []
    for emission in emissions[1:]:
        candidates = score[:, None] + transition
        parent = candidates.argmax(axis=0); back.append(parent)
        score = candidates[parent, np.arange(len(score))] + emission
    state = int(score.argmax()); result = [state]
    for parent in reversed(back): state = int(parent[state]); result.append(state)
    return result[::-1]

class HMM:
    def __init__(self, alpha=.1, min_freq=1): self.alpha = alpha; self.min_freq = min_freq

    def fit(self, rows):
        frequencies = Counter(char for row in rows for char in row['tokens'])
        self.vocabulary = {char: i + 1 for i, char in enumerate(sorted(c for c, n in frequencies.items() if n >= self.min_freq))}
        k = len(TAGS); a = self.alpha
        initial = np.full(k, a); transition = np.full((k, k), a)
        emission = np.full((k, len(self.vocabulary) + 1), a)
        index = {tag: i for i, tag in enumerate(TAGS)}
        for row in rows:
            y = [index[tag] for tag in row['bio_tags']]
            if not y: continue
            initial[y[0]] += 1
            for pos, (char, state) in enumerate(zip(row['tokens'], y)):
                emission[state, self.vocabulary.get(char, 0)] += 1
                if pos: transition[y[pos - 1], state] += 1
        self.initial = np.log(initial / initial.sum())
        self.transition = np.log(transition / transition.sum(axis=1, keepdims=True))
        self.emission = np.log(emission / emission.sum(axis=1, keepdims=True))
        return self

    def predict(self, row, constrained=True):
        words = [self.vocabulary.get(c, 0) for c in row['tokens']]
        states = viterbi(self.initial, self.transition, self.emission[:, words].T, constrained)
        return prediction(row, [TAGS[i] for i in states])

def features(row):
    chars = row['tokens']; result = []
    for i, char in enumerate(chars):
        f = {'bias': 1., 'digit': char.isdigit(), 'latin': char.isascii() and char.isalpha(),
             'punct': unicodedata.category(char).startswith('P')}
        for offset in range(-2, 3):
            j = i + offset; f[f'char{offset}'] = chars[j] if 0 <= j < len(chars) else ('<BOS>' if j < 0 else '<EOS>')
        for offset in [-1, 0]:
            j = i + offset
            if 0 <= j < len(chars) - 1: f[f'bigram{offset}'] = ''.join(chars[j:j+2])
        result.append(f)
    return result

class CRF:
    def __init__(self, c1=.1, c2=.1):
        self.model = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=c1, c2=c2, max_iterations=100, all_possible_transitions=True)

    def fit(self, rows):
        self.model.fit([features(r) for r in rows], [r['bio_tags'] for r in rows]); return self

    def predict(self, row): return prediction(row, list(self.model.predict_single(features(row))))
