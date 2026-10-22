"""Training-only surface-form trie with deterministic longest matching."""
from collections import Counter, defaultdict
from ner_experiment.evaluation.metrics import entities_list


class DictionaryNER:
    def fit(self, rows):
        frequencies = defaultdict(Counter)
        for row in rows:
            if row['bio_compatible'] is not True:
                raise ValueError('Only BIO-compatible training rows are allowed')
            for entity in row['entities']:
                surface = row['text'][entity['start']:entity['end']]
                if not surface or surface != entity['text']:
                    raise ValueError('Invalid training entity')
                frequencies[surface][entity['type']] += 1
        self.lexicon_ = {surface: min(counts, key=lambda typ: (-counts[typ], typ))
                         for surface, counts in frequencies.items()}
        self.trie_ = {}
        for surface, typ in sorted(self.lexicon_.items()):
            node = self.trie_
            for char in surface:
                node = node.setdefault(char, {})
            node[None] = typ
        return self

    def predict(self, text):
        spans = set()
        start = 0
        while start < len(text):
            node, position, match = self.trie_, start, None
            while position < len(text) and text[position] in node:
                node = node[text[position]]
                position += 1
                if None in node:
                    match = (start, position, node[None])
            if match is None:
                start += 1
            else:
                spans.add(match)
                start = match[1]
        return entities_list(spans, text)
