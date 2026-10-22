"""CPU linear-chain CRF with fixed character/context feature template."""
import unicodedata
from sklearn_crfsuite import CRF

FEATURE_VERSION = 'char-context-2-local-bigrams-v1'


def character_features(char, prefix):
    return {prefix + 'char': char, prefix + 'lower': char.lower(),
            prefix + 'isdigit': char.isdigit(), prefix + 'isalpha': char.isalpha(),
            prefix + 'isascii_alpha': char.isascii() and char.isalpha(),
            prefix + 'ispunct': unicodedata.category(char).startswith('P'),
            prefix + 'category': unicodedata.category(char)}


def sentence_features(text):
    result = []
    for i in range(len(text)):
        features = {'bias': 1.0}
        for offset in (-2, -1, 0, 1, 2):
            j = i + offset
            prefix = f'{offset:+d}:'
            if 0 <= j < len(text):
                features.update(character_features(text[j], prefix))
            else:
                features[prefix + ('BOS' if j < 0 else 'EOS')] = True
        for offset in (-2, -1, 0, 1):
            j = i + offset
            if 0 <= j < len(text) - 1:
                features[f'bigram{offset:+d}'] = text[j:j + 2]
        result.append(features)
    return result


class CharacterCRF:
    def __init__(self, c1=0.1, c2=0.1, max_iterations=100):
        self.c1, self.c2, self.max_iterations = c1, c2, max_iterations

    def fit_features(self, features, tags):
        self.model_ = CRF(algorithm='lbfgs', c1=self.c1, c2=self.c2,
                          max_iterations=self.max_iterations,
                          all_possible_transitions=True, verbose=False)
        self.model_.fit(features, tags)
        return self

    def predict(self, text):
        if not text:
            return []
        return list(self.model_.predict_single(sentence_features(text)))
