"""Supervised character HMM with additive smoothing and log Viterbi."""
from collections import Counter
import numpy as np
from ner_experiment.data.splitting import LABELS

STATES = ('O',) + tuple(f'{prefix}-{label}' for prefix in ('B', 'I') for label in LABELS)


def bio_masks(states):
    initial = np.array([not tag.startswith('I-') for tag in states], dtype=bool)
    transitions = np.array([[not dest.startswith('I-') or source in
                            ('B-' + dest[2:], 'I-' + dest[2:])
                            for dest in states] for source in states], dtype=bool)
    return initial, transitions


def viterbi(log_initial, log_transition, emissions, initial_mask=None, transition_mask=None):
    """Return maximum joint path and log score; first state wins exact ties.

    emissions[t, s] is log P(character_t | state_s). Constraints mask paths
    without renormalizing fitted probabilities. There is no EOS probability.
    """
    emissions = np.asarray(emissions, dtype=float)
    if emissions.ndim != 2:
        raise ValueError('emissions must be [positions, states]')
    length, n_states = emissions.shape
    if not length:
        return [], 0.0
    initial = np.asarray(log_initial, dtype=float).copy()
    transitions = np.asarray(log_transition, dtype=float).copy()
    if initial.shape != (n_states,) or transitions.shape != (n_states, n_states):
        raise ValueError('Inconsistent HMM shapes')
    if initial_mask is not None:
        initial[~np.asarray(initial_mask)] = -np.inf
    if transition_mask is not None:
        transitions[~np.asarray(transition_mask)] = -np.inf
    scores = initial + emissions[0]
    back = np.zeros((length, n_states), dtype=np.int32)
    columns = np.arange(n_states)
    for position in range(1, length):
        candidates = scores[:, None] + transitions
        back[position] = candidates.argmax(axis=0)
        scores = candidates[back[position], columns] + emissions[position]
    state = int(scores.argmax())
    score = float(scores[state])
    if not np.isfinite(score):
        raise ValueError('No finite Viterbi path')
    path = [state]
    for position in range(length - 1, 0, -1):
        state = int(back[position, state])
        path.append(state)
    return path[::-1], score


class CharacterHMM:
    def __init__(self, alpha=0.1, min_char_count=1):
        if alpha <= 0 or type(min_char_count) is not int or min_char_count < 1:
            raise ValueError('alpha > 0 and integer min_char_count >= 1 required')
        self.alpha = alpha
        self.min_char_count = min_char_count

    def fit(self, rows):
        rows = list(rows)
        if not rows:
            raise ValueError('Empty training data')
        frequencies = Counter(char for row in rows for char in row['text'])
        # Index 0 is always UNK; no literal string can collide with that index.
        self.vocabulary_ = {char: i + 1 for i, char in enumerate(sorted(
            char for char, count in frequencies.items() if count >= self.min_char_count))}
        self.states_ = STATES
        states = {tag: index for index, tag in enumerate(self.states_)}
        k, v = len(states), len(self.vocabulary_) + 1
        initial = np.full(k, self.alpha)
        transition = np.full((k, k), self.alpha)
        emission = np.full((k, v), self.alpha)
        initial_mask, transition_mask = bio_masks(self.states_)
        for row in rows:
            if row['bio_compatible'] is not True or not row['text'] or len(row['bio_tags']) != len(row['text']):
                raise ValueError('Nonempty, valid BIO training data required')
            tags = np.array([states[tag] for tag in row['bio_tags']], dtype=np.int32)
            if not initial_mask[tags[0]] or not transition_mask[tags[:-1], tags[1:]].all():
                raise ValueError('Illegal gold BIO transition')
            observations = np.array([self.vocabulary_.get(char, 0) for char in row['text']], dtype=np.int32)
            initial[tags[0]] += 1
            np.add.at(transition, (tags[:-1], tags[1:]), 1)
            np.add.at(emission, (tags, observations), 1)
        self.log_initial_ = np.log(initial / initial.sum())
        self.log_transition_ = np.log(transition / transition.sum(axis=1, keepdims=True))
        self.log_emission_ = np.log(emission / emission.sum(axis=1, keepdims=True))
        self.training_characters_ = sum(frequencies.values())
        self.training_unk_characters_ = sum(count for char, count in frequencies.items() if char not in self.vocabulary_)
        return self

    def predict(self, text, constrained=True):
        observations = [self.vocabulary_.get(char, 0) for char in text]
        emissions = self.log_emission_[:, observations].T
        initial_mask, transition_mask = bio_masks(self.states_)
        path, _ = viterbi(self.log_initial_, self.log_transition_, emissions,
                          initial_mask if constrained else None,
                          transition_mask if constrained else None)
        return [self.states_[state] for state in path]

    def vocabulary_statistics(self, texts):
        total, unknown = 0, 0
        for text in texts:
            total += len(text)
            unknown += sum(char not in self.vocabulary_ for char in text)
        return {'vocabulary_characters': len(self.vocabulary_), 'observation_symbols_including_unk': len(self.vocabulary_) + 1,
                'training_characters': self.training_characters_, 'training_unk_characters': self.training_unk_characters_,
                'dev_characters': total, 'dev_unk_characters': unknown,
                'dev_unk_rate': unknown / total if total else 0.0}
