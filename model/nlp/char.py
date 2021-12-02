"""
library of character-based RNN
@Author Chanwoo Gwon, Yonsei Univ. Researcher, since 2020.05. ~
@Date 2020.10.22
"""

import tensorflow as tf
import os
import numpy as np
import random

from model.core import ModelCore, LOSS, Net
from ko.character.spelling import HanJaMo


class CharRNN(ModelCore):
    """
    RNN Library Class

    1. Train and Predict using RNN based on Character
    2. Could be used like the encoder
    3. Can many-to-many or many-to-one RNN
    """
    def __init__(self, data_path, emb=100, loss=LOSS.SPARSE_CATEGORICAL_CROSSENTROPY, is_classify=False):

        self._time_step = 0
        self._text_set = None
        self._emb = emb
        self._char_set = []
        self._char2idx = None
        self._idx2char = None
        self._text_as_int = None
        self._last_dim = 0

        super().__init__(data_path, loss=loss, is_classify=is_classify)

    def read_data(self):
        char_path = os.path.join(self._data_path, 'data.txt')

        self._text_set = open(char_path, 'rb').read().decode(encoding='utf-8')

        self._make_word_world()

    def _make_word_world(self):
        self._set_char()

    def _set_char(self, system_word_list=None):
        self._char_set = sorted(set(self._text_set))

        if system_word_list is not None:
            self._char_set = self._char_set + system_word_list

        self._char2idx = {u: i for i, u in enumerate(self._char_set)}
        self._idx2char = np.array(self._char_set)

    def build_model(self):
        super().build_model()

    def to_vector(self, sequence, add_margin=True):
        if add_margin:
            if len(sequence) < self._time_step:
                sequence = "{0}{1}".format(sequence, ''.join([' '] * (self._time_step - len(sequence))))
        return [self._char2idx[c] for c in sequence]

    def to_text(self, sequence):
        return repr("".join(self._idx2char[sequence]))
