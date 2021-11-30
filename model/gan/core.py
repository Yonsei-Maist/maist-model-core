from abc import ABC

from model.core import ModelCore, Net, LOSS

import tensorflow as tf


class GanCore(ModelCore, ABC):
    def __init__(self, data_path, batch_size=64, avg_list=['loss'], loss=LOSS.CATEGORICAL_CROSSENTROPY, train_test_ratio=0.8, is_classify=False, generator=None, discriminator=None):
        if not isinstance(generator, ModelCore):
            raise ValueError('Generator is not ModelCore')
        if not isinstance(discriminator, ModelCore):
            raise ValueError('Discriminator is not ModelCore')

        super().__init__(data_path=data_path, batch_size=batch_size, avg_list=avg_list, loss=loss, train_test_ratio=train_test_ratio, is_classify=is_classify)

        self._generator = generator
        self._discriminator = discriminator

    def build_model(self):
        input_gen = self._generator.model.input
        output_gen = self._generator.model.output
        output_dis = self._discriminator.model(output_gen)

        self.model = tf.keras.Model(inputs=[input_gen], outputs=[output_dis])


class GanNetwork(Net):
    def train(self, pretrained_module_name=None, pretrained_module_index=None, epoch=10000, lr=0.001):
        pass

    def test(self, index):
        pass
