import math
import os
from abc import ABC

from model.core import ModelCore, Net, LOSS, Dataset, DatasetFactory

import tensorflow as tf
import numpy as np


class GanCore(ModelCore, ABC):
    def __init__(self, data_path, batch_size=64, avg_list=['loss'], loss=LOSS.CATEGORICAL_CROSSENTROPY, train_test_ratio=1, is_classify=False,
                 generator: ModelCore=None, discriminator: ModelCore=None, latent_space_size=100):
        if not isinstance(generator, ModelCore):
            raise ValueError('Generator is not ModelCore')
        if not isinstance(discriminator, ModelCore):
            raise ValueError('Discriminator is not ModelCore')

        self.generator = generator
        self.discriminator = discriminator
        self.latent_space_size = latent_space_size
        self._data_fake_seed = None

        self._train_data_real = Dataset(int(batch_size / 2), self)
        self._train_data_fake_seed = Dataset(int(batch_size / 2), self)
        self._test_data_real = Dataset(int(batch_size / 2), self)
        self._test_data_fake_seed = Dataset(int(batch_size / 2), self)

        super().__init__(data_path=data_path, batch_size=batch_size, avg_list=avg_list, loss=loss, train_test_ratio=train_test_ratio, is_classify=is_classify)

    def get_train_data_real(self):
        return self._train_data_real

    def get_train_data_fake_seed(self):
        return self._train_data_fake_seed

    def get_test_data_real(self):
        return self._test_data_real

    def get_test_data_fake_seed(self):
        return self._test_data_fake_seed

    def build_model(self):
        out = self.discriminator.model(self.generator.model.output, training=False)
        self.model = tf.keras.Model(inputs=[self.generator.model.input], outputs=[out])

    def make_dataset(self):
        if self._data_all is not None and len(self._data_all) > 0:
            self._train_data_real, self._test_data_real = DatasetFactory.make_dataset(self._train_data_real, self._test_data_real, self._data_all, self._train_test_ratio, self._is_classify)

        if self._data_fake_seed is not None and len(self._data_fake_seed) > 0:
            self._train_data_fake_seed, self._test_data_fake_seed = DatasetFactory.make_dataset(self._train_data_fake_seed, self._test_data_fake_seed, self._data_fake_seed, self._train_test_ratio, self._is_classify)


class GanNetwork(Net):

    def __init__(self, module_name, base_path, model_core: GanCore):
        """
        :param model_core: ModelCore instance
        """

        if not isinstance(model_core, GanCore):
            raise ValueError('core is not GanCore')

        super().__init__(module_name, base_path, model_core)
        self._model_core = model_core

    def create_fake_seed(self, count):
        return np.random.normal(0, 1, (count, self._model_core.latent_space_size)), np.array([[0, 1] for i in range(count)])

    def generate_fake_seed(self):
        real = self._model_core.get_train_data_real()
        iter = math.ceil(len(real) / real.batch_size)

        for i in range(iter):
            count = real.batch_size
            if i == iter - 1 and len(real) % real.batch_size != 0:
                count = len(real) % real.batch_size

            yield self.create_fake_seed(count)

    def _train_discriminator(self, optimizer, real_data, fake_data, real_label, fake_label):
        data_inputs = np.concatenate([real_data[0], fake_data])
        data_labels = np.concatenate([real_label, fake_label])

        with tf.GradientTape() as tape:
            discriminator_output = self._model_core.discriminator.model([data_inputs], training=True)

            discriminator_loss = self._model_core.discriminator.calculate_loss_function(discriminator_output, data_labels, axis=1)

        grads = tape.gradient(discriminator_loss,
                              self._model_core.discriminator.model.trainable_variables)  # calculate gradients
        optimizer.apply_gradients(
            zip(grads, self._model_core.discriminator.model.trainable_variables))  # update gradients

        return discriminator_loss

    def _train_generator(self, optimizer, real_data, fake_data_seed, real_label, fake_label):
        with tf.GradientTape() as tape:
            gan_output = self._model_core.model(fake_data_seed, training=True)
            gan_loss = self._model_core.calculate_loss_function(gan_output, real_label, axis=1)

        grads = tape.gradient(gan_loss, self._model_core.model.trainable_variables)  # calculate gradients
        optimizer.apply_gradients(zip(grads, self._model_core.model.trainable_variables))  # update gradients

        return gan_loss

    def save_when(self, epoch, result_values):
        return epoch != 0 and epoch % 10 == 0

    def train(self, pretrained_module_name=None, pretrained_module_index=None, epoch=10000, lr=0.001):

        train_data_real = self._model_core.get_train_data_real()
        train_data_fake_seed = self._model_core.get_train_data_fake_seed()

        optimizer_discriminator = tf.keras.optimizers.Adam(lr=lr)
        optimizer_generator = tf.keras.optimizers.Adam(lr=lr)

        for i in range(epoch):
            self._model_core.avg_logger.refresh()
            gen_real = list(train_data_real.get())
            if train_data_fake_seed is not None and len(train_data_fake_seed) > 0:
                gen_fake_seed = list(train_data_fake_seed.get())
            else:
                gen_fake_seed = list(self.generate_fake_seed())

            for data_real, data_fake_seed in zip(gen_real, gen_fake_seed):

                inputs_real = data_real[0]
                labels_real = data_real[1]

                inputs_fake_seed = data_fake_seed[0]
                labels_fake = data_fake_seed[1]

                # train discriminator
                inputs_fake = self._model_core.generator.model(inputs_fake_seed.copy())

                discriminator_loss = self._train_discriminator(optimizer_discriminator, inputs_real, inputs_fake, labels_real, labels_fake)

                gan_loss = self._train_generator(optimizer_generator, inputs_real, inputs_fake_seed, labels_real, labels_fake)

                if isinstance(gan_loss, list):
                    self._model_core.avg_logger.update_state(gan_loss)
                else:
                    self._model_core.avg_logger.update_state([gan_loss])

            log_result = self._model_core.avg_logger.result()
            print('Epoch: {} {}'.format(i, log_result))

            # save weight every 100 epochs
            if (i % 100 == 0 and i != 0) or self.save_when(i, self._model_core.avg_logger.result_value()):
                self._model_core.model.save_weights(os.path.join(self._base_path,
                                                './checkpoints/{}_{}.tf'.format(self.name, i)))

    def test(self, index):
        if index > -1:
            self._model_core.model.load_weights(os.path.join(self._base_path,
                                                             './checkpoints/{}_{}.tf'.format(self.name, index)))
        else:
            self._model_core.load_weight()
        self._model_core.avg_logger.refresh()
        fake_seed, fake_label = self.create_fake_seed(16)
        gen_list = self._model_core.generator.model(fake_seed, training=False)
        output = self._model_core.discriminator.model(gen_list, training=False)
        loss = self._model_core.discriminator.calculate_loss_function(output, tf.convert_to_tensor(fake_label, dtype=tf.float32), axis=1)

        if isinstance(loss, list):
            self._model_core.avg_logger.update_state(loss)
        else:
            self._model_core.avg_logger.update_state([loss])

        log_result = self._model_core.avg_logger.result_value()

        return log_result, gen_list

    def predict(self, index, data):
        if self._model_core.model is None:
            self._model_core.build_model()

            if index > -1:
                self._model_core.model.load_weights(os.path.join(self._base_path,
                                                                 './checkpoints/{}_{}.tf'.format(self.name, index)))
            else:
                self._model_core.load_weight()
        pre_data = data
        if data is None:
            seed, label = self.generate_fake_seed()
            pre_data = seed[:1]

        return self._model_core.generator.model([pre_data], training=False)
