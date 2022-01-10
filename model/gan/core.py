import math
import os
from abc import ABC, abstractmethod

from PIL import Image

from model.core import ModelCore, LOSS, Dataset, DatasetFactory, AvgLogger

import tensorflow as tf
import numpy as np


class GanCore(ModelCore, ABC):

    @staticmethod
    def noisy_labels(y, p_flip):
        # determine the number of labels to flip
        n_select = int(p_flip * int(y.shape[0]))
        # choose labels to flip
        flip_ix = np.random.choice([i for i in range(int(y.shape[0]))], size=n_select)

        op_list = []
        # invert the labels in place
        # y_np[flip_ix] = 1 - y_np[flip_ix]
        for i in range(int(y.shape[0])):
            if i in flip_ix:
                op_list.append(tf.subtract(1, y[i]))
            else:
                op_list.append(y[i])

        outputs = tf.stack(op_list)
        return tf.cast(outputs, dtype=tf.float32)

    @staticmethod
    def smooth_positive_labels(y):
        return y - 0.3 + (np.random.random(y.shape) * 0.5)

    @staticmethod
    def smooth_negative_labels(y):
        return y + np.random.random(y.shape) * 0.3

    @staticmethod
    def is_fake(label):
        return label[0][0] == 0

    def __init__(self, data_path, save_path, batch_size=64, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), train_test_ratio=1, is_classify=False,
                 generator: ModelCore=None, discriminator: ModelCore=None, latent_space_size=100, flip_coin=False, metrics=None, optimizer=None, loss_weights=None):
        if not isinstance(generator, ModelCore):
            raise ValueError('Generator is not ModelCore')
        if not isinstance(discriminator, ModelCore):
            raise ValueError('Discriminator is not ModelCore')

        self.generator = generator
        self.discriminator = discriminator
        self.latent_space_size = latent_space_size
        self._data_fake_seed = None
        self.is_flip_coin = flip_coin

        if self.is_flip_coin:
            data_batch = batch_size
        else:
            data_batch = batch_size / 2

        self._train_data_real = Dataset(int(data_batch), self)
        self._train_data_fake_seed = Dataset(int(data_batch), self)
        self._test_data_real = Dataset(int(data_batch), self)
        self._test_data_fake_seed = Dataset(int(data_batch), self)

        super().__init__(data_path=data_path, save_path=save_path, batch_size=batch_size, loss=loss,
                         train_test_ratio=train_test_ratio, validation_ratio=0, is_classify=is_classify,
                         metrics=metrics, optimizer=optimizer, loss_weights=loss_weights)

    def get_train_data_real(self):
        return self._train_data_real

    def get_train_data_fake_seed(self):
        return self._train_data_fake_seed

    def get_test_data_real(self):
        return self._test_data_real

    def get_test_data_fake_seed(self):
        return self._test_data_fake_seed

    def build_model(self):
        input_ = self.generator.model.input
        out = self.generator.model.output
        out = self.discriminator.model(out, training=True)
        self.model = tf.keras.Model(inputs=input_, outputs=out)

    def make_dataset(self):
        if self._data_all is not None and len(self._data_all) > 0:
            self._train_data_real, self._test_data_real = DatasetFactory.make_dataset(self._train_data_real, self._test_data_real, self._data_all, self._train_test_ratio, self._is_classify)

        if self._data_fake_seed is not None and len(self._data_fake_seed) > 0:
            self._train_data_fake_seed, self._test_data_fake_seed = DatasetFactory.make_dataset(self._train_data_fake_seed, self._test_data_fake_seed, self._data_fake_seed, self._train_test_ratio, self._is_classify)

    def create_fake_seed(self, count):
        return np.random.normal(0, 1, (count, self.latent_space_size)), np.array([[0] for i in range(count)])

    def flip_coin(self, chance=0.5):
        return np.random.binomial(1, chance)

    def data_getter(self):
        """
        make data to train
        :return (train_data_for_discriminator, label_data_for_discriminator,
            fake_seed_for_discriminator, label_for_discriminator, fake_seed_for_generator, label_for_generator)
        """
        train_data_real = self.get_train_data_real()
        train_data_fake_seed = self.get_train_data_fake_seed()

        gen_real = list(train_data_real.get())

        if train_data_fake_seed is not None and len(train_data_fake_seed) > 0:
            gen_fake_seed = list(train_data_fake_seed.get())
        else:
            gen_fake_seed = None

        real_idx = 0

        while real_idx < len(gen_real):
            if self.is_flip_coin:
                data = gen_real[real_idx]
                len_data = len(data[0][0])
                fake_seed_gen = gen_fake_seed[real_idx] if gen_fake_seed is not None else self.create_fake_seed(len_data)
                if self.flip_coin(0.9):
                    fake_label_gen = data[1]
                else:
                    fake_label_gen = fake_seed_gen[1]

                if self.flip_coin():
                    # return real only
                    yield data[0], data[1], [np.array([]).reshape([0] + list(data[0][0].shape[1:]))], np.array([]).reshape([0] + list(data[1].shape[1:])), \
                          fake_seed_gen[0], fake_label_gen

                    real_idx = real_idx + 1
                else:
                    # return fake only
                    fake_seed = gen_fake_seed[real_idx] if gen_fake_seed is not None else self.create_fake_seed(len_data)
                    yield [np.array([]).reshape([0] + list(data[0][0].shape[1:]))], np.array([]).reshape([0] + list(data[1].shape[1:])), \
                          fake_seed[0], fake_seed[1], fake_seed_gen[0], fake_label_gen
            else:
                data = gen_real[real_idx]
                len_data = len(data[0][0])
                fake_seed = gen_fake_seed[real_idx] if gen_fake_seed is not None else self.create_fake_seed(len_data)
                fake_seed_gen = gen_fake_seed[real_idx] if gen_fake_seed is not None else self.create_fake_seed(len_data)

                yield data[0], data[1], fake_seed[0], fake_seed[1], fake_seed_gen[0], data[1]

                real_idx = real_idx + 1

    def train(self, epoch=10000, save_each_epoch=100, callbacks=None):
        agv_logger = AvgLogger(['discriminator_loss', 'GAN_loss'])

        for i in range(epoch):
            train_data = list(self.data_getter())

            agv_logger.refresh()
            for input_real, label_real, input_fake_seed, label_fake, input_fake_seed_gen, label_fake_gen in train_data:

                dis_loss = self._train_discriminator(input_real, label_real, input_fake_seed, label_fake)

                # train generator
                gan_loss = self._train_generator(input_real, label_real, input_fake_seed_gen, label_fake_gen)

                agv_logger.update_state([dis_loss, gan_loss])

            print("|Epoch: {epoch:04d}| {loss}".format(epoch=i, loss=agv_logger.result()))

            # save weight every 100 epochs
            if i > 0 and i % save_each_epoch == 0:
                self.model.save_weights(os.path.join(self._save_path, "ckpt_{epoch:04d}.tf".format(epoch=i)))

    def test(self, epoch=None, seed=None):
        if epoch is not None:
            self.model.load_weights(os.path.join(self._save_path, "ckpt_{epoch:04d}.tf".format(epoch=epoch)))
        else:
            raise ValueError("Need specific epoch number")

        if seed is None:
            fake_seed, fake_label = self.create_fake_seed(16)
        else:
            fake_seed = seed[0]
            fake_label = seed[1]

        gen_list = self.generator.model(fake_seed, training=False)
        output = self.discriminator.model(gen_list, training=False)
        # loss = self.discriminator.calculate_loss_function(tf.convert_to_tensor(fake_label, dtype=tf.float32), output, axis=1)

        return output, gen_list

    def _train_discriminator(self, input_real, label_real, input_fake_seed, label_fake):
        input_fake = self.generator.model(input_fake_seed, training=True) if input_fake_seed[0].shape[0] > 0 else input_fake_seed[0]
        input_data = np.concatenate([input_real[0], input_fake])
        label_data = np.concatenate([label_real, label_fake])

        return self.discriminator.model.train_on_batch([input_data], label_data)

    def _train_generator(self, input_real, label_real, input_fake_seed, label_fake):
        with tf.GradientTape() as tape:
            gan_output = self.model(input_fake_seed, training=True)
            gan_loss = self.model.compiled_loss(label_fake, gan_output)  # self.calculate_loss_function(label_fake, gan_output, axis=1)

        grads = tape.gradient(gan_loss, self.generator.model.trainable_variables)  # calculate gradients
        self.model.optimizer.apply_gradients(zip(grads, self.generator.model.trainable_variables))  # update gradients

        return gan_loss
        # return self.model.train_on_batch(input_fake_seed, label_fake)
