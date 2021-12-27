import math
import os
from abc import ABC, abstractmethod

from PIL import Image

from model.core import ModelCore, Net, LOSS, Dataset, DatasetFactory

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

    def __init__(self, data_path, batch_size=64, avg_list=['loss'], loss=LOSS.CATEGORICAL_CROSSENTROPY, train_test_ratio=1, is_classify=False,
                 generator: ModelCore=None, discriminator: ModelCore=None, latent_space_size=100, flip_coin=False):
        if not isinstance(generator, ModelCore):
            raise ValueError('Generator is not ModelCore')
        if not isinstance(discriminator, ModelCore):
            raise ValueError('Discriminator is not ModelCore')

        self.generator = generator
        self.discriminator = discriminator
        self.latent_space_size = latent_space_size
        self._data_fake_seed = None
        self.flip_coin = flip_coin

        if self.flip_coin:
            data_batch = batch_size
        else:
            data_batch = batch_size / 2

        self._train_data_real = Dataset(int(data_batch), self)
        self._train_data_fake_seed = Dataset(int(data_batch), self)
        self._test_data_real = Dataset(int(data_batch), self)
        self._test_data_fake_seed = Dataset(int(data_batch), self)

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
        input_ = self.generator.model.input
        out = self.generator.model.output
        out = self.discriminator.model(out, training=True)
        self.model = tf.keras.Model(inputs=input_, outputs=out)

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
        return np.random.normal(0, 1, (count, self._model_core.latent_space_size)), np.array([[0] for i in range(count)])

    def flip_coin(self, chance=0.5):
        return np.random.binomial(1, chance)

    def _train_discriminator(self, optimizer, input_real, label_real, input_fake_seed, label_fake):
        input_fake = self._model_core.generator.model(input_fake_seed, training=True) if input_fake_seed[0].shape[0] > 0 else input_fake_seed[0]
        input_data = np.concatenate([input_real[0], input_fake])
        label_data = np.concatenate([label_real, label_fake])
        with tf.GradientTape() as tape:
            discriminator_output = self._model_core.discriminator.model([input_data], training=True)

            discriminator_loss = self._model_core.discriminator.calculate_loss_function(label_data, discriminator_output, axis=1)

        grads = tape.gradient(discriminator_loss,
                              self._model_core.discriminator.model.trainable_variables)  # calculate gradients
        optimizer.apply_gradients(
            zip(grads, self._model_core.discriminator.model.trainable_variables))  # update gradients

        return discriminator_loss

    def _train_generator(self, optimizer, input_real, label_real, input_fake_seed, label_fake):
        with tf.GradientTape() as tape:
            gan_output = self._model_core.model(input_fake_seed, training=True)
            gan_loss = self._model_core.calculate_loss_function(label_fake, gan_output, axis=1)

        grads = tape.gradient(gan_loss, self._model_core.generator.model.trainable_variables)  # calculate gradients
        optimizer.apply_gradients(zip(grads, self._model_core.generator.model.trainable_variables))  # update gradients

        return gan_loss

    def data_getter(self):
        """
        make data to train
        :return (train_data_for_discriminator, label_data_for_discriminator,
            fake_seed_for_discriminator, label_for_discriminator, fake_seed_for_generator, label_for_generator)
        """
        train_data_real = self._model_core.get_train_data_real()
        train_data_fake_seed = self._model_core.get_train_data_fake_seed()

        gen_real = list(train_data_real.get())

        if train_data_fake_seed is not None and len(train_data_fake_seed) > 0:
            gen_fake_seed = list(train_data_fake_seed.get())
        else:
            gen_fake_seed = None

        real_idx = 0

        while real_idx < len(gen_real):
            if self._model_core.flip_coin:
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

    def train(self, pretrained_module_name=None, pretrained_module_index=None, epoch=10000, lr=0.001, beta_1=0.9):
        optimizer_discriminator = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
        optimizer_generator = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)

        for i in range(epoch):
            self._model_core.discriminator.avg_logger.refresh()
            self._model_core.avg_logger.refresh()
            train_data = list(self.data_getter())

            for input_real, label_real, input_fake_seed, label_fake, input_fake_seed_gen, label_fake_gen in train_data:

                discriminator_loss = self._train_discriminator(optimizer_discriminator, input_real, label_real, input_fake_seed, label_fake)

                # train generator
                gan_loss = self._train_generator(optimizer_generator, input_real, label_real, input_fake_seed_gen, label_fake_gen)

                if isinstance(gan_loss, list):
                    self._model_core.avg_logger.update_state(gan_loss)
                else:
                    self._model_core.avg_logger.update_state([gan_loss])

                if isinstance(discriminator_loss, list):
                    self._model_core.discriminator.avg_logger.update_state(discriminator_loss)
                else:
                    self._model_core.discriminator.avg_logger.update_state([discriminator_loss])

            log_result = self._model_core.avg_logger.result()
            log_discriminator = self._model_core.discriminator.avg_logger.result()
            print('Epoch: {} discriminator: {} generator: {}'.format(i, log_discriminator, log_result))

            # save weight every 100 epochs
            if (i % 1 == 0 and i != 0) or self.save_when(i, self._model_core.avg_logger.result_value()):
                self._model_core.model.save_weights(os.path.join(self._base_path,
                                                './checkpoints/{}_{}.tf'.format(self.name, i)))

    def test(self, index, seed=None):
        if index > -1:
            self._model_core.model.load_weights(os.path.join(self._base_path,
                                                             './checkpoints/{}_{}.tf'.format(self.name, index)))
        else:
            self._model_core.load_weight()
        self._model_core.avg_logger.refresh()
        if seed is None:
            fake_seed, fake_label = self.create_fake_seed(16)
        else:
            fake_seed = seed[0]
            fake_label = seed[1]
        gen_list = self._model_core.generator.model(fake_seed, training=False)
        output = self._model_core.discriminator.model(gen_list, training=False)
        loss = self._model_core.discriminator.calculate_loss_function(tf.convert_to_tensor(fake_label, dtype=tf.float32), output, axis=1)

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
            seed, label = self.create_fake_seed(1)
            pre_data = seed[:1]

        return self._model_core.generator.model([pre_data], training=False)


class Util:
    @staticmethod
    def load_image(list_files, width=64, height=64, gray=False):
        image_arr = []
        for f in list_files:
            if gray:
                im = Image.open(f).convert("L")
            else:
                im = Image.open(f).convert("RGB")

            re_width = width
            re_height = height
            if width <= 0:
                re_width = im.width
            if height <= 0:
                re_height = im.height

            im = im.resize((re_width, re_height))
            im_np = np.asarray(im)
            image_arr.append(im_np)

        return image_arr

    @staticmethod
    def get_image_list(base_path, extension=".png"):
        list_files = []
        for file in os.listdir(base_path):
            if file.endswith(extension):
                list_files.append(os.path.join(base_path, file))

        return list_files

    @staticmethod
    def image_normalization(np_arr):
        return (np_arr - 127.5) / 127.5

    @staticmethod
    def to_image(np_one):
        return (np_one - np.min(np_one)) / (np.max(np_one) - np.min(np_one))
