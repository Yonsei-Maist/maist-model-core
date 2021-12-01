from model.core import ModelCore, LOSS
from model.examples.core import Example

import tensorflow as tf
import numpy as np

from model.gan.core import GanCore, GanNetwork


class GanExample(Example, GanCore):
    WIDTH = 28
    HEIGHT = 28

    class Generator(ModelCore):
        LATENT_SPACE_SIZE = 100
        BLOCK_STARTING_SIZE = 128
        NUM_BLOCK = 4

        def __init__(self):
            super().__init__("", loss=LOSS.BINARY_CROSSENTROPY, train_test_ratio=0)

        def build_model(self):
            block_size = GanExample.Generator.BLOCK_STARTING_SIZE
            input_ = tf.keras.Input([GanExample.Generator.LATENT_SPACE_SIZE])
            dense = tf.keras.layers.Dense(block_size)(input_)
            leaky_relu = tf.keras.layers.LeakyReLU(0.2)(dense)
            batch_normal = tf.keras.layers.BatchNormalization(momentum=0.8)(leaky_relu)
            for i in range(GanExample.Generator.NUM_BLOCK - 1):
                block_size = block_size * 2
                dense = tf.keras.layers.Dense(block_size)(batch_normal)
                leaky_relu = tf.keras.layers.LeakyReLU(0.2)(dense)
                batch_normal = tf.keras.layers.BatchNormalization(momentum=0.8)(leaky_relu)

            dense = tf.keras.layers.Dense(GanExample.WIDTH * GanExample.HEIGHT, activation=tf.keras.activations.tanh)(batch_normal)
            reshape = tf.keras.layers.Reshape([GanExample.WIDTH, GanExample.HEIGHT])(dense)

            self.model = tf.keras.Model(inputs=[input_], outputs=[reshape])

        def read_data(self):
            self._data_all = []

    class Discriminator(ModelCore):
        def __init__(self):
            super().__init__("", loss=LOSS.CATEGORICAL_CROSSENTROPY, train_test_ratio=0)

        def build_model(self):
            input_ = tf.keras.Input([GanExample.WIDTH, GanExample.HEIGHT])
            flatten = tf.keras.layers.Flatten()(input_)
            dense = tf.keras.layers.Dense(GanExample.WIDTH * GanExample.HEIGHT)(flatten)
            leaky_relu = tf.keras.layers.LeakyReLU(0.2)(dense)
            dense = tf.keras.layers.Dense(GanExample.WIDTH * GanExample.HEIGHT / 2)(leaky_relu)
            leaky_relu = tf.keras.layers.LeakyReLU(0.2)(dense)
            dense = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)(leaky_relu)

            self.model = tf.keras.Model(inputs=[input_], outputs=[dense])

        def read_data(self):
            self._data_all = []

    def __init__(self):
        GanCore.__init__(self, "", loss=LOSS.CATEGORICAL_CROSSENTROPY, batch_size=16, discriminator=GanExample.Discriminator(), generator=GanExample.Generator())

    def load_data(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist

        return fashion_mnist.load_data()

    def run(self):
        net = GanNetwork("FashionGenerator", "./fashion_gen/", self)
        net.train()

    def read_data(self):
        (train_images, train_labels), (test_images, test_labels) = self.load_data()

        images = np.concatenate([train_images, test_images]) / 255.
        labels = np.concatenate([train_labels, test_labels])

        self._data_all = []
        for i in range(len(images)):
            self._data_all.append({'input': images[i], 'output': [1, 0]})
