from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np

from model.core import ModelCore, Net


class Example(metaclass=ABCMeta):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def run(self):
        pass


class ClassifierExample(Example, ModelCore):
    def __init__(self):
        ModelCore.__init__(self, "")

    def build_model(self):
        input_ = tf.keras.Input([28, 28])
        flatten = tf.keras.layers.Flatten()(input_)
        dense = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(flatten)
        dense = tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)(dense)

        self.model = tf.keras.Model(inputs=[input_], outputs=[dense])

    def read_data(self):
        (train_images, train_labels), (test_images, test_labels) = self.load_data()

        images = np.concatenate([train_images, test_images]) / 255.
        labels = np.concatenate([train_labels, test_labels])

        self._data_all = []
        for i in range(len(images)):
            zeros = np.zeros(10)
            zeros[labels[i]] = 1
            self._data_all.append({'input': images[i], 'output': zeros})

    def load_data(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist

        return fashion_mnist.load_data()

    def run(self):
        net = Net("FashionMNIST", "./fashion/", self)
        net.train(epoch=1000)
        best_loss = 1000
        best_loss_index = -1

        for i in range(10):
            idx = (i + 1) * 100
            loss = net.test(idx)
            if best_loss > loss:
                best_loss = loss
                best_loss_index = idx

        print("best loss: ", best_loss, " idx: ", best_loss_index)
