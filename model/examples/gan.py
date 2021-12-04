from PIL import Image

from model.core import ModelCore, LOSS
from model.examples.core import Example

import tensorflow as tf
import numpy as np

from model.gan.core import GanCore, GanNetwork
import subprocess
import os


class GanExample(Example, GanCore):
    WIDTH = 28
    HEIGHT = 28
    CHANNEL = 1

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
            reshape = tf.keras.layers.Reshape([GanExample.WIDTH, GanExample.HEIGHT, GanExample.CHANNEL])(dense)

            self.model = tf.keras.Model(inputs=[input_], outputs=[reshape])

        def read_data(self):
            self._data_all = []

    class Discriminator(ModelCore):
        def __init__(self):
            super().__init__("", loss=LOSS.BINARY_CROSSENTROPY, train_test_ratio=0)

        def build_model(self):
            input_ = tf.keras.Input([GanExample.WIDTH, GanExample.HEIGHT, GanExample.CHANNEL])
            flatten = tf.keras.layers.Flatten()(input_)
            dense = tf.keras.layers.Dense(GanExample.WIDTH * GanExample.HEIGHT)(flatten)
            leaky_relu = tf.keras.layers.LeakyReLU(0.2)(dense)
            dense = tf.keras.layers.Dense(GanExample.WIDTH * GanExample.HEIGHT / 2)(leaky_relu)
            leaky_relu = tf.keras.layers.LeakyReLU(0.2)(dense)
            dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(leaky_relu)

            self.model = tf.keras.Model(inputs=[input_], outputs=[dense])

        def read_data(self):
            self._data_all = []

    def __init__(self):
        GanCore.__init__(self, "", loss=LOSS.BINARY_CROSSENTROPY, batch_size=128, discriminator=GanExample.Discriminator(), generator=GanExample.Generator())

    def load_data(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist

        return fashion_mnist.load_data()

    def run(self):
        net = GanNetwork("FashionGenerator", "./fashion_gen/", self)
        net.train(epoch=60000, lr=0.1)
        return net.test(60000)

    def read_data(self):
        (train_images, train_labels), (test_images, test_labels) = self.load_data()

        images = np.concatenate([train_images, test_images]) / 255.
        labels = np.concatenate([train_labels, test_labels])

        all = []
        for i in range(len(images)):
            if labels[i] == 0:
                all.append(images[i])

        self._data_all = []
        for i in range(len(all)):
            self._data_all.append({'input': np.reshape(all[i], (GanExample.WIDTH, GanExample.HEIGHT, GanExample.CHANNEL)), 'output': [1]})


class DCGanExample(Example, GanCore):
    WIDTH = 64
    HEIGHT = 64
    CHANNEL = 1

    class Discriminator(ModelCore):

        def __init__(self):
            super().__init__("", loss=LOSS.BINARY_CROSSENTROPY, train_test_ratio=0)

        def build_model(self):
            input_ = tf.keras.Input([DCGanExample.WIDTH, DCGanExample.HEIGHT, DCGanExample.CHANNEL])
            conv = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(input_)
            dropout = tf.keras.layers.Dropout(0.3)(conv)
            batch_normal = tf.keras.layers.BatchNormalization()(dropout)
            conv = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(batch_normal)
            dropout = tf.keras.layers.Dropout(0.3)(conv)
            batch_normal = tf.keras.layers.BatchNormalization()(dropout)

            flatten = tf.keras.layers.Flatten()(batch_normal)
            dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(flatten)

            self.model = tf.keras.Model(inputs=[input_], outputs=[dense])

        def read_data(self):
            self._data_all = []

    class Generator(ModelCore):
        LATENT_SPACE_SIZE = 100

        def __init__(self):
            super().__init__("", loss=LOSS.BINARY_CROSSENTROPY, train_test_ratio=0)

        def build_model(self):
            input_ = tf.keras.Input([DCGanExample.Generator.LATENT_SPACE_SIZE])
            dense = tf.keras.layers.Dense(256 * 8 * 8, activation=tf.keras.layers.LeakyReLU(0.2))(input_)
            batch_normal = tf.keras.layers.BatchNormalization()(dense)
            reshape = tf.keras.layers.Reshape([8, 8, 256])(batch_normal)
            up_sample = tf.keras.layers.UpSampling2D()(reshape)

            conv = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(up_sample)
            batch_normal = tf.keras.layers.BatchNormalization()(conv)
            up_sample = tf.keras.layers.UpSampling2D()(batch_normal)

            conv = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(up_sample)
            batch_normal = tf.keras.layers.BatchNormalization()(conv)
            up_sample = tf.keras.layers.UpSampling2D()(batch_normal)

            conv = tf.keras.layers.Conv2D(DCGanExample.CHANNEL, kernel_size=(5, 5), padding="same", activation=tf.keras.activations.tanh)(up_sample)

            self.model = tf.keras.Model(inputs=[input_], outputs=[conv])

        def read_data(self):
            self._data_all = []

    def __init__(self, data_path, train_test_ratio=.2):
        GanCore.__init__(self, data_path, loss=LOSS.BINARY_CROSSENTROPY, batch_size=128, train_test_ratio=train_test_ratio, discriminator=DCGanExample.Discriminator(), generator=DCGanExample.Generator())

    def load_image(self, list_files, width=64, height=64, gray=False):
        image_arr = []
        for f in list_files:
            if gray:
                im = Image.open(f).convert("L")
            else:
                im = Image.open(f).convert("RGB")

            im = im.resize((width, height))
            im_np = np.asarray(im)
            image_arr.append(im_np)

        return image_arr

    def load_data(self):
        if not os.path.exists(os.path.join(self._data_path, "data/church_outdoor_train_lmdb/expanded")):
            subprocess.Popen("cd", cwd=self._data_path)
            subprocess.Popen(["git", "clone",  "https://github.com/fyu/lsun.git"])
            subprocess.Popen(["mkdir", "data"])
            subprocess.Popen(["python", "lsun/download.py", "-o", "./data"])
            subprocess.Popen(["unzip", "./data/church_outdoor_train_lmdb.zip"])
            subprocess.Popen(["unzip", "./data/church_outdoor_val_lmdb.zip"])
            subprocess.Popen(["mkdir", "./data/church_outdoor_train_lmdb/expanded"])
            subprocess.Popen(["python", "lsun/data.py", "export", "./data/church_outdoor_train_lmdb", "--out_dir",
                              "./data/church_outdoor_train_lmdb/expanded", "--flat"])

        list_files = []
        base_path = os.path.join(self._data_path, "data/church_outdoor_train_lmdb/expanded/")
        for file in os.listdir(base_path):
            if file.endswith(".webp"):
                list_files.append(os.path.join(base_path, file))

        return self.load_image(list_files, gray=True)

    def run(self):
        net = GanNetwork("DCGanExample", "./dcgan", self)
        net.train(60000)
        return net.test(60000)

    def read_data(self):
        npy_path = os.path.join(self._data_path, "data/train.npy")
        if not os.path.exists(npy_path):
            datas = self.load_data()
            np.save(npy_path, datas)
        else:
            datas = np.load(npy_path)

        self._data_all = []
        for i in range(len(datas)):
            self._data_all.append({"input": datas[i], "output": [1]})
