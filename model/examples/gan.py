from PIL import Image

from model.core import ModelCore, LOSS
from model.examples.core import Example

import tensorflow as tf
import numpy as np

from model.gan.core import GanCore, Util
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
            super().__init__("", "./fashion_gen", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), train_test_ratio=0)

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
            super().__init__("", "./fashion_gen", loss=tf.keras.losses.BinaryCrossentropy(),
                             optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9), train_test_ratio=0,
                             input_dtype=tf.float32, output_dtype=tf.float32)

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
        GanCore.__init__(self, "", "./fashion_gen", loss=tf.keras.losses.BinaryCrossentropy(), batch_size=128,
                         discriminator=GanExample.Discriminator(), generator=GanExample.Generator(),
                         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9))

    def load_data(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist

        return fashion_mnist.load_data()

    def run(self, index=1, train=True, seed=None):
        asd = {1, '22'}.to
        if train:
            self.train(epoch=60000)
        return self.test(index, seed=seed)

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
    CHANNEL = 3

    class DiscriminatorLoss(tf.keras.losses.BinaryCrossentropy):
        def call(self, y_true, y_pred):
            noise_labels = GanCore.noisy_labels(y_true, 0.05)
            noise_labels = tf.cast(noise_labels, dtype=tf.float32)

            if tf.math.equal(GanCore.is_fake(y_true), True):
                smooth_labels = GanCore.smooth_negative_labels(noise_labels)
            else:
                smooth_labels = GanCore.smooth_positive_labels(noise_labels)

            return super().call(smooth_labels, y_pred)

    class GanLoss(tf.keras.losses.BinaryCrossentropy):
        def call(self, y_true, y_pred):
            if tf.math.equal(GanCore.is_fake(y_true), True):
                smooth_labels = GanCore.smooth_negative_labels(y_true)
            else:
                smooth_labels = GanCore.smooth_positive_labels(y_true)

            return super().call(smooth_labels, y_pred)

    class Discriminator(ModelCore):

        def __init__(self):
            super().__init__("", "", loss=DCGanExample.DiscriminatorLoss(), train_test_ratio=0)

        def build_model(self):
            input_ = tf.keras.Input([DCGanExample.WIDTH, DCGanExample.HEIGHT, DCGanExample.CHANNEL])
            # input_ = tf.keras.layers.GaussianNoise(0.6)(input_)
            conv = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(input_)
            dropout = tf.keras.layers.Dropout(0.3)(conv)
            batch_normal = tf.keras.layers.BatchNormalization()(dropout)
            # batch_normal = tf.keras.layers.GaussianNoise(0.6)(batch_normal)
            conv = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(batch_normal)
            dropout = tf.keras.layers.Dropout(0.3)(conv)
            batch_normal = tf.keras.layers.BatchNormalization()(dropout)
            # batch_normal = tf.keras.layers.GaussianNoise(0.6)(batch_normal)

            flatten = tf.keras.layers.Flatten()(batch_normal)
            dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(flatten)

            self.model = tf.keras.Model(inputs=[input_], outputs=[dense])

        def read_data(self):
            self._data_all = []

    class Generator(ModelCore):
        LATENT_SPACE_SIZE = 100

        def __init__(self):
            super().__init__("", "", loss=LOSS.BINARY_CROSSENTROPY, train_test_ratio=0)

        def build_model(self):
            input_ = tf.keras.Input([DCGanExample.Generator.LATENT_SPACE_SIZE])
            dense = tf.keras.layers.Dense(256 * 8 * 8, activation=tf.keras.layers.LeakyReLU(0.2))(input_)
            batch_normal = tf.keras.layers.BatchNormalization()(dense)
            reshape = tf.keras.layers.Reshape([8, 8, 256])(batch_normal)
            up_sample = tf.keras.layers.UpSampling2D()(reshape)

            conv = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), padding="same", activation=tf.keras.activations.relu)(up_sample)
            batch_normal = tf.keras.layers.BatchNormalization()(conv)
            up_sample = tf.keras.layers.UpSampling2D()(batch_normal)

            conv = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), padding="same", activation=tf.keras.activations.relu)(up_sample)
            batch_normal = tf.keras.layers.BatchNormalization()(conv)
            up_sample = tf.keras.layers.UpSampling2D()(batch_normal)

            conv = tf.keras.layers.Conv2D(DCGanExample.CHANNEL, kernel_size=(5, 5), padding="same", activation=tf.keras.activations.tanh)(up_sample)

            self.model = tf.keras.Model(inputs=[input_], outputs=[conv])

        def read_data(self):
            self._data_all = []

    def __init__(self, data_path, load_data_ratio=.2):
        self._load_data_ratio = load_data_ratio
        GanCore.__init__(self, data_path, "./dcgan", loss=DCGanExample.GanLoss(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.2), batch_size=128,
                         train_test_ratio=1., discriminator=DCGanExample.Discriminator(),
                         generator=DCGanExample.Generator(), flip_coin=True)

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

        list_files = Util.get_image_list(os.path.join(self._data_path, "data/church_outdoor_train_lmdb/expanded/"), ".webp")

        return Util.load_image(list_files, gray=False)

    def run(self, train=True, index=1, seed=None):
        if train:
            self.train(epoch=501)
        return self.test(index, seed)

    def read_data(self):
        npy_path = os.path.join(self._data_path, "data/train.npy")
        if not os.path.exists(npy_path):
            datas = self.load_data()
            np.save(npy_path, datas)
        else:
            datas = (np.load(npy_path) - 127.5) / 127.5

        self._data_all = []
        for i in range(int(len(datas) * self._load_data_ratio)):
            self._data_all.append({"input": np.reshape(datas[i], (DCGanExample.WIDTH, DCGanExample.HEIGHT, DCGanExample.CHANNEL)), "output": [1]})


class Pix2pixExample(Example, GanCore):
    WIDTH = 256
    HEIGHT = 256
    CHANNEL = 3

    class Discriminator(ModelCore):

        def __init__(self):
            super().__init__("", "./pix2pix", loss=tf.keras.losses.MeanSquaredError(),
                             train_test_ratio=0, optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, decay=1e-5))

        def build_model(self):
            input_1 = tf.keras.Input([Pix2pixExample.WIDTH, Pix2pixExample.HEIGHT, Pix2pixExample.CHANNEL])
            input_2 = tf.keras.Input([Pix2pixExample.WIDTH, Pix2pixExample.HEIGHT, Pix2pixExample.CHANNEL])

            input_ = tf.keras.layers.Concatenate(axis=-1)([input_1, input_2])

            up_layer_1 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(input_)
            leaky_layer_1 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_layer_1)
            up_layer_2 = tf.keras.layers.Conv2D(64 * 2, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(leaky_layer_1)
            leaky_layer_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_layer_2)
            up_layer_3 = tf.keras.layers.Conv2D(64 * 4, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(leaky_layer_2)
            leaky_layer_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_layer_3)
            up_layer_4 = tf.keras.layers.Conv2D(64 * 8, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(leaky_layer_3)
            leaky_layer_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_layer_4)

            out = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding="same")(leaky_layer_4)

            self.model = tf.keras.Model(inputs=[input_1, input_2], outputs=out)

        def read_data(self):
            self._data_all = []

    class Generator(ModelCore):
        LATENT_SPACE_SIZE = 100

        def __init__(self):
            super().__init__("", "./pix2pix", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                             train_test_ratio=0, optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, decay=1e-5))

        def build_model(self):
            input_ = tf.keras.Input([Pix2pixExample.WIDTH, Pix2pixExample.HEIGHT, Pix2pixExample.CHANNEL])

            # Simple U-net Encoder
            down_1 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(input_)
            down_2 = tf.keras.layers.Conv2D(64 * 2, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(down_1)
            norm_2 = tf.keras.layers.BatchNormalization()(down_2)
            down_3 = tf.keras.layers.Conv2D(64 * 4, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(norm_2)
            norm_3 = tf.keras.layers.BatchNormalization()(down_3)

            # Keep the samplings
            down_4 = tf.keras.layers.Conv2D(64 * 8, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(norm_3)
            norm_4 = tf.keras.layers.BatchNormalization()(down_4)
            down_5 = tf.keras.layers.Conv2D(64 * 8, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(norm_4)
            norm_5 = tf.keras.layers.BatchNormalization()(down_5)
            down_6 = tf.keras.layers.Conv2D(64 * 8, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(norm_5)
            norm_6 = tf.keras.layers.BatchNormalization()(down_6)
            down_7 = tf.keras.layers.Conv2D(64 * 8, kernel_size=4, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2))(norm_6)
            norm_7 = tf.keras.layers.BatchNormalization()(down_7)

            # Keep filters and Up-sampling
            upsample_1 = tf.keras.layers.UpSampling2D(size=2)(norm_7)
            up_conv_1 = tf.keras.layers.Conv2D(64 * 8, kernel_size=4, strides=1, padding="same", activation=tf.keras.activations.relu)(upsample_1)

            norm_up_1 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_conv_1)
            add_skip_1 = tf.keras.layers.Concatenate()([norm_up_1, norm_6])

            upsample_2 = tf.keras.layers.UpSampling2D(size=2)(add_skip_1)
            up_conv_2 = tf.keras.layers.Conv2D(64 * 8, kernel_size=4, strides=1, padding="same", activation=tf.keras.activations.relu)(upsample_2)

            norm_up_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_conv_2)
            add_skip_2 = tf.keras.layers.Concatenate()([norm_up_2, norm_5])

            upsample_3 = tf.keras.layers.UpSampling2D(size=2)(add_skip_2)
            up_conv_3 = tf.keras.layers.Conv2D(64 * 8, kernel_size=4, strides=1, padding="same",
                                               activation=tf.keras.activations.relu)(upsample_3)

            norm_up_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_conv_3)
            add_skip_3 = tf.keras.layers.Concatenate()([norm_up_3, norm_4])

            # Decoder
            upsample_4 = tf.keras.layers.UpSampling2D(size=2)(add_skip_3)
            up_conv_4 = tf.keras.layers.Conv2D(64 * 4, kernel_size=4, strides=1, padding="same", activation=tf.keras.activations.relu)(upsample_4)
            norm_up_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_conv_4)
            add_skip_4 = tf.keras.layers.Concatenate()([norm_up_4, norm_3])

            upsample_5 = tf.keras.layers.UpSampling2D(size=2)(add_skip_4)
            up_conv_5 = tf.keras.layers.Conv2D(64 * 2, kernel_size=4, strides=1, padding="same", activation=tf.keras.activations.relu)(upsample_5)
            norm_up_5 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_conv_5)
            add_skip_5 = tf.keras.layers.Concatenate()([norm_up_5, norm_2])

            upsample_6 = tf.keras.layers.UpSampling2D(size=2)(add_skip_5)
            up_conv_6 = tf.keras.layers.Conv2D(64 * 4, kernel_size=4, strides=1, padding="same", activation=tf.keras.activations.relu)(upsample_6)
            norm_up_6 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_conv_6)
            add_skip_6 = tf.keras.layers.Concatenate()([norm_up_6, down_1])

            # Output
            upsample_last = tf.keras.layers.UpSampling2D(size=2)(add_skip_6)
            out = tf.keras.layers.Conv2D(Pix2pixExample.CHANNEL, kernel_size=4, strides=1, padding="same", activation=tf.keras.activations.tanh)(upsample_last)

            self.model = tf.keras.Model(inputs=input_, outputs=out)

        def read_data(self):
            self._data_all = []

    def __init__(self, data_path, load_data_ratio=.2):
        self._load_data_ratio = load_data_ratio
        self._loss_weight = [1, 100]

        GanCore.__init__(self, data_path, "./pix2pix", loss=[tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanAbsoluteError()], batch_size=2,
                         train_test_ratio=1., discriminator=Pix2pixExample.Discriminator(), loss_weights=[1, 100],
                         generator=Pix2pixExample.Generator(), optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5))

    def run(self, index=1, train=True, seed=None):
        if train:
            self.train(epoch=100)
        return self.test(index, seed)

    def build_model(self):
        origin_target = tf.keras.Input([Pix2pixExample.WIDTH, Pix2pixExample.HEIGHT, Pix2pixExample.CHANNEL])
        origin_input = tf.keras.Input([Pix2pixExample.WIDTH, Pix2pixExample.HEIGHT, Pix2pixExample.CHANNEL])

        fake_input = self.generator.model(origin_input, training=True)

        valid = self.discriminator.model([fake_input, origin_input], training=True)

        self.model = tf.keras.Model(inputs=[origin_target, origin_input], outputs=[valid, fake_input])

    def load_data(self, train=True):
        if train:
            dir = os.path.join(self._data_path, "train")
        else:
            dir = os.path.join(self._data_path, "val")

        list_file = Util.get_image_list(dir, ".jpg")
        image_all = np.array(Util.load_image(list_file, -1, -1, gray=False))

        origin_a = []
        origin_b = []

        for img in image_all:
            origin_a.append((np.array(img[:, :Pix2pixExample.WIDTH]) - 127.5) / 127.5)
            origin_b.append((np.array(img[:, Pix2pixExample.WIDTH:]) - 127.5) / 127.5)

        return origin_a, origin_b

    def read_data(self):
        origin_a, origin_b = self.load_data()
        self._data_all = []
        self._data_fake_seed = []
        for i in range(len(origin_a)):
            origin_a_item = origin_a[i]
            origin_b_item = origin_b[i]
            self._data_all.append({'input': origin_a_item, 'output': np.ones((int(Pix2pixExample.WIDTH / 2**4), int(Pix2pixExample.HEIGHT / 2**4), 1))})
            self._data_fake_seed.append({'input': origin_b_item, 'output': np.zeros((int(Pix2pixExample.WIDTH / 2**4), int(Pix2pixExample.HEIGHT / 2**4), 1))})

    def _train_discriminator(self, input_real, label_real, input_fake_seed, label_fake):
        fake_a = self.generator.model.predict(input_fake_seed[0])

        loss_real = self.discriminator.model.train_on_batch([input_real[0], input_fake_seed[0]], label_real)
        loss_fake = self.discriminator.model.train_on_batch([fake_a, input_fake_seed[0]], label_fake)

        return 0.5 * tf.math.add(loss_real, loss_fake)

    def _train_generator(self, input_real, label_real, input_fake_seed, label_fake):
        return self.model.train_on_batch([input_real[0], input_fake_seed[0]], [label_real, input_real[0]])
