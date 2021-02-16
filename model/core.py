import tensorflow as tf
from enum import Enum
import math
import os
from abc import *


class LOSS(Enum):
    MSE = 1
    COSINE_SIMILARITY = 2
    CATEGORICAL_CROSSENTROPY = 3
    BINARY_CROSSENTROPY = 4


class AvgLogger:
    def __init__(self, avg_list: list):
        self.__avg_type_list = avg_list
        self.__avg_list = None
        self.__build_avg()

    def __build_avg(self):
        self.__avg_list = []

        for avg_type in self.__avg_type_list:
            self.__avg_list.append(tf.keras.metrics.Mean(avg_type, dtype=tf.float32))

    def refresh(self):
        self.__build_avg()

    def update_state(self, value_list: list):
        if len(value_list) != len(self.__avg_type_list):
            raise Exception(
                "Unmatch Length: {0} compared to type list length({1})".format(len(value_list),
                                                                               len(self.__avg_type_list)))

        for i in range(len(self.__avg_list)):
            avg = self.__avg_list[i]
            value = value_list[i]
            avg.update_state(value)

    def result(self):
        return ",".join("{0}: {1}".format(self.__avg_type_list[i], self.__avg_list[i].result().numpy())
                        for i in range(len(self.__avg_type_list)))

    def result_value(self):
        return [self.__avg_list[i].result().numpy() for i in range(len(self.__avg_type_list))]


class LossFunction:
    def __init__(self, loss: LOSS):
        self.__lambda = None
        if loss == LOSS.MSE:
            self.__lambda = lambda labels, outputs, axis: tf.keras.losses.MSE(labels, outputs, axis)
        elif loss == LOSS.COSINE_SIMILARITY:
            self.__lambda = lambda labels, outputs, axis: tf.keras.losses.cosine_similarity(labels, outputs, axis)
        elif loss == LOSS.BINARY_CROSSENTROPY:
            self.__lambda = lambda labels, outputs, axis: tf.keras.losses.binary_crossentropy(labels, outputs, axis)
        else:  # default
            self.__lambda = lambda labels, outputs, axis: tf.keras.losses.categorical_crossentropy(labels, outputs,
                                                                                                   axis)

    def calculate(self, labels, outputs, axis):
        return self.__lambda(labels, outputs, axis)


class Dataset:
    def __init__(self, batch_size):
        self.__inputs = None
        self.__labels = None
        self.__batch_size = batch_size

    def set(self, inputs, labels):
        len_input = -1
        for i in range(len(inputs)):
            if i == 0:
                len_input = len(inputs[i])
            elif len_input != len(inputs[i]):
                raise Exception("Doesn't match a length of all inputs")

        for i in range(len(labels)):
            if len_input != len(labels[i]):
                raise Exception("Doesn't match between the length of inputs and a length of labels")

        self.__inputs = inputs
        self.__labels = labels

    def get(self):
        iter = math.ceil(len(self) / self.__batch_size)
        for i in range(iter):
            if len(self.__labels) == 1:
                yield [item[i * self.__batch_size: i * self.__batch_size + self.__batch_size] for item in self.__inputs], \
                      self.__labels[0][i * self.__batch_size: i * self.__batch_size + self.__batch_size]
            else:
                yield [item[i * self.__batch_size: i * self.__batch_size + self.__batch_size] for item in self.__inputs], \
                      [item[i * self.__batch_size: i * self.__batch_size + self.__batch_size] for item in self.__labels]

    def __len__(self):
        if len(self.__inputs) > 0:
            return len(self.__inputs[0])

        return 0


class ModelCore(metaclass=ABCMeta):
    def __init__(self, data_path, batch_size=64, avg_list=['loss'], loss=LOSS.CATEGORICAL_CROSSENTROPY):
        self._train_data = Dataset(batch_size)
        self._test_data = Dataset(batch_size)
        self.model = None
        self.batch_size = batch_size
        self._data_path = data_path
        self.avg_logger = AvgLogger(avg_list)
        self.loss_function = LossFunction(loss)

        self.read_data()
        self.build_model()

    def check_integer_string(self, int_value):
        try:
            return int(int_value)
        except ValueError as e:
            return -1

    def get_train_data(self):
        return self._train_data

    def get_test_data(self):
        return self._test_data

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def read_data(self):
        pass


class Net:
    def __init__(self, module_name, base_path, model_core: ModelCore):
        """
        :param model_core: ModelCore instance
        """
        self.name = module_name
        self._base_path = base_path
        self._model_core = model_core

    def get_value_train_step(self, outputs, labels):
        """
        overload if you want to calculate other
        :param outputs: predict value from model
        :param labels: values of label (real answer)
        :return: calculated values list
        """
        return []

    def train(self, epoch=10000, lr=0.001):
        self._model_core.build_model()
        model = self._model_core.model
        optimizer = tf.keras.optimizers.Adam(lr=lr)

        train_data = self._model_core.get_train_data()

        for i in range(epoch):
            self._model_core.avg_logger.refresh()
            gen = list(train_data.get())

            for data in gen:
                inputs = data[0]
                labels = data[1]

                with tf.GradientTape() as tape:
                    outputs = model(inputs, training=True)
                    loss = self._model_core.loss_function.calculate(labels, outputs, axis=1)

                values = self.get_value_train_step(outputs, labels)

                grads = tape.gradient(loss, model.trainable_variables)  # calculate gradients
                optimizer.apply_gradients(zip(grads, model.trainable_variables))  # update gradients
                self._model_core.avg_logger.update_state([loss] + values)

            log_result = self._model_core.avg_logger.result()
            print('Epoch: {} {}'.format(i, log_result))

            # save weight every 100 epochs
            if i % 100 == 0 and i != 0:
                model.save_weights(os.path.join(self._base_path,
                                                './checkpoints/{}_{}.tf'.format(self.name, i)))

    def test(self, index):
        self._model_core.build_model()
        self._model_core.model.load_weights(os.path.join(self._base_path,
                                                         './checkpoints/{}_{}.tf'.format(self.name, index)))

        model = self._model_core.model
        test_data = self._model_core.get_test_data()

        self._model_core.avg_logger.refresh()
        all_outputs = []
        gen = list(test_data.get())
        for data in gen:
            inputs = data[0]
            labels = data[1]
            outputs = model(inputs, training=False)
            loss = self._model_core.loss_function.calculate(labels, outputs, axis=1)

            values = self.get_value_train_step(outputs, labels)

            self._model_core.avg_logger.update_state([loss] + values)

            all_outputs.extend(outputs)

        log_result = self._model_core.avg_logger.result_value()

        model.reset_states()

        return log_result, all_outputs

    def predict(self, index, data):
        self._model_core.build_model()
        self._model_core.model.load_weights(os.path.join(self._base_path,
                                                         './checkpoints/{}_{}.tf'.format(self.name, index)))

        return self._model_core.model(data, training=False)
