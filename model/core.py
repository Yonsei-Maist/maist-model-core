import tensorflow as tf
import tensorflowjs as tfjs
from enum import Enum
import math
import os
import random
from abc import *


class LOSS(Enum):
    MSE = 1
    COSINE_SIMILARITY = 2
    CATEGORICAL_CROSSENTROPY = 3
    BINARY_CROSSENTROPY = 4
    SPARSE_CATEGORICAL_CROSSENTROPY = 5


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
            self.__lambda = lambda labels, outputs, axis: tf.keras.losses.MSE(labels, outputs)
        elif loss == LOSS.COSINE_SIMILARITY:
            self.__lambda = lambda labels, outputs, axis: tf.keras.losses.cosine_similarity(labels, outputs, axis)
        elif loss == LOSS.BINARY_CROSSENTROPY:
            self.__lambda = lambda labels, outputs, axis: tf.keras.losses.binary_crossentropy(labels, outputs, axis)
        elif loss == LOSS.BINARY_CROSSENTROPY:
            self.__lambda = lambda labels, outputs, axis: tf.keras.losses.sparse_categorical_crossentropy(labels, outputs, axis)
        else:  # default
            self.__lambda = lambda labels, outputs, axis: tf.keras.losses.categorical_crossentropy(labels, outputs,
                                                                                                   axis)

    def calculate(self, labels, outputs, axis):
        return self.__lambda(labels, outputs, axis)


class Dataset:
    def __init__(self, batch_size, model=None):
        self.__inputs = None
        self.__labels = None
        self.__origins = None
        self.__batch_size = batch_size
        self.__model = model

    def set(self, inputs, labels, origin_file=None):
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
        self.__origins = origin_file

    def get(self):
        iter = math.ceil(len(self) / self.__batch_size)

        transform = None

        output_transform = lambda item: tf.convert_to_tensor(item, dtype=tf.float32)

        def transform(model, inputs):

            if model is None:
                return [tf.convert_to_tensor(item, dtype=tf.float32) for item in inputs]
            else:
                len_batch = len(inputs[0])
                batch_result = []

                for i in range(len_batch):
                    res = model.data_transform([item[i] for item in inputs])
                    for j in range(len(res)):
                        if len(batch_result) <= j:
                            batch_result.append([])

                        batch_result[j].append(res[j])

                return [tf.convert_to_tensor(item, dtype=tf.float32) for item in batch_result]

        for i in range(iter):
            raw_input = [item[i * self.__batch_size: i * self.__batch_size + self.__batch_size] for item in
                       self.__inputs]
            if len(self.__labels) == 1:
                yield transform(self.__model, raw_input), \
                      output_transform(self.__labels[0][i * self.__batch_size: i * self.__batch_size + self.__batch_size])
            else:
                yield transform(self.__model, raw_input), \
                      [output_transform(item[i * self.__batch_size: i * self.__batch_size + self.__batch_size]) for item in self.__labels]

    def get_origin(self):
        if self.__origins is None:
            raise Exception("the origin file is empty")

        iter = math.ceil(len(self) / self.__batch_size)
        for i in range(iter):
            yield self.__origins[i * self.__batch_size: i * self.__batch_size + self.__batch_size]

    def __len__(self):
        if len(self.__inputs) > 0:
            return len(self.__inputs[0])

        return 0


class DatasetFactory:
    @staticmethod
    def get_dtype(value):
        if isinstance(value, int):
            return tf.int32
        else:
            return tf.float32

    @staticmethod
    def make_dataset(train_data, test_data, data_all, sp_ratio, is_classify):

        item_one = data_all[0]

        data_train = []
        data_test = []

        if is_classify:
            dic_label = {}

            for data in data_all:
                zero_list = data['output']  # it is zero-base label
                label_one = zero_list.index(max(zero_list))
                if label_one in dic_label:
                    dic_label[label_one].append(data)
                else:
                    dic_label[label_one] = [data]

            for key in dic_label.keys():
                data_arr = dic_label[key]
                sp = int(len(data_arr) * sp_ratio)
                random.shuffle(data_arr)

                data_train.extend(data_arr[:sp])
                data_test.extend(data_arr[sp:])
        else:
            sp = int(len(data_all) * sp_ratio)
            data_train = data_all[:sp]
            data_test = data_all[sp:]

        input_train = None
        input_test = None
        output_train = None
        output_test = None
        origin_train = None
        origin_test = None

        if isinstance(item_one['input'], dict):
            input_train = [[item['input'][i] for item in data_train] for i in range(len(item_one['input'].keys()))]
            input_test = [[item['input'][i] for item in data_test] for i in range(len(item_one['input'].keys()))]
        else:
            input_train = [[item['input'] for item in data_train]]
            input_test = [[item['input'] for item in data_test]]

        if isinstance(item_one['output'], dict):
            output_train = [[item['output'][i] for item in data_train] for i in range(len(item_one['output'].keys()))]
            output_test = [[item['output'][i] for item in data_test] for i in range(len(item_one['output'].keys()))]
        else:
            output_train = [[item['output'] for item in data_train]]
            output_test = [[item['output'] for item in data_test]]

        if 'origin' in item_one:
            origin_train = [item['origin'] for item in data_train]
            origin_test = [item['origin'] for item in data_test]

        train_data.set(input_train, output_train, origin_train)
        test_data.set(input_test, output_test, origin_test)

        return train_data, test_data


class ModelCore(metaclass=ABCMeta):
    def __init__(self, data_path, batch_size=64, avg_list=['loss'], loss=LOSS.CATEGORICAL_CROSSENTROPY, train_test_ratio=0.8, is_classify=False):
        self._train_data = Dataset(batch_size)
        self._test_data = Dataset(batch_size)
        self.model = None
        self.batch_size = batch_size
        self._data_path = data_path
        self.avg_logger = AvgLogger(avg_list)
        self._data_all = None
        self._train_test_ratio = train_test_ratio
        self._is_classify = is_classify

        self.is_multi_output = isinstance(loss, list)

        if self.is_multi_output:
            self._loss_function_list = [LossFunction(loss_function) for loss_function in loss]
        else:
            self._loss_function = LossFunction(loss)

        self.read_data()
        self.make_dataset()
        self.build_model()

    def calculate_loss_function(self, labels, outputs, axis):
        if self.is_multi_output:
            if len(self._loss_function_list) != len(labels) and len(labels) != len(outputs):
                raise Exception("unmatch length of labels and outputs.")
            return [loss_function.calculate(labels[i], outputs[i], axis) for i, loss_function in
                    enumerate(self._loss_function_list)]
        else:
            return self._loss_function.calculate(labels, outputs, axis)

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

    def make_dataset(self):
        self._train_data, self._test_data = DatasetFactory.make_dataset(self._train_data, self._test_data, self._data_all, self._train_test_ratio, self._is_classify)

    def data_transform(self, item):
        return item

    def load_weight(self):
        """
        load weight without checkpoint
        """
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

    def save_when(self, epoch, result_values):
        """
        condition when save
        :param epoch: current epoch
        :param result_values: current loss and logger result value
        :return: true or false
        """
        return False

    def train(self, pretrained_module_name=None, pretrained_module_index=None, epoch=10000, lr=0.001):
        self._model_core.build_model()
        if pretrained_module_name is not None and pretrained_module_index is not None:
            self._model_core.model.load_weights(os.path.join(self._base_path,
                                                './checkpoints/{0}_{1}.tf'.format(pretrained_module_name, pretrained_module_index)))
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
                    loss = self._model_core.calculate_loss_function(labels, outputs, axis=1)

                values = self.get_value_train_step(outputs, labels)

                grads = tape.gradient(loss, model.trainable_variables)  # calculate gradients
                optimizer.apply_gradients(zip(grads, model.trainable_variables))  # update gradients

                if isinstance(loss, list):
                    self._model_core.avg_logger.update_state(loss + values)
                else:
                    self._model_core.avg_logger.update_state([loss] + values)

            log_result = self._model_core.avg_logger.result()
            print('Epoch: {} {}'.format(i, log_result))

            # save weight every 100 epochs
            if (i % 100 == 0 and i != 0) or self.save_when(i, self._model_core.avg_logger.result_value()):
                model.save_weights(os.path.join(self._base_path,
                                                './checkpoints/{}_{}.tf'.format(self.name, i)))

    def test(self, index):
        self._model_core.build_model()
        if index > -1:
            self._model_core.model.load_weights(os.path.join(self._base_path,
                                                             './checkpoints/{}_{}.tf'.format(self.name, index)))
        else:
            self._model_core.load_weight()

        model = self._model_core.model
        test_data = self._model_core.get_test_data()

        self._model_core.avg_logger.refresh()
        all_outputs = []
        gen = list(test_data.get())
        for data in gen:
            inputs = data[0]
            labels = data[1]
            outputs = model(inputs, training=False)

            loss = self._model_core.calculate_loss_function(labels, outputs, axis=1)

            values = self.get_value_train_step(outputs, labels)

            if isinstance(loss, list):
                self._model_core.avg_logger.update_state(loss + values)
            else:
                self._model_core.avg_logger.update_state([loss] + values)

            all_outputs.extend(outputs)

        log_result = self._model_core.avg_logger.result_value()

        model.reset_states()

        return log_result, all_outputs

    def predict(self, index, data):
        self._model_core.build_model()

        if index > -1:
            self._model_core.model.load_weights(os.path.join(self._base_path,
                                                             './checkpoints/{}_{}.tf'.format(self.name, index)))
        else:
            self._model_core.load_weight()

        return self._model_core.model(data, training=False)

    def get_test_result(self, index):

        avg_loss, res = self.test(index)  # means test all of data using {index}'th checkpoint

        ldl_c_d_for_graph = []
        data_for_graph = []
        for i in range(len(res)):
            predict_index = int(tf.math.argmax(res[i]))
            ldl_c_d_for_graph.append(predict_index)

        gen = list(self._model_core.get_test_data().get())

        for data in gen:
            for i in range(len(data[1])):
                data_for_graph.append(int(tf.math.argmax(data[1][i])))

        return data_for_graph, ldl_c_d_for_graph

    def save_to_js(self, index, path):
        self._model_core.build_model()
        if index > -1:
            self._model_core.model.load_weights(os.path.join(self._base_path,
                                                             './checkpoints/{}_{}.tf'.format(self.name, index)))
        else:
            self._model_core.load_weight()

        tfjs.converters.save_keras_model(self._model_core.model, path)
