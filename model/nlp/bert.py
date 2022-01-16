from model.core import ModelCore, LOSS
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf


class PretrainedBertClassifier(tf.keras.Model):
    def __init__(self, bert, num_classes):
        super(PretrainedBertClassifier, self).__init__()

        self.bert = bert
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_classes
                                                , kernel_initializer=tf.keras.initializers.TruncatedNormal(
                self.bert.config.initializer_range)
                                                , activation=tf.keras.activations.softmax)

    def call(self, inputs, training=False):
        outputs = self.bert(inputs[0], attention_mask=inputs[1], token_type_ids=inputs[2])
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        return logits


class PretrainedBert(ModelCore):
    def __init__(self, data_path, save_path, max_length, num_classes, pretrained_model_name):
        self.max_length = max_length
        self.num_classes = num_classes
        self.bert = TFBertModel.from_pretrained(pretrained_model_name, from_pt=True)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        ModelCore.__init__(self, data_path=data_path, save_path=save_path, loss=LOSS.CATEGORICAL_CROSSENTROPY, train_test_ratio=0.8,
                           is_classify=True, input_dtype=tf.int32, batch_size=256)

    def build_model(self):
        self.model = PretrainedBertClassifier(self.bert, self.num_classes)

    def read_data(self):
        self._data_all = []
        with open(self._data_path, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                content_split = line.strip().split('\t')

                if len(content_split) != 2:
                    continue

                input_text = content_split[0]
                label = content_split[1]

                zero = [0] * self.num_classes
                zero[int(label)] = 1

                input_ids, attention_mask, token_type_ids = self.tokenize(input_text)

                self._data_all.append({'input': {0: input_ids, 1: attention_mask, 2: token_type_ids}, 'output': zero})

    def tokenize(self, text):
        encoded_dic = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True
        )

        return encoded_dic['input_ids'], encoded_dic['attention_mask'], encoded_dic['token_type_ids']