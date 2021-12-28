from model.core import ModelCore, LOSS
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf


class PretrainedBert(ModelCore):
    def __init__(self, data_path, max_length, num_classes, pretrained_model_name):
        self.max_length = max_length
        self.num_classes = num_classes
        self.bert = TFBertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        super().__init__(data_path=data_path, loss=LOSS.SPARSE_CATEGORICAL_CROSSENTROPY)

    def build_model(self):
        inputs = tf.keras.Input([self.max_length])
        attention_mask = tf.keras.Input([self.max_length])
        token_type_ids = tf.keras.Input([self.max_length])

        bert_output = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = bert_output[1]
        drop_out = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)(pooled_output)
        outputs = tf.keras.layers.Dense(self.num_classes,
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range),
                                        actiavtion=tf.keras.activations.softmax)(drop_out)

        self.model = tf.keras.Model(inputs=[inputs, attention_mask, token_type_ids], outputs=[outputs])

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

                input_ids, attention_mask, token_type_ids = self.tokenize(input_text)

                self._data_all.append([{0: input_ids, 1: attention_mask, 2: token_type_ids}, label])

    def tokenize(self, text):
        encoded_dic = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_attention_mask=True
        )

        return encoded_dic['input_id'], encoded_dic['attention_mask'], encoded_dic['token_type_ids']
