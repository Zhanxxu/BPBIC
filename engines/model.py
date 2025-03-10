# -*- coding: utf-8 -*-

from abc import ABC

import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood
from engines.utils.dice import DiceLoss
from keras.callbacks import TensorBoard
from tensorflow.python.ops import summary_ops_v2
import datetime

class NerModel(tf.keras.Model, ABC):
    def __init__(self, configs, vocab_size,num_classes):
        super(NerModel, self).__init__()
        self.use_pretrained_model = configs.use_pretrained_model
        self.finetune = configs.finetune
        if self.use_pretrained_model and self.finetune:
            if configs.pretrained_model == 'Bert':
                from transformers import TFBertModel
                self.pretrained_model = TFBertModel.from_pretrained('bert-base-chinese')
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, configs.embedding_dim, mask_zero=True)

        self.use_middle_model = configs.use_middle_model
        self.middle_model = configs.middle_model
        if self.use_middle_model:
            if self.middle_model == 'bilstm':
                self.hidden_dim = configs.hidden_dim
                self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))

            if self.middle_model == 'idcnn':
                filter_nums = configs.filter_nums
                self.idcnn_nums = configs.idcnn_nums
                self.cnn = tf.keras.Sequential([
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=2)])
                self.idcnn = [self.cnn for _ in range(self.idcnn_nums)]

            if self.middle_model == 'bilstm+idcnn':
                filter_nums = configs.filter_nums
                self.idcnn_nums = configs.idcnn_nums
                self.cnn = tf.keras.Sequential([
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=2)])
                self.idcnn = [self.cnn for _ in range(self.idcnn_nums)]
                self.hidden_dim = configs.hidden_dim
                self.bilstm = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))

            if self.middle_model == 'idcnn+bilstm':
                self.hidden_dim = configs.hidden_dim
                self.bilstm = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
                filter_nums = configs.filter_nums
                self.idcnn_nums = configs.idcnn_nums
                self.cnn = tf.keras.Sequential([
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=2)])
                self.idcnn = [self.cnn for _ in range(self.idcnn_nums)]

            if self.middle_model == 'bilstm+idcnn-U':
                self.hidden_dim = configs.hidden_dim
                self.bilstm = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
                filter_nums = configs.filter_nums
                self.idcnn_nums = configs.idcnn_nums
                self.cnn = tf.keras.Sequential([
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=2)])
                self.idcnn = [self.cnn for _ in range(self.idcnn_nums)]

            if self.middle_model == 'bilstm+idcnn-1':
                #底层并联
                self.hidden_dimes = configs.hidden_dimes
                self.bilstmes = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.hidden_dimes,return_sequences=True))
                filter_nums = configs.filter_nums
                self.idcnn_numes = configs.idcnn_numes
                self.cnns = tf.keras.Sequential([
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=2)])
                self.idcnns = [self.cnns for _ in range(self.idcnn_numes)]
                #上层串联
                self.hidden_dim = configs.hidden_dim
                self.bilstm = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.hidden_dim,return_sequences=True))
                filter_nums = configs.filter_nums
                self.idcnn_nums = configs.idcnn_nums
                self.cnn = tf.keras.Sequential([
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=1),
                    tf.keras.layers.Conv1D(filters=filter_nums, kernel_size=3, activation='relu',
                                           padding='same', dilation_rate=2)])
                self.idcnn = [self.cnn for _ in range(self.idcnn_nums)]
                #self.hidden_dim = configs.hidden_dim
                #self.bilstm = tf.keras.layers.Bidirectional(
                 #   tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))


        self.dropout_rate = configs.dropout
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense = tf.keras.layers.Dense(num_classes)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(num_classes, num_classes)))


    @tf.function
    def call(self, inputs, inputs_length, targets, training=None):

        if self.use_pretrained_model:
            if self.finetune:
                embedding_inputs = self.pretrained_model(inputs[0], attention_mask=inputs[1])[0]
            else:
                embedding_inputs = inputs
        else:
            embedding_inputs = self.embedding(inputs)

        outputs = self.dropout(embedding_inputs, training)

        if self.use_middle_model:

            if self.middle_model == 'bilstm':
                outputs = self.bilstm(outputs)
                #outputs = self.attention(inputs_length,bioutputs)
            if self.middle_model == 'idcnn':
                cnn_outputs = [idcnn(outputs) for idcnn in self.idcnn]
                if self.idcnn_nums == 1:
                    outputs = cnn_outputs[0]
                else:
                    outputs = tf.keras.layers.concatenate(cnn_outputs, axis=-1, name='concatenate')
            if self.middle_model == 'bilstm+idcnn':
                #串联
                outputs = self.bilstm(outputs)
                cnn_outputs = [idcnn(outputs) for idcnn in self.idcnn]
                if self.idcnn_nums == 1:
                    outputs = cnn_outputs[0]
                else:
                    outputs = tf.keras.layers.concatenate(cnn_outputs, axis=-1, name='concatenate')
            if self.middle_model == 'idcnn+bilstm':
                # 串联
                cnn_outputs = [idcnn(outputs) for idcnn in self.idcnn]
                if self.idcnn_nums == 1:
                    outputs = cnn_outputs[0]
                else:
                    outputs = tf.keras.layers.concatenate(cnn_outputs, axis=-1, name='concatenate')
                outputs = self.bilstm(outputs)

            if self.middle_model == 'bilstm+idcnn-U':
                #并联
                bioutputs = self.bilstm(outputs)
                cnn_outputs = [idcnn(outputs) for idcnn in self.idcnn]
                if self.idcnn_nums == 1:
                    outputs = cnn_outputs[0]
                else:
                    outputs = tf.keras.layers.concatenate(cnn_outputs, axis=-1, name='concatenate')
                outputs = tf.keras.layers.concatenate([bioutputs,outputs],axis=-1,name='concatenate')

            if self.middle_model == 'bilstm+idcnn-1':
                #bioutputs = self.bilstm(outputs)
                #bdoutputs = tf.keras.layers.concatenate([bioutputs,outputs],axis=-1,name='concatenate')
                #outputs = tf.keras.layers.concatenate([bdoutputs, bioutputs], axis=-1, name='concatenate')
                #shape(8,512,1024)

                #shape(None,None,768)
                outputs = self.bilstmes(outputs)
                cnn_outputs = [idcnns(outputs) for idcnns in self.idcnns]
                if self.idcnn_nums == 1:
                    outputs = cnn_outputs[0]
                else:
                    outputs = tf.keras.layers.concatenate(cnn_outputs, axis=-1, name='concatenate')
                bioutputs = self.bilstmes(outputs)
                bdoutputs = tf.keras.layers.concatenate([bioutputs,outputs],axis=-1,name='concatenate')
                #outputs = tf.keras.layers.concatenate([bdoutputs, bioutputs], axis=-1, name='concatenate')

                outputs = self.bilstm(bdoutputs)
                cnn_outputs = [idcnn(outputs) for idcnn in self.idcnn]
                if self.idcnn_nums == 1:
                    outputs = cnn_outputs[0]
                else:
                    outputs = tf.keras.layers.concatenate(cnn_outputs, axis=-1, name='concatenate')


        logits = self.dense(outputs)
        tensor_targets = tf.convert_to_tensor(targets, dtype=tf.int32)

        #loss = DiceLoss(tensor_targets,logits)

        log_likelihood, self.transition_params = crf_log_likelihood(
            logits, tensor_targets, inputs_length, transition_params=self.transition_params)
        return logits, log_likelihood, self.transition_params

