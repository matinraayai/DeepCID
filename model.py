# -*- coding: utf-8 -*-
"""
Author: Matin Raayai Ardakani
Starter code used form https://github.com/XiaqiongFan/DeepCID
Contains different models used in Chemometric analysis of RAMAN spectra.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPool1D
from tensorflow.keras.initializers import TruncatedNormal, Constant


class DeepCID(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(DeepCID, self).__init__()
        self.conv1 = Conv1D(filters=32,
                            kernel_size=5,
                            strides=2,
                            padding='same',
                            activation=tf.nn.relu,
                            kernel_initializer=TruncatedNormal(stddev=0.1),
                            bias_initializer=Constant(0.1))
        self.max_pool1 = MaxPool1D(2)
        self.dropout1 = Dropout(dropout_rate)
        self.conv2 = Conv1D(filters=64,
                            kernel_size=5,
                            strides=2,
                            padding='same',
                            activation=tf.nn.relu,
                            kernel_initializer=TruncatedNormal(stddev=0.1),
                            bias_initializer=Constant(0.1))
        self.max_pool2 = MaxPool1D(2)
        self.dropout2 = Dropout(dropout_rate)
        self.flat = Flatten()
        self.fc1 = Dense(1024, activation=tf.nn.relu)
        self.fc2 = Dense(2, activation=tf.nn.softmax)

    def call(self, x, training=False, **kwargs):
        x = tf.reshape(x, [-1, 881, 1])
        x = self.conv1(x)
        x = self.max_pool1(x)
        if training:
            x = self.dropout1(x, training)
        x = self.conv2(x)
        x = self.max_pool2(x)
        if training:
            x = self.dropout2(x, training)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


