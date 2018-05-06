#!/usr/bin/python3
import numpy as np
import tensorflow as tf
from tf_model import tf_model
from data_model import data_model

class RNN(tf_model):
    def __init__(self, data: data_model, window_size, rnn_size=128):
        super(RNN, self).__init__(data)
        self.window_size = window_size
        self.num_windows = self.data.input_dim//self.window_size
        self.rnn_size = rnn_size
        self.output = self.setup_model()

    def setup_model(self):
        x = tf.reshape(self.x, [-1, self.num_windows, self.window_size])
        x = tf.transpose(x, [1,0,2])
        x = tf.reshape(x, [-1, self.window_size])
        x = tf.split(x, self.num_windows, 0)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size,state_is_tuple=True)
        outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
        W = self.Variable([self.rnn_size, self.data.output_dim])
        b = self.Variable([self.data.output_dim])
        output = tf.matmul(outputs[-1],W) + b
        return output
