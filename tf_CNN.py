#!/usr/bin/python3
import numpy as np
import tensorflow as tf
from tf_model import tf_model
from data_model import data_model

class CNN(tf_model):
    def __init__(self, data: data_model):
        super(CNN, self).__init__(data)
        self.output = self.setup_model()

    def setup_model(self):
        hidden_num = 1024
        filter_num = 4
        x = tf.reshape(self.x, shape=[-1, 28, 28, 1])
        W = self.Variable([5,5,1,filter_num])
        b = self.Variable([filter_num])
        conv = tf.nn.relu(self.conv2d(x, W) + b)
        pool = self.maxpool2d(conv)
        # W = self.Variable([5,5,filter_num,filter_num])
        # b = self.Variable([filter_num])
        # conv = tf.nn.relu(self.conv2d(conv, W) + b)
        # pool = self.maxpool2d(conv)
        dim = pool.get_shape().as_list()
        dim = np.prod(dim[1:])
        W = self.Variable([dim, hidden_num])
        b = self.Variable([hidden_num])
        hidden = tf.reshape(pool, [-1, dim])
        hidden = tf.nn.relu(tf.matmul(hidden, W) + b)
        W = self.Variable([hidden_num, self.data.output_dim])
        b = self.Variable([self.data.output_dim])
        output = tf.nn.softmax(tf.matmul(hidden, W) + b)
        return output

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def maxpool2d(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
