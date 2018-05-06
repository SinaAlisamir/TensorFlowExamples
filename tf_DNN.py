#!/usr/bin/python3
import numpy as np
import tensorflow as tf
from tf_model import tf_model
from data_model import data_model

class DNN(tf_model):
    def __init__(self, data: data_model, h_layers = [512,256]):
        super(DNN, self).__init__(data)
        self.h_layers = h_layers
        self.output = self.setup_model()

    def setup_model(self):
        W = self.Variable([self.data.input_dim, self.h_layers[0]])
        b = self.Variable([self.h_layers[0]])
        h = tf.nn.relu(tf.matmul(self.x, W) + b)
        for i in range(1,len(self.h_layers)):
            W = self.Variable([self.h_layers[i-1], self.h_layers[i]])
            b = self.Variable([self.h_layers[i]])
            h = tf.nn.relu(tf.matmul(h, W) + b)
        W = self.Variable([self.h_layers[-1], self.data.output_dim])
        b = self.Variable([self.data.output_dim])
        h = tf.nn.relu(tf.matmul(h, W) + b)
        return tf.nn.softmax(h)
