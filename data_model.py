#!/usr/bin/python3
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class data_model():
    def __init__(self, path="./data/"):
        super(data_model, self).__init__()
        self.mnist = input_data.read_data_sets(path, one_hot=True)
        self.num_train = self.mnist.train.num_examples
        self.input_dim = 28*28
        self.output_dim = 10
        self.counter = 0

    def train_input(self):
        return self.mnist.train.images
    def train_output(self):
        return self.mnist.train.labels
    def test_input(self):
        return self.mnist.test.images
    def test_output(self):
        return self.mnist.test.labels

    def next_batch(self, batch_size):
        rng = range(self.counter*batch_size, (self.counter+1)*batch_size)
        self.counter += 1
        if self.counter == self.num_train//batch_size: self.counter=0
        return self.train_input()[rng], self.train_output()[rng]
