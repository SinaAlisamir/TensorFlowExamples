#!/usr/bin/python3
import numpy as np
import tensorflow as tf
from data_model import data_model

class tf_model():
    def __init__(self, data: data_model, save_dir='./model'):
        super(tf_model, self).__init__()
        self.data = data
        self.save_dir = save_dir
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, self.data.input_dim])
        self.y = tf.placeholder(tf.float32)
        self.output = self.setup_default_nn()

    def setup_default_nn(self):
        hidden_num = 100
        W1 = self.Variable([self.data.input_dim, hidden_num])
        b1 = self.Variable([hidden_num])
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        W2 = self.Variable([hidden_num, self.data.output_dim])
        b2 = self.Variable([self.data.output_dim])
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
        return tf.nn.softmax(h2)

    def Variable(self, inputs):
        return tf.Variable(tf.random_normal(inputs, dtype=tf.float32))

    def train(self, max_epochs=100, batch_size=100, l_rate=1e-03):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
        optimizer = tf.train.AdamOptimizer(l_rate).minimize(cost)
        correct = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        self.sess.run(tf.global_variables_initializer())
        train_dic = {self.x:self.data.train_input(), self.y:self.data.train_output()}
        test_dic = {self.x:self.data.test_input(), self.y:self.data.test_output()}
        for epoch in range(max_epochs):
            epoch_loss = 0
            for _ in range(self.data.num_train//batch_size):
                x, y = self.data.next_batch(batch_size)
                _, loss = self.sess.run([optimizer, cost], feed_dict=train_dic)
                epoch_loss += loss
            self.save_model()
            print("Epoch", epoch+1, "/", max_epochs, 'epoch_loss:', epoch_loss, \
             'train_accuracy:', self.sess.run(accuracy, feed_dict=train_dic), \
             'test_accuracy:', self.sess.run(accuracy, feed_dict=test_dic))
        print('Accuracy:', self.sess.run(accuracy, feed_dict=test_dic))

    def save_model(self):
        self.saver = tf.train.Saver(max_to_keep=1)
        self.saver.save(self.sess, self.save_dir+'/model.ckpt')

    def restore_model(self):
        self.saver = tf.train.Saver(max_to_keep=1)
        self.saver.restore(self.sess, self.save_dir+'/model.ckpt')

    def test_input(self, test_input):
        dict = {self.x:test_input}
        result = self.sess.run(tf.argmax(self.output,1), feed_dict=dict)
        print("Test Result:", result)
