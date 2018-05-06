#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import scipy.ndimage
from tf_DNN import DNN
from tf_RNN import RNN
from tf_CNN import CNN
from data_model import data_model

theData = data_model()

# theModel = DNN(theData, h_layers = [512,256,128])
# theModel.train(max_epochs=100, batch_size=5000, l_rate=1e-03)

theModel = RNN(theData, 28, rnn_size=128)
theModel.train(max_epochs=100, batch_size=5000, l_rate=1e-03)

# theModel = CNN(theData)
# theModel.train(max_epochs=100, batch_size=5000, l_rate=1e-03)


theModel.save_dir='./model_trained'
theModel.restore_model()
dat = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("test1.png", flatten=True)))
theModel.test_input([dat])
