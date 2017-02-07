import os
os.chdir(r'C:\Users\Kyle\workspace\CSC421\A4\neural-networks-and-deep-learning-master\neural-networks-and-deep-learning-master\src')

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network

net = network.Network([784, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)