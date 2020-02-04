# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:14:43 2018

@author: jrslagle
"""

### Libraries
# Standard Python 
import time

# book's code
import mnist_loader
import network_mixed_seeds

shape = [784, 100, 10]
epochs, batch_size, learning_rate, seed = (40, 10, 3.0, 0)
print "shape = %s, epochs = %i, batch_size = %i, learning rate = %.1f" % (shape, epochs, batch_size, learning_rate)

while True:
    start_time = time.time()
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    learning_rate = 3.0
    net = network_mixed_seeds.Network(shape,seed)
    net.SGD(training_data, epochs/4, batch_size, learning_rate, test_data=test_data)
    learning_rate = learning_rate / 2
    net.SGD(training_data, epochs/4, batch_size, learning_rate, test_data=test_data)
    learning_rate = learning_rate / 2
    net.SGD(training_data, epochs/4, batch_size, learning_rate, test_data=test_data)
    learning_rate = learning_rate / 2
    score = net.SGD(training_data, epochs/4, batch_size, learning_rate, test_data=test_data)
    end_time = time.time()
    print "%i,%s,%.1f" % (seed, score, end_time - start_time)
    seed += 1
