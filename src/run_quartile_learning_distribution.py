# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:14:43 2018

@author: jrslagle
"""

### Libraries
# Standard Python 
#import time

# book's code
import mnist_loader
import network_fixed

shape = [784, 100, 10]
epochs, batch_size, learning_rate, seed = (40, 10, 3.0, 0)
print "shape = %s, epochs = %i, batch_size = %i, learning rate = %.1f" % (shape, epochs, batch_size, learning_rate)

while True:
#    start_time = time.time()
    learning_rate = 3.0
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network_fixed.Network(shape,seed)
    score00 = net.evaluate(test_data)
    
    net.SGD(training_data, epochs/20, batch_size, learning_rate, test_data=test_data)
    score05 = net.evaluate(test_data)    

    net.SGD(training_data, epochs/20, batch_size, learning_rate, test_data=test_data)
    score10 = net.evaluate(test_data)    

    net.SGD(training_data, epochs/20, batch_size, learning_rate, test_data=test_data)
    score15 = net.evaluate(test_data)    

    net.SGD(training_data, epochs/20, batch_size, learning_rate, test_data=test_data)
    score20 = net.evaluate(test_data)
    
    net.SGD(training_data, epochs/20, batch_size, learning_rate, test_data=test_data)
    score25 = net.evaluate(test_data)
    
    learning_rate = learning_rate / 2
    net.SGD(training_data, epochs/4, batch_size, learning_rate, test_data=test_data)
    score50 = net.evaluate(test_data)

    learning_rate = learning_rate / 2
    net.SGD(training_data, epochs/4, batch_size, learning_rate, test_data=test_data)
    score75 = net.evaluate(test_data)

    learning_rate = learning_rate / 2
    net.SGD(training_data, epochs/4, batch_size, learning_rate, test_data=test_data)
    score100 = net.evaluate(test_data)

    learning_rate = learning_rate / 2
    net.SGD(training_data, epochs/4, batch_size, learning_rate, test_data=test_data)
    score100 = net.evaluate(test_data)
    
#    total_time = time.time() - start_time
#    print "%i,%.4f,%.4f" % (seed, score00, score05)
    print "%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (seed, score00, score05, score10, score15, score20, score25, score50, score75, score100)
    seed += 1
