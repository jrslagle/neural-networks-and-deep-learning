# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:14:43 2018

@author: jrslagle
"""

### Libraries
# Standard Python 
import time
start_time = time.time()

# book's code
import mnist_loader

# my code
import network_nabla_distribution

shape, epochs, batch_size, learning_rate, seed = ([784, 2, 10], 10, 10, 3.0, 9)
print "shape = %s, epochs = %i, batch_size = %i, learning rate = %.1f, seed = %i" % (
    shape, epochs, batch_size, learning_rate, seed)


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network_nabla_distribution.Network(shape,seed)

training_batch = training_data[:3]
nablas = net.compile_nablas(training_batch)

example, layer, to_neuron, from_neuron = (50, 1, 5, 0)
#print "nabla context = \n%s" % nablas[example][layer][to_neuron]
#nabla = nablas[example][layer][to_neuron][from_neuron]
#print "In training example %i, layer %i, from neuron %i to neuron %i, the gradient on that weight is %.8f" % (example, layer, from_neuron, to_neuron, nabla)

stdevs = net.compute_stdevs(nablas)
#print stdevs.shape
#print "stdev context = \n%s" % stdevs[layer][to_neuron][from_neuron]
#stdev = stdevs[layer][to_neuron][from_neuron][example]
#print "In layer %i, from neuron %i to neuron %i, the stdev of that weight is %.8f" % (layer, from_neuron, to_neuron, stdev)

#vectors_on_weight = net.get_weight(nablas, index)
#net.export_csv(stdevs, "stdevs.csv")
#net.export_csv(vectors_on_weight, "vector_dist.csv")
#
#net.SGD(training_data, epochs, batch_size, learning_rate, test_data=test_data)
#learning_rate = learning_rate / 2
#net.SGD(training_data, epochs, batch_size, learning_rate, test_data=test_data)
#learning_rate = learning_rate / 2
#net.SGD(training_data, epochs, batch_size, learning_rate, test_data=test_data)
#learning_rate = learning_rate / 2
#net.SGD(training_data, epochs, batch_size, learning_rate, test_data=test_data)


end_time = time.time()
print "This exercise took %.1f s" % (end_time - start_time)
