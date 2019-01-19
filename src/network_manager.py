# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:08:57 2018

@author: jrslagle
"""
# book's code
import mnist_loader

# my code
import sandbox


# hyper refers to the list of hyper parameters which can recreate an instance
# of Network exactly, from initialization, through training, to trained.

# network refers to a saved Network, with the weights and biases. it's
# intended to save a trained network.

class Network_Manager(object):
    def __init__(self):
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        print "Loaded MNIST data in %.2f sec" % (time.time()-start)

    """    
    Endlessly generates random hypers for all non-fixed parameters, trains that
    network, saves its hyper (w/ score) to the registry, and saves the 
    network if its at or above min_save_score.
    """
    def add_hypers(hyper, min_save_score=None):
        # initialize network
        # seed must be between 0 and 4294967295 (2^32 - 1)
        net = sandbox.Network(sizes,seed)
        
        # train network
        net.SGD(training_data, epochs, batch_size, learning_rate, test_data=test_data)

    
    """
    Searches the registry in 'network_scores.dat' using hyper and
    returns a list of Hyper's that match.
    """
    def get_hypers(hyper):
        
    """
    Searches the folder of network files and returns a list of filenames
    that match hyper.
    """
    def get_networks(hyper):
    
    """
    Called by add_hypers. It saves the current trained network to a file.
    """
    def save_network(network):
        
    def count_hypers(hyper):        
        
    def count_networks(hyper):
        
    def delete_networks(hyper):
        