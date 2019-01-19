# -*- coding: utf-8 -*-
"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import time

# Third-party libraries
import numpy as np

class Network(object):
    
    def __init__(self, sizes, seed=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        if seed is None:
            seed = np.random.randint(0,pow(2,32))
        self.master_seed = seed  # Max seed = 4,294,967,295
        seed_generator = np.random.RandomState(self.master_seed)
        bias_generator = np.random.RandomState(seed_generator.randint(0,pow(2,32)))
        weight_generator = np.random.RandomState(seed_generator.randint(0,pow(2,32)))
        self.shuffle_seeds = np.random.RandomState(seed_generator.randint(0,pow(2,32)))
        
        self.biases = [bias_generator.randn(y,1) for y in sizes[1:]]
        self.weights = [weight_generator.randn(y,x)
            for x,y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
        
    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            shuffle_seed = self.shuffle_seeds.randint(0,pow(2,32))
            shuffler = np.random.RandomState(shuffle_seed)
            shuffler.shuffle(training_data)

            # if there is any leftover, it goes in a smaller batch at the end.
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

#            stop = time.time()
#            train_time = stop - start
            if test_data:    
#                start = time.time()
                score = self.evaluate(test_data) / float(n_test)
#                stop = time.time()
#                test_time = stop - start
#                print "Epoch %d: score = %.4f, training time = %.3f s, testing time = %.3f s" % (
#                    j, score, train_time, test_time)
                print "%.4f" % (score)
            else:
                print "Epoch {0} complete".format(j)
                
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "training_batch" is a list of tuples "(x, y)". "(x, y) is a
        training example x and the answer, y. "learning_rate"
        is a coefficient applied to the error gradient."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
            for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
            for b, nb in zip(self.biases, nabla_b)]

    def compile_nablas(self, training_batch):
        """Regenerate all the nabla arrays using backprop for each training
        example in training_batch and store them in an array. The
        "training_batch" is a list of tuples "(x, y)". "(x, y) is a
        training example x and the answer, y."""

        # this block takes 0.0086 s to execute        
        nabla_list = []
        for x, y in training_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_list.append(np.array(delta_nabla_w))
            print type(nabla_list)
        print "Nabla list"
        print nabla_list
        print type(nabla_list)

        # nabla is a 4D array with format
        # nablas[example][layer][to_neuron][from_neuron]
        # size = training examples * weights
        nablas = np.vstack(nabla_list)
#        nablas = nablas.reshape((3,2,10,2))

        print "Nablas array is a %s with shape %s." % (type(nablas), nablas.shape)

        return nablas
    
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    def compute_stdevs(self, nablas):
        
        # nablas[example][layer][to_neuron][from_neuron]
        # stdevs[layer][to_neuron][from_neuron]
        stdevs = nablas.std(axis=0)
#        reshaped = []
#        
#        for example in nablas:
#            for layer in example:
#                for to_neuron in layer:
#                    for from_neuron in to_neuron:
##                        reshaped[layer][to_neuron][from_neuron][example] = nablas[example][layer][to_neuron][from_neuron]
#        
#        
        
        # size = weights
        return stdevs
        
    def get_weight(nablas, index):
        
        # size = len(training_data)
        return vectors_on_weight
        
    # This should probably be in a more general class
    def export_csv(complex_array, filename):
        return filename
    

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
