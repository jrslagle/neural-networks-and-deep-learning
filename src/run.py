
### Libraries
# Standard Python 
import time

# book's code
import mnist_loader
import network

start = time.time()

# load training data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# initialize network
net = network.Network([784, 10])

# train network
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

stop = time.time()
print "All %i epochs took %.1f seconds" % (epochs, stop - start)
