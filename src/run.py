import mnist_loader
import sandbox
import time

# load training data
start = time.time()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

hypers = [[784,30,10], 4294967295, 4, 10, 3.0]
sizes = hypers[0]
seed = hypers[1]
epochs = hypers[2]
batch_size = hypers[3]
learning_rate = hypers[4]

# initialize network
# seed must be between 0 and 4294967295 (2^32 - 1)
net = sandbox.Network(sizes,seed)

# train network
net.SGD(training_data, epochs, batch_size, learning_rate, test_data=test_data)

#net.save_network('mynetwork.net')

#net2 = network.load_network('mynetwork.net')

#print net
#print net2

stop = time.time()
print "All %i epochs took %.1f seconds" % (epochs, stop - start)