
### Libraries
# Standard Python 
import time

# my code
import network_manager
import hyper

# load training data
start = time.time()

# Monte Carlo search for best network
constraints = hyper.Hyper(epochs=30, batch_size=10)
manager = network_manager.Network_Manager()
manager.add_hypers(constraints=constraints)

#net.save_network('mynetwork.net')

#net2 = network.load_network('mynetwork.net')

#print net
#print net2

stop = time.time()
print "All %i epochs took %.1f seconds" % (epochs, stop - start)