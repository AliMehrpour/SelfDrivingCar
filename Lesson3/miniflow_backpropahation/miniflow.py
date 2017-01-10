import numpy as np

class Layer:
	def __init__(self, inbound_layers=[]):
		# Layer from which this layer receives value
		self.inbound_layers = inbound_layers
		# Layer to which this layer passes value
		self.outbound_layers = []
		# Gradient: Keys are the input to this layer and 
		# their values are the partials of this layer with respect to that input
		self.gradients = {}
		# Calculated value
		self.value = None
		# For each inbound layer here, add this layer as an outbound layer
		for layer in self.inbound_layers:
			layer.outbound_layers.append(self)

		def forward(self):
			"""
			Each layer that uses this class as a base class will need
			to define its own 'forward' method
			"""
			raise NotImplemented

		def backward(self):
			"""
			Each layer that uses this class as a base class will need
			to define its own 'backward' method
			"""
			raise NotImplemented

class Input(Layer):
	def __init__(self):
		# An Input neuron has no inbound neurons, so no need to pass anything to Neuron constructor
	    Layer.__init__(self)

	# Input neuron in the only node where value may be passed as an argument to forward()
	def forward(self, value=None):
		# Do nothing because nothing is calculated
		pass

	def backward(self):
		# An input layer has no inputs so gradient(derivative) is zero.
		# The key, self, is reference to this object
		self.gradients = {self: 0}

		# Weights and bias may be inputs, so we need to sum
		# the gradient from output gradients
		for n in self.outbound_layers:
			grad_cost = n.gradients[self]
			self.gradients[self] += grad_cost * 1

class Linear(Layer):
	def __init__(self, X, W, b):
		# Weights and bias treat like an inbound layers
		Layer.__init__(self, [X, W, b])

	def forward(self):
		"""
		Performs the math behind a linear transform
		Z = XW + b
		"""
		X = self.inbound_layers[0].value
		W = self.inbound_layers[1].value
		b = self.inbound_layers[2].value

		self.value = np.dot(X, W) + b
		
	def backward(self):
		"""
		Calculates the gradient based on outputs
		"""
		# Initialize a partial for each of inbound_layers
		self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}

		# Cycle through the outputs. Graddient will change depending on each output,
		# So the gradient will summed over all outputs
		for n in self.outbound_layers:
			# Get partial of cost with respect to this layer
			grad_cost = n.gradients[self]
			# Set the partial of the loss w.r.t this layer's inputs
			self.gradients[self.inbound_layers[0]] += np.dot(grad_cost, self.inbound_layers[1].value.T)
			# Set the partial of the loss w.r.t this layer's weights
			self.gradients[self.inbound_layers[1]] += np.dot(self.inbound_layers[0].value.T, grad_cost)
			# Set the partial of the loss w.r.t. this layer's bias
			self.gradients[self.inbound_layers[2]] += np.sum(grad_cost, axis=0,  keepdims=False)

class Sigmoid(Layer):
	def __init__(self, layer):
		Layer.__init__(self, [layer])

	def _sigmoid(self, x):
		return 1. / (1. + np.exp(-x))

	def forward(self):
		input_value = self.inbound_layers[0].value
		self.value = self._sigmoid(input_value)

	def backward(self):
		# Initialize gradient to zero
		self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}

		# Cycle through the outputs
		for n in self.outbound_layers:
			grad_cost = n.gradients[self]
			sigmoid = self.value
			self.gradients[self.inbound_layers[0]] = sigmoid * (1 - sigmoid) * grad_cost

class MSE(Layer):
	def __init__(self, y, a):
		Layer.__init__(self, [y, a])

	def forward(self):
		"""
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
		y = self.inbound_layers[0].value.reshape(-1, 1)
		a = self.inbound_layers[1].value.reshape(-1, 1)

        # Save the computed output for backward.
		self.m = self.inbound_layers[0].value.shape[0]
		self.diff = y - a
		self.value = np.mean(self.diff**2)

	def backward(self):
		self.gradients[self.inbound_layers[0]] = (2 / self.m) * self.diff
		self.gradients[self.inbound_layers[1]] = (-2 / self.m) * self.diff



def topological_sort(feed_dict):
	"""
	Sort generic nodes in topological order using Kahn's Algorithm
	"""
	input_layers = [n for n in feed_dict.keys()]

	G = {}
	layers = [n for n in input_layers]
	while len(layers) > 0:
		n = layers.pop(0)
		if n not in G:
			G[n] = {'in': set(), 'out': set()}
		for m in n.outbound_layers:
			if m not in G:
				G[m] = {'in': set(), 'out': set()}
			G[n]['out'].add(m)
			G[m]['in'].add(n)
			layers.append(m)

	L = []
	S = set(input_layers)
	while len(S) > 0:
		n = S.pop()

		if isinstance(n, Input):
			n.value = feed_dict[n]

		L.append(n)
		for m in n.outbound_layers:
			G[n]['out'].remove(m)
			G[m]['in'].remove(n)
			# if no other incoming edges add to S
			if len(G[m]['in']) == 0:
				S.add(m)
	return L

def forward_and_backward(graph):
	for n in graph:
		n.forward()

	for n in graph[::-1]:
		n.backward()
