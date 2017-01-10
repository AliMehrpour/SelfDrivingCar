import numpy as np

class Layer:
	def __init__(self, inbound_layers=[]):
		# Layer from which this layer receives value
		self.inbound_layers = inbound_layers
		# Layer to which this layer passes value
		self.outbound_layers = []
		# Calculated value
		self.value = None
		# For each inbound layer here, add this layer as an outbound layer
		for layer in self.inbound_layers:
			layer.outbound_layers.append(self)

		def forward(self):
			"""
			Forward propagartion
			"""
			raise NotImplemented

		def backward(self):
			"""
			Backward propagation
			"""
			raise NotImplemented

class Input(Layer):
	def __init__(self):
		# An Input neuron has no inbound neurons, so no need to pass anything to Neuron constructor
	    Layer.__init__(self)

	# Input neuron in the only node where value may be passed as an argument to forward()
	def forward(self, value=None):
		if value:
			self.value = value

class Linear(Layer):
	def __init__(self, inputs, weights, bias):
		Layer.__init__(self, [inputs, weights, bias])
		self.weights = weights
		self.bias = bias

	def forward(self):
		# Z = XW + B
		self.value = np.dot(self.inbound_layers[0].value, self.weights.value) + self.bias.value
		

class Sigmoid(Layer):
	def __init__(self, layer):
		Layer.__init__(self, [layer])

	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def forward(self):
		self.value = self._sigmoid(self.inbound_layers[0].value)

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

def forward_pass(output_layer, sorted_layers):
	for n in sorted_layers:
		n.forward()

	return output_layer.value
