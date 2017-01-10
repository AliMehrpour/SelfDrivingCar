class Neuron:
	def __init__(self, inbound_neurons=[], label=''):
		# An optional description for the neuron
		self.label = label
		# Neuron from which this neuron receives value
		self.inbound_neurons = inbound_neurons
		# Neuron to which this neuron passes value
		self.outbound_neurons = []
		# Calculated value
		self.value = None
		# For each inbound neuron here, add this neuron as an outbound neuron
		for n in self.inbound_neurons:
			n.outbound_neurons.append(self)

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

class Input:
	def __init__(self):
		# An Input neuron has no inbound neurons, so no need to pass anything to Neuron constructor
	    Neuron.__init__(self)

	# Input neuron in the only node where value may be passed as an argument to forward()
	def forward(self, value=None):
		if value:
			self.value = value

class Add:
	def __init__(self, *inputs):
	    Neuron.__init__(self, inputs)

	def forward(self):
		# Set the value of this neuron
		sum = 0
		for n in self.inbound_neurons:
			sum += n.value

		self.value = sum


def topological_sort(feed_dict):
	"""
	Sort generic nodes in topological order using Kahn's Algorithm
	"""
	input_neurons = [n for n in feed_dict.keys()]

	G = {}
	neurons = [n for n in input_neurons]
	while len(neurons) > 0:
		n = neurons.pop(0)
		if n not in G:
			G[n] = {'in': set(), 'out': set()}
		for m in n.outbound_neurons:
			if m not in G:
				G[m] = {'in': set(), 'out': set()}
			G[n]['out'].add(m)
			G[m]['in'].add(n)
			neurons.append(m)

	L = []
	S = set(input_neurons)
	while len(S) > 0:
		n = S.pop()

		if isinstance(n, Input):
			n.value = feed_dict[n]

		L.append(n)
		for m in n.outbound_neurons:
			G[n]['out'].remove(m)
			G[m]['in'].remove(n)
			# if no other incoming edges add to S
			if len(G[m]['in']) == 0:
				S.add(m)
	return L

def forward_pass(output_neuron, sorted_neurons):
	for n in sorted_neurons:
		n.forward()

	return output_neuron.value