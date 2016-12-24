from miniflow import *
import numpy as np

X, W, b = Input(), Input(), Input()
y = Input()
f = Linear(X, W, b)
a = Sigmoid(f)
cost = MSE(y, a)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2.], [3.]])
b_ = np.array([-3.])
y_ = np.array([1, 2])

feed_dict = {
	X: X_,
	y: y_,
	W: W_,
	b: b_,
}

graph = topological_sort(feed_dict)
forward_and_backward(graph)
gradients = [t.gradients[t] for t in [X, y, W, b]]
print(gradients)