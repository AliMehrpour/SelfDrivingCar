from miniflow import *

x, y , z = Input(), Input(), Input()
inputs = [x, y, z]

weigth_x,  weigth_y, weigth_z = Input(), Input(), Input()
weigths = [weigth_x, weigth_y, weigth_z] 

bias = Input()

f = Linear(inputs, weigths, bias)

feed_dict = {
	x: 6,
	y: 14,
	z: 3,
	weigth_x: 0.5,
	weigth_y: 0.25,
	weigth_z: 1.4,
	bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print("Output: ", output)