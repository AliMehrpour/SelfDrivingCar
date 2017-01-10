import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

learning_rate = 0.01
n_input = 784
n_classes = 10

# Import MINST data
minst = input_data.read_data_sets('.', one_hot=True)

# Features and Lables
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights and biases
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logit
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


save_file = 'train_model.ckpt'
batch_size = 128
n_epochs = 10000


saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for epoch in range(n_epochs):
		batch_features, batch_labels = minst.train.next_batch(batch_size)
		sess.run(optimizer, feed_dict={features: batch_features, labels:batch_labels})

		if epoch % 10 == 0:
			valid_accuracy = sess.run(accuracy, feed_dict={features: minst.validation.images, labels: minst.validation.labels})
			print('Epoch {:<3} - Validation Accuracy: {}'.format(epoch, valid_accuracy))

	saver.save(sess, save_file)
	print('Trained Model Saved.')



