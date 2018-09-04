import tensorflow as tf
import numpy as np

# Placeholders for a loaded image (300x300 pixels in a 90000-dimensional vector) and its label (the image could show one out of 32 possible leaves)
x = tf.placeholder(tf.float32, [None, 117])
y_ = tf.placeholder(tf.float32, [None, 1])

# Placeholder for the probability that a neuron's output is kept during dropout
keep_prob = tf.placeholder(tf.float32)


# Reshape the input image for use within a convolutional neural net
# (Last dimension is for "features"; there is only one here, since images are grayscale)
x_image = tf.reshape(x, [-1, 100, 100, 1])

# First convolutional layer - maps one grayscale image to 20 feature maps
W_conv1 = weight_variable([5, 5, 1, 20])
b_conv1 = bias_variable([20])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# First pooling layer - downsamples by 2X
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer - maps 20 feature maps to 40
W_conv2 = weight_variable([5, 5, 20, 40])
b_conv2 = bias_variable([40])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# Second pooling layer - downsamples by 2X
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer; after 2 round of downsampling, the 100x100 image is down to 25x25x40 feature maps which are mapped to 1024 features
W_fc1 = weight_variable([25 * 25 * 40, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 25 * 40])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Map the 1024 features to 10 classes, one for each digit
W_fc2 = weight_variable([1024, 32])
b_fc2 = bias_variable([32])

# Apply dropout to the results of the fully connected layer
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define the loss function
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# l2-regularization: Get all trainable variables in the current network and add the sum of their losses to the loss calculated above
vars = tf.trainable_variables()
loss = loss + 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in vars])

# Define the optimizer with the desired learning rate
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Calculate where the correct label has been predicted; results in a list of booleans
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Calculate the fraction of correctly predicted labels
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    best_validation_accuracy = 0
    below_best_accuracy_count = 0
    saver = tf.train.Saver()

    num_steps = 20000
    for i in range(num_steps):
        # Run a training step on the training data
        sess.run(train_step, feed_dict={
            x: flavia.get('train').get('images'), y_: flavia.get('train').get('labels'), keep_prob: 0.5})

        # Log current accuracy on the training data every 1000th iteration
        if i % 5 == 0:
            train_accuracy = sess.run(
                accuracy, feed_dict={x: flavia.get('train').get('images'), y_: flavia.get('train').get('labels'), keep_prob: 1.0})
            print('training accuracy after %d steps: %g' % (i, train_accuracy))
            test_step_accuracy = sess.run(accuracy,
                feed_dict={x: flavia.get('test').get('images'), y_: flavia.get('test').get('labels'), keep_prob: 1.0})
            print('test accuracy after %d steps: %g' % (i, test_step_accuracy))

    # Calculate and print accuracy on the test images after training has finished
    test_accuracy = sess.run(accuracy, feed_dict={
        x: flavia.get('test').get('images'), y_: flavia.get('test').get('labels'), keep_prob: 1.0})
    print('test accuracy after training has finished: %g' % test_accuracy)