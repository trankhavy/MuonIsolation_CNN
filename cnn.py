import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load in the data
x_background = np.load("npy/test_background.npy")
x_signal = np.load("npy/test_signal.npy")
print(x_signal.shape)
train_data = np.vstack((x_background,x_signal))
print("Train data: ", train_data.shape)
train_out = np.array([0]*len(x_background) + [1]*len(x_signal))
numtrain = len(train_out)
random.seed(1)
order = range(numtrain)
random.shuffle(order)

order = np.array(order)

print(type(order))
train_out = train_out[order]
print("train_out shape: ", train_out.shape)
train_data = train_data[order]
print("train_data shape: ", train_data.shape)
ratio = 0.9
x_train =  train_data[:int(numtrain*ratio),:]
x_test = train_data[int(numtrain*ratio):,:]
print("x_train shape: ",x_train.shape)
print("x_test shape: ",x_test.shape)
y_train = train_out[:int(numtrain*ratio)].reshape((int(numtrain*ratio),1))
print("y_train shape: ",y_train.shape)
y_test = train_out[int(numtrain*ratio):].reshape((numtrain-int(numtrain*ratio)),1)
print("y_test shape: ",y_test.shape)

# Define hyperparameters
training_iters = 200
learning_rate = 1e-3


# Network parameters
# Image input shape: (28,28,4)
n_input = 28

# Create placeholder
x = tf.placeholder("float", [None,28,28,4])
y = tf.placeholder("float", [None,1])

def conv2d(x,W,b,strides=1):
    # Conv2D wrapper with bias and relu activation
    # The first stride is 1 because the first is for image-number
    x = tf.nn.conv2d(x,W,strides=[1,strides, strides, 1], padding = "SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

# Max-pooling filter will be a square 2x2 and stride by which the filter move in is also 2. ksize is the size of kernel
def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1],padding="SAME") # ??? Should stride[1,k,k,4]???

# Define weight

weights = {
        # Shape argument: first and second are the filter size, third is the number of channels, fourth is number of filters
        # For wc2, wc3: third is the umber of channels from previous output.
        # wd1 stands for downsampling, 28x28x1 -> 4x4x1 and need to flatten this downsampled output to feed this as input to the fully connected layer. 4*4*128 is the output by the convolution layer 3. The second element indicate the number of neuron that we want in the fully connected layer
        "wc1": tf.get_variable("W0",shape=(3,3,4,32), initializer=tf.contrib.layers.xavier_initializer()),
        "wc2": tf.get_variable("W1",shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
        "wc3": tf.get_variable("W2",shape=(3,3,64,128),initializer=tf.contrib.layers.xavier_initializer()),
        # ???? shape(128,1) or shape(128,2) How many neuron do we want?
        "wd1": tf.get_variable("W3", shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()),
        "out": tf.get_variable("W4", shape=(128,1), initializer= tf.contrib.layers.xavier_initializer())
        }

biases = {
        "bc1": tf.get_variable("B0", shape=(32), initializer= tf.contrib.layers.xavier_initializer()),
        "bc2": tf.get_variable("B1", shape=(64), initializer= tf.contrib.layers.xavier_initializer()),
        "bc3": tf.get_variable("B2", shape=(128), initializer= tf.contrib.layers.xavier_initializer()),
        "bd1": tf.get_variable("B3", shape=(128), initializer= tf.contrib.layers.xavier_initializer()),
        "out": tf.get_variable("B4", shape=(1), initializer= tf.contrib.layers.xavier_initializer())
        }

def conv_net(x, weights, biases):
    # Call the conv2d function we defined above and pass in the parameters
    conv1 = conv2d(x, weights["wc1"], biases["bc1"])
    # Maxpooling, this choose the max value from 2x2 matrix and output a 14x14 matrix
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights["wc3"], biases["bc3"])
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    # [-1] in reshape means infer the first dimension on its own
    fc = tf.reshape(conv3, [-1,weights["wd1"].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, weights["wd1"]), biases["out"])
    fc = tf.nn.relu(fc)
    out = tf.add(tf.matmul(fc, weights["out"]),biases["out"])
    return out


pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Check whether the prediction against the true value
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

# Calculate accuracy across all the given images and average them
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variable
init = tf.global_variables_initializer()
num_epoch = 20
num_batch = 95
batch_size = int(len(x_train)/num_batch)

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    #summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(num_epoch):
        for batch in range(num_batch):
            batch_x = x_train[batch*batch_size: batch*batch_size+batch_size]
            batch_y = y_train[batch*batch_size: batch*batch_size+ batch_size]

        # Run optimization
        # Calculate loss and accuracy
            opt = sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
            loss, acc = sess.run([cost,accuracy], feed_dict={x:batch_x, y:batch_y})

        print("Epoch: "+str(i) + "\n"
              "Loss: {}".format(loss) + "\n"
              "Training accuracy: {}".format(acc))
    print("Optimization finished!")

    # Calculate test set accuracy
    train_loss.append(loss)
    train_accuracy.append(acc)

    # Test set accuracy
    test_acc, test_loss = sess.run([accuracy, cost], feed_dict={x:x_test, y:y_test})
    print("Test accuracy: ", test_acc)






















































































































































