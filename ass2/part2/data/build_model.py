import os
import sys
import time
import pdb

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from part1 import data_type

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.
fully_connected_neurons = 256
keep_prob = .7

######################################## Model ########################################
def weight_variable(shape,name,layer):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    #return tf.Variable(initial,dtype=data_type())
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=SEED, dtype=data_type())
    with tf.variable_scope(name):
        return tf.get_variable("weights_" + layer, shape, initializer=initializer)


def bias_variable(shape,name,layer):
    #initial = tf.constant(0.1, shape=shape)
    #return tf.Variable(initial,dtype=data_type())
    initializer = tf.constant_initializer(0.1)
    with tf.variable_scope(name):
        return tf.get_variable("biais_" + layer, shape, initializer=initializer)


"""
# Dictionary of weights
OneLinear_weights = {
    "W": weight_variable([IMAGE_SIZE*IMAGE_SIZE, NUM_LABELS]),
    "b": bias_variable([NUM_LABELS])
}
fully_connected_neurons1 = 128
OneHidden_weights = {
    "W1": weight_variable([IMAGE_SIZE*IMAGE_SIZE, fully_connected_neurons1]),
    "b1": bias_variable([fully_connected_neurons1]),
    "W2": weight_variable([fully_connected_neurons1, NUM_LABELS]),
    "b2": bias_variable([NUM_LABELS])
}
fully_connected_neurons2 = 256
TwoHidden_weights = {
    "W1": weight_variable([IMAGE_SIZE*IMAGE_SIZE, fully_connected_neurons2]),
    "b1": bias_variable([fully_connected_neurons2]),
    "W2": weight_variable([fully_connected_neurons2, fully_connected_neurons2]),
    "b2": bias_variable([fully_connected_neurons2]),
    "W3": weight_variable([fully_connected_neurons2, NUM_LABELS]),
    "b3": bias_variable([NUM_LABELS])
}
size_filters = 3
num_filters = 32
fully_connected_neurons3 = 256
Conv1_weights = {
    "Wconv1": weight_variable([size_filters,size_filters, NUM_CHANNELS, num_filters]),
    "bconv1": bias_variable([num_filters]),
    "Wconv2": weight_variable([size_filters,size_filters,num_filters,num_filters]),
    "bconv2": bias_variable([num_filters]),
    "Wdense1": weight_variable([int(IMAGE_SIZE/4*IMAGE_SIZE/4*num_filters), fully_connected_neurons3]),
    "bdense1": bias_variable([fully_connected_neurons3]),
    "Wdense2": weight_variable([fully_connected_neurons3, NUM_LABELS]),
    "bdense2": bias_variable([NUM_LABELS])
}
conv_weights = {
    "Wconv1": weight_variable([size_filters,size_filters, NUM_CHANNELS, num_filters]),
    "bconv1": bias_variable([num_filters]),
    "Wconv2": weight_variable([size_filters,size_filters,num_filters,2*num_filters]),
    "bconv2": bias_variable([2*num_filters]),
    "Wdense1": weight_variable([int(IMAGE_SIZE/4*IMAGE_SIZE/4*2*num_filters), fully_connected_neurons3]),
    "bdense1": bias_variable([fully_connected_neurons3]),
    "Wdense2": weight_variable([fully_connected_neurons3, NUM_LABELS]),
    "bdense2": bias_variable([NUM_LABELS])
}
"""

def model(x, name, cell="LSTM", nlayers=1, nunits=32, training=False):
    #with tf.variable_scope('RNN') as scope:
        """
        # Weigth for linear transformation
        weight_embedding = weight_variable([IMAGE_SIZE*IMAGE_SIZE,nunits])
        biais_embedding = bias_variable([nunits])
        weight_embedding_tile = tf.tile(weight_embedding,[1,IMAGE_SIZE*IMAGE_SIZE])
        biais_embedding_tile = tf.tile(biais_embedding,[IMAGE_SIZE*IMAGE_SIZE])
        images_embedded = tf.matmul(x,weight_embedding_tile) #+ biais_embedding_tile
        imshape = images_embedded.get_shape().as_list()
        images_embedded = tf.reshape(images_embedded, [imshape[0],IMAGE_SIZE*IMAGE_SIZE,nunits]) # shape [batch_size,IMAGE_SIZE * IMAGE_SIZE,nunits]
        """
        imshape = x.get_shape().as_list()
        images_embedded = tf.reshape(x, [imshape[0],imshape[1],1]) # shape [batch_size,IMAGE_SIZE * IMAGE_SIZE,nunits]

        #pdb.set_trace()

        # Creating base Cell
        if cell=="LSTM":
            simple_cell = tf.nn.rnn_cell.BasicLSTMCell(nunits)
        elif cell=="GRU":
            simple_cell = tf.nn.rnn_cell.GRUCell(nunits)
        else:
            raise Exception("Unknown cell type")
        # dropout
        if training:
            simple_cell = tf.nn.rnn_cell.DropoutWrapper(simple_cell,
                        input_keep_prob=1, output_keep_prob=keep_prob)
        # Stack Cells if needed
        if nlayers>1:
            cells = tf.nn.rnn_cell.MultiRNNCell([simple_cell] * nlayers)
        else:
            cells = simple_cell
        # Build RNN network
        seq_lent = IMAGE_SIZE*IMAGE_SIZE * np.ones([imshape[0]])
        #pdb.set_trace()
        outputs, state = tf.nn.dynamic_rnn(cells, images_embedded, dtype=data_type(), sequence_length=seq_lent) # outputs shape: [batch,IMAGE_SIZE * IMAGE_SIZE,nunits]
        outshape = outputs.get_shape().as_list()
        outputs_transpose = l = tf.transpose(outputs, perm=[0, 2, 1])
        #outputs = tf.reshape(outputs, [outshape[0],outshape[1]*outshape[2]]) # outputs shape: [batch,IMAGE_SIZE * IMAGE_SIZE * nunits]
        out =  outputs_transpose[:,:,-1]# outputs shape: [batch, nunits]

        # Weights for affine transformation
        #weight_linear = weight_variable([outshape[1]*outshape[2],fully_connected_neurons],name,"linear")
        weight_linear = weight_variable([nunits,fully_connected_neurons],name,"linear")
        biais_linear = bias_variable([fully_connected_neurons],name,"linear")
        # Fully connected layer with ReLU non linearity
        #z = tf.nn.relu(tf.matmul(outputs, weight_linear) + biais_linear)
        z = tf.nn.relu(tf.matmul(out, weight_linear) + biais_linear)
        # Weights for classification layer
        weight_class = weight_variable([fully_connected_neurons,NUM_LABELS],name,"class")
        biais_class = bias_variable([NUM_LABELS],name,"class")

        return tf.matmul(z,weight_class) + biais_class

"""
def Onelinear(x,weights):
    # reshape image
    x_ = tf.reshape(x,[-1,IMAGE_SIZE*IMAGE_SIZE])
    return tf.matmul(x_, weights["W"]) + weights["b"]

def OneHidden(x,weights):
    fully_connected_neurons = 128
    # reshape image
    x_ = tf.reshape(x,[-1,IMAGE_SIZE*IMAGE_SIZE])
    # Relu layer
    shape1 = [IMAGE_SIZE*IMAGE_SIZE, fully_connected_neurons]
    z1 = tf.nn.relu(tf.matmul(x_, weights["W1"]) + weights["b1"])
    # Linear layer
    shape2 = [fully_connected_neurons, NUM_LABELS]
    return tf.matmul(z1, weights["W2"]) + weights["b2"]

def TwoHidden(x,weights):
    fully_connected_neurons = 256
    # reshape image
    x_ = tf.reshape(x,[-1,IMAGE_SIZE*IMAGE_SIZE])
    # first Relu layer
    z1 = tf.nn.relu(tf.matmul(x_, weights["W1"]) + weights["b1"])
    # second Relu layer
    z2 = tf.nn.relu(tf.matmul(z1, weights["W2"]) + weights["b2"])
    # Linear layer
    return tf.matmul(z2, weights["W3"]) + weights["b3"]

def Conv1(x,weights):
    fully_connected_neurons = 256
    # first conv layer
    zconv1 = tf.nn.conv2d(x, weights["Wconv1"], strides=[1, 1, 1, 1], padding='SAME')
    zconv1 = tf.nn.max_pool(zconv1 + weights["bconv1"], ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    # second conv layer
    zconv2 = tf.nn.conv2d(zconv1, weights["Wconv2"], strides=[1, 1, 1, 1], padding='SAME')
    zconv2 = tf.nn.max_pool(zconv2 + weights["bconv2"], ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    # Flatten
    zconv2_flat = tf.reshape(zconv2,[-1,int(IMAGE_SIZE/4*IMAGE_SIZE/4*num_filters)])
    # Relu layer
    zdense1 = tf.nn.relu(tf.matmul(zconv2_flat, weights["Wdense1"]) + weights["bdense1"])
    # Linear layer
    return tf.matmul(zdense1, weights["Wdense2"]) + weights["bdense2"]

def conv(x,weights,train=False):
    fully_connected_neurons = 256
    # first conv layer
    zconv1 = tf.nn.conv2d(x, weights["Wconv1"], strides=[1, 1, 1, 1], padding='SAME')
    zconv1 = tf.nn.max_pool(zconv1 + weights["bconv1"], ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    # second conv layer
    zconv2 = tf.nn.conv2d(zconv1, weights["Wconv2"], strides=[1, 1, 1, 1], padding='SAME')
    zconv2 = tf.nn.max_pool(zconv2 + weights["bconv2"], ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    # Flatten
    zconv2_flat = tf.reshape(zconv2,[-1,int(IMAGE_SIZE/4*IMAGE_SIZE/4*2*num_filters)])
    # Relu layer
    zdense1 = tf.nn.relu(tf.matmul(zconv2_flat, weights["Wdense1"]) + weights["bdense1"])
    # Add a 50% dropout during training only
    if train:
        zdense1 = tf.nn.dropout(zdense1, 0.5, seed=SEED)
    # Linear layer
    return tf.matmul(zdense1, weights["Wdense2"]) + weights["bdense2"]
"""
