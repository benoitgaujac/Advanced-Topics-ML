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
fully_connected_neurons = 100
keep_prob = 0.8

######################################## Model ########################################
def weight_variable(shape,name,layer):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=SEED, dtype=data_type())
    return tf.get_variable("weights_" + layer, shape, initializer=initializer)

def bias_variable(shape,name,layer):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable("biais_" + layer, shape, initializer=initializer)

def model(x, name, cell="LSTM", nlayers=1, nunits=32, training=False):
        imshape = x.get_shape().as_list()
        images_embedded = tf.reshape(x, [imshape[0],imshape[1],1]) # shape [batch_size,IMAGE_SIZE * IMAGE_SIZE,1]

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
        outputs, state = tf.nn.dynamic_rnn(cells, images_embedded, dtype=data_type(), sequence_length=seq_lent) # outputs shape: [batch,IMAGE_SIZE * IMAGE_SIZE,nunits]
        out = outputs[:,-1,:] # We are interested only on the last output of the RNN. out shape : [batch,nunits]
        # Weights for affine transformation
        weight_linear = weight_variable([nunits,fully_connected_neurons],name,"linear")
        biais_linear = bias_variable([fully_connected_neurons],name,"linear")
        # Fully connected layer with ReLU non linearity
        z = tf.nn.relu(tf.matmul(out, weight_linear) + biais_linear)
        # Weights for classification layer
        weight_class = weight_variable([fully_connected_neurons,NUM_LABELS],name,"class")
        biais_class = bias_variable([NUM_LABELS],name,"class")

        return tf.matmul(z,weight_class) + biais_class
