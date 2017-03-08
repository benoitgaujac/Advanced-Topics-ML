import os
import sys
import time
import pdb

import numpy as np
from six.moves import urllib
import tensorflow as tf
import part3


IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 1
SEED = 66478  # Set to None for random seed.
keep_prob = .75

######################################## Utils functions ########################################
def weight_variable(shape,name,layer):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=SEED, dtype=part3.data_type())
    return tf.get_variable("weights_" + layer, shape, initializer=initializer)

def bias_variable(shape,name,layer):
    initializer = tf.constant_initializer(0.1,dtype=part3.data_type())
    return tf.get_variable("biais_" + layer, shape, initializer=initializer)

def base_cell(cell_type="LSTM", nlayers=1, nunits=32, training=False):
    # Creating base Cell
    if cell_type=="LSTM":
        simple_cell = tf.nn.rnn_cell.BasicLSTMCell(nunits)
    elif cell_type=="GRU":
        simple_cell = tf.nn.rnn_cell.GRUCell(nunits)
    else:
        raise Exception("Unknown cell type")
    # dropout
    dropout_training = tf.constant(keep_prob)
    dropout_testing = tf.constant(1.0)
    dropout_prob = tf.select(training,dropout_training,dropout_testing)
    simple_cell = tf.nn.rnn_cell.DropoutWrapper(simple_cell,
                                            input_keep_prob=1,
                                            output_keep_prob=dropout_prob)
    # Stack Cells if needed
    if nlayers>1:
        cells = tf.nn.rnn_cell.MultiRNNCell([simple_cell] * nlayers)
    else:
        cells = simple_cell
    return cells

######################################## Model ########################################
def model(x, name, cell="LSTM", nlayers=1, nunits=32, training=False):
    images_embedded = tf.reshape(x, [-1,IMAGE_SIZE*IMAGE_SIZE,1]) # shape [batch_size,IMAGE_SIZE * IMAGE_SIZE,1]
    # Creating base Cell
    cells = base_cell(cell, nlayers, nunits, training)
    # Build RNN network
    with tf.variable_scope("RNN"):
        # Weights for classification layer
        weight_class = weight_variable([nunits,NUM_LABELS],name,"class")
        biais_class = bias_variable([NUM_LABELS],name,"class")
        outputs, state = tf.nn.dynamic_rnn(cells, images_embedded, dtype=part3.data_type()) # outputs shape: [batch,IMAGE_SIZE * IMAGE_SIZE,nunits]
    out = tf.reshape(outputs, [-1,nunits]) #out shape [batch*(IMAGE_SIZE*IMAGE_SIZE),nunits]
    y = tf.matmul(out,weight_class) + biais_class # y shape: [batch*IMAGE_SIZE*IMAGE_SIZE,1]
    y_reshape = tf.reshape(y,[-1,IMAGE_SIZE*IMAGE_SIZE]) # y shape: [batch,IMAGE_SIZE*IMAGE_SIZE]
    return y_reshape[:,:-1]
