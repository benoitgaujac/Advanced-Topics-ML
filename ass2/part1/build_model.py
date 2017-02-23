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
keep_prob = 0.75

######################################## Utils functions ########################################
def weight_variable(shape,name,layer):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=SEED, dtype=data_type())
    return tf.get_variable("weights_" + layer, shape, initializer=initializer)

def bias_variable(shape,name,layer):
    initializer = tf.constant_initializer(0.1)
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

def batch_norm(x, n_out, phase_train):
    beta = tf.get_variable(name="beta",shape=[n_out], dtype = data_type(),
                                initializer=tf.constant_initializer(0.0),)
    gamma = tf.get_variable(name="gamma",shape=[n_out], dtype = data_type(),
                                initializer=tf.constant_initializer(0.0))
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    def mean_var_no_update():
        return ema.average(batch_mean), ema.average(batch_var)

    mean, var = tf.cond(phase_train,
                    mean_var_with_update,
                    mean_var_no_update)

    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

######################################## Model ########################################
def model(x, name, cell="LSTM", nlayers=1, nunits=32, training=False):
        images_embedded = tf.reshape(x, [-1,IMAGE_SIZE * IMAGE_SIZE,1]) # shape [batch_size,IMAGE_SIZE * IMAGE_SIZE,1]
        # Creating base Cell
        cells = base_cell(cell, nlayers, nunits, training)
        # Build RNN network
        outputs, state = tf.nn.dynamic_rnn(cells, images_embedded, dtype=data_type()) # outputs shape: [batch,IMAGE_SIZE * IMAGE_SIZE,nunits]
        out = outputs[:,-1,:] # We are interested only on the last output of the RNN. out shape : [batch,nunits]
        # batch normalization
        out_norm = batch_norm(out, nunits, training)
        # Weights for affine transformation
        weight_linear = weight_variable([nunits,fully_connected_neurons],name,"linear")
        biais_linear = bias_variable([fully_connected_neurons],name,"linear")
        # Fully connected layer with ReLU non linearity
        z = tf.nn.relu(tf.matmul(out_norm, weight_linear) + biais_linear)
        # Weights for classification layer
        weight_class = weight_variable([fully_connected_neurons,NUM_LABELS],name,"class")
        biais_class = bias_variable([NUM_LABELS],name,"class")

        return tf.matmul(z,weight_class) + biais_class
