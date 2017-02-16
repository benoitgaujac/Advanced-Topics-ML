import os
import sys
import time
import pdb

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from part2 import data_type

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 1
SEED = 66478  # Set to None for random seed.
#fully_connected_neurons = 256
keep_prob = .7

######################################## Utils functions ########################################
# Sample pixel values from logits
def get_samples(logits,nsamples,last_known=False):
    logits_shape = logits.get_shape().as_list()
    # if output from last known pixel, we want 10 samples
    # otherwise, we just want 1 sample from sample pixel
    if last_known:
        nsample = 10
    else:
        nsample = 1
    # get log proba
    sig = tf.sigmoid(logits)
    ones = tf.ones_like(sig,dtype=data_type())
    sig_ = tf.subtract(ones,sig)
    proba = tf.stack([sig,sig_],axis=1)
    proba = tf.reshape(proba,[-1,2])
    # sampling
    samples = tf.multinomial(proba, nsamples, seed=SEED)
    return tf.to_float(tf.reshape(samples,[-1,1,1])) # reshaping to [10*batch,1,1]

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
    outputs, state = tf.nn.dynamic_rnn(cells, images_embedded, dtype=data_type(), sequence_length=seq_lent) # outputs shape: [batch,IMAGE_SIZE * IMAGE_SIZE,nunits]
    out = tf.reshape(outputs, [-1,nunits]) #out shape [batch*(IMAGE_SIZE*IMAGE_SIZE),nunits]
    # Weights for classification layer
    weight_class = weight_variable([nunits,NUM_LABELS],name,"class")
    biais_class = bias_variable([NUM_LABELS],name,"class")
    y = tf.matmul(out,weight_class) + biais_class # y shape: [batch*IMAGE_SIZE*IMAGE_SIZE,1]
    y_reshape = tf.reshape(y,[-1,IMAGE_SIZE*IMAGE_SIZE]) # y shape: [batch,IMAGE_SIZE*IMAGE_SIZE]
    #pdb.set_trace()
    return y_reshape[:,:-1]

def model_inpainting(x, name, cell="LSTM", nlayers=1, nunits=32, nsamples=10, training=False):
    imshape = x.get_shape().as_list()
    images_embedded = tf.reshape(x, [imshape[0],imshape[1],1]) # shape [batch_size,IMAGE_SIZE * IMAGE_SIZE,1]
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
    seq_lent = imshape[1]*imshape[1] * np.ones([imshape[0]])
    with tf.variable_scope(name) as scope:
        # Weights for classification layer
        weight_class = weight_variable([nunits,NUM_LABELS],name,"class")
        biais_class = bias_variable([NUM_LABELS],name,"class")
        # Get state and ouputs up to the last visible pixel
        outputs, state = tf.nn.dynamic_rnn(cells, images_embedded,  # outputs shape: [batch,IMAGE_SIZE * IMAGE_SIZE,nunits]
                                                dtype=data_type(),
                                                sequence_length=seq_lent)
        last_out = outputs[:,-1,:] # last_out shape: [batch, nunits]
        last_out = tf.matmul(last_out,weight_class) + biais_class # last_out shape: [batch, 1]
        # sample 10 value given last output
        inputs = get_samples(last_out,nsamples,last_known=True) # inputs shape [10*batch,1,1]
        out = tf.tile(last_out,[1,nsamples]) # out shape [batch,10]
        out = tf.reshape(out,[-1,1]) # out shape [10*batch,1]
        # list of pixels predictions and pixels logits
        out_predictions = [tf.reshape(inputs,[-1,1]),]
        out_logits = [out,]
        # run the RNN throught the hidden 300 last pixels
        inputs_shape = inputs.get_shape().as_list()
        seq_lent = np.ones([inputs_shape[0]])
        scope.reuse_variables()
        for i in range(300):
            out, state = tf.nn.dynamic_rnn(cells, inputs,  # out shape [10*batch,1,nunits]
                                        dtype=data_type(),
                                        sequence_length=seq_lent,
                                        initial_state=state)
            out = tf.reshape(out,[-1,nunits]) # out shape [10*batch,nunits]
            out = tf.matmul(out,weight_class) + biais_class # out shape: [10*batch,1]
            out_logits.append(out)
            inputs = get_samples(out,nsamples) # inputs shape [10*batch,1,1]
            out_predictions.append(tf.reshape(inputs,[-1,1])) # prediction shape: [10*batch,1]

    # out_logits is list of lenth 300 of output logits [10*batch,1]
    # out_predictions is list of lenth 300 of predicted pixels [10*batch,1]
    return out_logits[:-1], out_predictions[:-1]
