import os
import sys
import time
import pdb

import numpy as np
from six.moves import urllib
import tensorflow as tf
import part2
#import inpainting

nsample = 100
nsamples = 11

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 1
SEED = 66478  # Set to None for random seed.
#fully_connected_neurons = 256
keep_prob = .5

######################################## Utils functions ########################################
def weight_variable(shape,name,layer):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=SEED, dtype=part2.data_type())
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
    if training:
        simple_cell = tf.nn.rnn_cell.DropoutWrapper(simple_cell,
                    input_keep_prob=1, output_keep_prob=keep_prob)
    # Stack Cells if needed
    if nlayers>1:
        cells = tf.nn.rnn_cell.MultiRNNCell([simple_cell] * nlayers)
    else:
        cells = simple_cell
    return cells

# Sample pixel values from logits
def get_samples(logits):
    logits_shape = logits.get_shape().as_list()
    log = tf.reshape(logits,[-1,nsamples,1])
    # Bernoulli sample
    log_b = log[:,1:,:]
    bernoulli = tf.contrib.distributions.Bernoulli(logits=log_b, dtype=tf.float32)#, logits_shape)
    bernoulli_samples = bernoulli.sample()
    # Most probable sample
    log_mp = log[:,0,:]
    mostprobable_samples = get_mostprobable_sample(log_mp)
    # Concat samples
    samples = tf.concat(1,[mostprobable_samples,bernoulli_samples])
    samples = tf.reshape(samples,[logits_shape[0],logits_shape[1],1])

    return samples # reshaping to [nsamples*batch,1,1]

def get_mostprobable_sample(logits):
    proba = tf.sigmoid(logits)
    samples = tf.round(proba)

    return tf.reshape(samples,[-1,1,1]) # reshaping to [nsamples*batch,1,1]

######################################## Model ########################################
def model(x, name, cell="LSTM", nlayers=1, nunits=32, training=False):
    imshape = x.get_shape().as_list()
    images_embedded = tf.reshape(x, [imshape[0],imshape[1],1]) # shape [batch_size,IMAGE_SIZE * IMAGE_SIZE,1]
    #pdb.set_trace()
    # Creating base Cell
    cells = base_cell(cell, nlayers, nunits, training)
    # Build RNN network
    seq_lent = IMAGE_SIZE*IMAGE_SIZE * np.ones([imshape[0]])
    outputs, state = tf.nn.dynamic_rnn(cells, images_embedded, dtype=part2.data_type(), sequence_length=seq_lent) # outputs shape: [batch,IMAGE_SIZE * IMAGE_SIZE,nunits]
    out = tf.reshape(outputs, [-1,nunits]) #out shape [batch*(IMAGE_SIZE*IMAGE_SIZE),nunits]
    # Weights for classification layer
    weight_class = weight_variable([nunits,NUM_LABELS],name,"class")
    biais_class = bias_variable([NUM_LABELS],name,"class")
    y = tf.matmul(out,weight_class) + biais_class # y shape: [batch*IMAGE_SIZE*IMAGE_SIZE,1]
    y_reshape = tf.reshape(y,[-1,IMAGE_SIZE*IMAGE_SIZE]) # y shape: [batch,IMAGE_SIZE*IMAGE_SIZE]
    #pdb.set_trace()
    return y_reshape[:,:-1]

def model_inpainting(x, name, cell="LSTM", nlayers=1, nunits=32, nsamples=10, training=False):
    xshape = x.get_shape().as_list()
    # We create nsamples identic images from the original one
    images_embedded = tf.tile(x,(1,nsamples)) # shape [batch_size,IMAGE_SIZE * IMAGE_SIZE*nsamples]
    images_embedded = tf.reshape(images_embedded,[xshape[0]*nsamples,-1,1]) # shape [batch_size*nsamples,IMAGE_SIZE * IMAGE_SIZE]
    imshape = images_embedded.get_shape().as_list()
    # Creating base Cell
    cells = base_cell(cell, nlayers, nunits, training)
    # Build RNN network
    seq_lent = imshape[1] * np.ones([imshape[0]])
    with tf.variable_scope(name) as scope:
        # Weights for classification layer
        weight_class = weight_variable([nunits,NUM_LABELS],name,"class")
        biais_class = bias_variable([NUM_LABELS],name,"class")
        # Get state and ouputs up to the last visible pixel
        outputs, state = tf.nn.dynamic_rnn(cells, images_embedded,  # outputs shape: [nsamples*batch,IMAGE_SIZE * IMAGE_SIZE,nunits]
                                                dtype=part2.data_type(),
                                                sequence_length=seq_lent)
        last_out = outputs[:,-1,:] # last_out shape: [nsamples*batch, nunits]
        last_out = tf.matmul(last_out,weight_class) + biais_class # last_out shape: [nsamples*batch, 1]
        # sample nsamples pixel values from last output
        inputs = get_samples(last_out) # inputs shape [nsamples*batch,1,1]
        # list of pixels predictions and pixels logits
        im_pred = tf.concat(1,[images_embedded,inputs])
        out_logits = [last_out,]
        # run the RNN throught the hidden 300 last pixels
        inputs_shape = inputs.get_shape().as_list()
        seq_lent = np.ones([inputs_shape[0]])
        scope.reuse_variables()
        for i in range(300):
            # sample from bernuolli logits
            out, state = tf.nn.dynamic_rnn(cells, inputs,  # out shape [nsamples*batch,1,nunits]
                                        dtype=part2.data_type(),
                                        sequence_length=seq_lent,
                                        initial_state=state)
            out = tf.reshape(out,[-1,nunits]) # out shape [nsamples*batch,nunits]
            out = tf.matmul(out,weight_class) + biais_class # out shape: [nsamples*batch,1]
            out_logits.append(out)
            inputs = get_samples(out) # inputs shape [nsamples*batch,1,1]
            im_pred = tf.concat(1,[im_pred,inputs])

    # out_logits is list of lenth 300 of output logits [nsamples*batch,1]
    # im_pred shape: [nsamples*batch,28x28]
    return out_logits[:-1], im_pred[:,:-1]
