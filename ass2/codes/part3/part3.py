import gzip
import os
import sys
import time
import pdb

import numpy as np
import csv
from sklearn.utils import shuffle
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io
import tensorflow as tf
from itertools import product

import build_model_part3

from optparse import OptionParser
parser = OptionParser()
parser.add_option('-d', '--data', action='store', dest='data',
    help="dataset to use in {1x1,2x2}")

IMAGE_SIZE = 28
NUM_CHANNELS = 1
SEED = 66478
gru1l128u = {"name": "gru1l128u", "cell": "GRU", "layers": 1, "units":128, "init_learning_rate": 0.005}

######################################## Utils functions  ########################################
def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    return tf.float32

def get_data(dataset):
    # download the data if needed
    DIR_NAME = "../../data"
    if dataset=="1x1":
        NAME = "one_pixel_inpainting.npy"
    elif dataset=="2x2":
        NAME = "2X2_pixels_inpainting.npy"
    else:
        raise Exception("Invalid data")
    DST = os.path.join(DIR_NAME,NAME)
    data = np.load(DST)
    return data[1], data[0]

def get_proposal_from_missing(missing,dataset):
    shpe = np.shape(missing)
    # Create list of in-painting pixels proposale
    if dataset=="1x1":
        nproposals = 1
        lst = []
    elif dataset=="2x2":
        nproposals = 4
    else:
        raise Exception("Invalid data")
    combi = list(product(range(2), repeat=nproposals))
    combi = np.array(combi)
    # Get idx of missing pixels
    proposals = np.zeros([np.shape(combi)[0],shpe[0],shpe[1]]) #shape: [2^nproposals,nimages,IMAGE_SIZExIMAGE_SIZE]
    for i in range(np.shape(combi)[0]):
        proposals[i] = missing
        proposals[i][np.where(missing==-1)] = np.ndarray.flatten(np.stack([combi[i] for _ in range(shpe[0])]))
    return np.reshape(np.transpose(proposals,(1,0,2)),[-1,IMAGE_SIZE*IMAGE_SIZE]), np.shape(combi)[0]#shape: [nimages*2^nproposals,IMAGE_SIZExIMAGE_SIZE]

def get_Xentropy(logits,targets,nproposals,targets_GT=True):
    # Reshapping logits
    log_shape = logits.get_shape().as_list()
    # Reshapping targets
    if targets_GT:
        tar = tf.tile(targets,[1,nproposals]) # tar shape: [1000,IMAGE_SIZE*IMAGE_SIZE*nproposals]
        tar = tf.reshape(tar,[-1,IMAGE_SIZE*IMAGE_SIZE-1]) # tar shape: [1000*nproposals,IMAGE_SIZE*IMAGE_SIZE-1]
    else:
        tar = tf.reshape(targets,[-1,IMAGE_SIZE*IMAGE_SIZE-1]) # tar shape: [1000*nproposals,IMAGE_SIZE*IMAGE_SIZE-1]
    # Xentropy for each images (nsamples*nbsample images)
    sig_Xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar,logits=logits),axis=1) # shape: [1000*nproposals]
    sig_Xentropy = tf.reshape(sig_Xentropy,shape=[-1,nproposals]) # shape: [1000,nproposals]
    idx = tf.argmin(sig_Xentropy, axis=1) #shape: [1000]
    Xentropy = tf.reduce_min(sig_Xentropy, axis=1) #shape: [1000]
    return Xentropy, idx

def process_images(originals, missings, proposals, nproposals, idx_most_probable, idx_tosave):
    data_shape = np.shape(originals)
    # Original
    original_data = originals[idx_tosave] #shape: [nsave, 28x28]
    original_data_tostack = np.reshape(original_data,[-1,IMAGE_SIZE*IMAGE_SIZE,1]) #shape: [nsave, 28x28,1]
    # Missing
    missing_data = missings[idx_tosave]
    missing_data_tostack = np.reshape(missing_data,[-1,IMAGE_SIZE*IMAGE_SIZE,1]) #shape: [nsave, 28x28,1]
    # Most probable
    cache = np.arange(data_shape[0]) #shape: [1000,1]
    idx_toselect = nproposals*cache + idx_most_probable #shape: [1000,1]
    mostprobable_data = proposals[idx_toselect] #shape: [1000,28*28]
    mostprobable_data =  mostprobable_data[idx_tosave] #shape: [1000,28*28]
    mostprobable_data_tostack = np.reshape(mostprobable_data,[-1,IMAGE_SIZE*IMAGE_SIZE,1])
    images = np.concatenate((original_data_tostack,missing_data_tostack,mostprobable_data_tostack), axis=2) #shape: [nbtodraw, 28x28, 3]
    images = np.reshape(images,[-1,IMAGE_SIZE,IMAGE_SIZE,3])*255.0 #shape: [nbtodraw, 28, 28, 3]
    return images.astype("int32")

def save_images(images,nproposals):
    data_shape = np.shape(images) #shape: [nbtodraw, 28, 28, 3]
    for i in range(data_shape[0]):
        if nproposals>2:
            DIR_NAME = "../../trained_models/part3/model_gru1l128u/2x2"
        else:
            DIR_NAME = "../../trained_models/part3/model_gru1l128u/1x1"
        if not tf.gfile.Exists(DIR_NAME):
            tf.gfile.MkDir(DIR_NAME)
        FILE_NAME = "example_" + str(i) + ".png"
        fig = plt.figure()
        for j in range(data_shape[3]):
            plt.subplot(1,3,j+1)
            if j==0:
                plt.title("Original", fontsize=10)
            elif j==1:
                plt.title("missing", fontsize=10)
            elif j==2:
                plt.title("most probable", fontsize=10)
            plt.imshow(images[i,:,:,j], cmap="gray", interpolation=None)
            plt.axis("on")
        DST = os.path.join(DIR_NAME, FILE_NAME)
        fig.savefig(DST)
        plt.close()

######################################## Main ########################################
def main(originals, missings, proposals, nproposals):
    nn_model = gru1l128u["name"]
    DST = "../../trained_models/part3/"
    NAME = "model_gru1l128u"
    SUB_DST = os.path.join(os.path.join(DST,NAME),NAME)

    start_time = time.time()
    print("\nPreparing variables and building model {}...".format(nn_model))
    ###### Create tf placeholder ######
    original_data_node = tf.placeholder(dtype=data_type(), shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS))
    proposals_data_node = tf.placeholder(dtype=data_type(), shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS))

    ###### Build model and loss ######
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    logits = build_model_part3.model(proposals_data_node, name=nn_model,
                                                    cell=gru1l128u["cell"],
                                                    nlayers=gru1l128u["layers"],
                                                    nunits=gru1l128u["units"],
                                                    training=phase_train)
    grtr_Xentropy,   _ = get_Xentropy(logits=logits,targets=original_data_node[:,1:],
                                                nproposals=nproposals, targets_GT=True)
    pred_Xentropy, idx = get_Xentropy(logits=logits,targets=proposals_data_node[:,1:],
                                                nproposals=nproposals, targets_GT=False)
    saver = tf.train.Saver()
    print("Model {} built, took {:.4f}s".format(nn_model,time.time()-start_time))

    ###### Create a local session to run the training ######
    with tf.Session() as sess:
        print("\nStart testing...")
        # Testing
        if not tf.gfile.Exists(SUB_DST+".ckpt.meta"):
            raise Exception("no weights given")
        saver.restore(sess, SUB_DST+".ckpt")
        start_time = time.time()
        to_compute = [grtr_Xentropy, pred_Xentropy, idx]
        results = sess.run(to_compute, feed_dict={original_data_node: originals,
                                                    proposals_data_node: proposals,
                                                    phase_train: False})
        print("Testing done, took: {:.4f}s".format(time.time()-start_time))
        # Save Xentropy
        FILE_NAME = os.path.join(DST,"Perf/GT_" + str(int(np.log(nproposals)/np.log(2))) + ".mat")
        scipy.io.savemat(FILE_NAME, {'mat':results[0]})
        FILE_NAME = os.path.join(DST,"Perf/PR_" + str(int(np.log(nproposals)/np.log(2))) + ".mat")
        scipy.io.savemat(FILE_NAME, {'mat':results[1]})

        # in painting images and save
        print("\nStart inpainting...\n")
        idxtosave = np.random.randint(0,np.shape(originals)[0],100)
        inpainting_images = process_images(originals, missings, proposals, nproposals, results[-1], idxtosave)
        save_images(inpainting_images,nproposals)


if __name__ == '__main__':
    options, arguments = parser.parse_args(sys.argv)
    ###### Load and get data ######
    originals, missings = get_data(options.data)
    proposals, nproposals = get_proposal_from_missing(missings,options.data)
    main(originals,missings,proposals,nproposals)
