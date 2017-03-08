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

import build_model_part2
import part2

IMAGE_SIZE = 28
SEED = 66478  # Set to None for random seed.
nsample = 100
nsamples = 11
npixels = [1, 10, 28, 300]

######################################## Utils functions ########################################
def get_loss(logits,targets,targets_GT=False):
    # Reshapping logits
    log = tf.stack(logits,axis=0) # log shape: [300,nsamples*nbsample,1]
    log = tf.reshape(tf.transpose(log,perm=[1,0,2]),[-1,300]) # log shape: [nsamples*nbsample,300]
    # Reshapping targets
    if targets_GT:
        tar = tf.tile(targets,[1,nsamples]) # tar shape: [nbsample,300*nsamples]
        tar = tf.reshape(tar,[-1,300]) # tar shape: [nsamples*nbsample,300]
    else:
        tar = tf.reshape(targets[:,-300:],[-1,300]) # tar shape: [nsamples*nbsample,300]

    # Xentropy for each images (nsamples*nbsample images)
    sig_Xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar,logits=log),1)
    # mean loss
    loss = tf.reduce_mean(sig_Xentropy)
    # Xentropy for each images as average over nsamples
    Xentropy = tf.reshape(sig_Xentropy,[nsample,nsamples,-1])
    samples_Xentropy = [tf.reduce_mean(Xentropy,1),]
    npix = npixels[:-1]
    npix.reverse()
    for npixel in npix:
        # Xentropy for each images (nsamples*nbsample images)
        sigXentr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar[:,:-300+npixel],logits=log[:,:-300+npixel]),1)
        # Xentropy for each images as average over nsamples
        Xentr = tf.reshape(sigXentr,[nsample,nsamples,-1])
        sampleXentr = tf.reduce_mean(Xentr,1)
        samples_Xentropy.append(sampleXentr)
    samples_Xentropy = tf.stack(samples_Xentropy,axis=1)
    return loss, samples_Xentropy

def process_images(data, predictions, npixels):
    data_shape = np.shape(data)
    # Original
    original_data = data #shape: [nbtodraw, 28x28]
    # tiling original as in build_model
    tilded_original_data = np.tile(original_data,(1,nsamples)) #shape: [nbtodraw, 28x28*nsamples]
    tilded_original_data = np.reshape(tilded_original_data,[-1,IMAGE_SIZE*IMAGE_SIZE]) #shape: [nbtodraw*nsamples,28x28]
    original_data_tostack = np.reshape(original_data,[-1,IMAGE_SIZE*IMAGE_SIZE,1]) #shape: [nbtodraw,28x28,1]
    # Cache data with ones
    cache_data = 0.5*np.ones_like(original_data) #shape: [nbtodraw, 28x28]
    cache_data[:,:-300] = original_data[:,:-300] #shape: [nbtodraw, 28x28]
    tilded_cache_data = np.tile(cache_data,(1,nsamples)) #shape: [nbtodraw, 28x28*nsamples]
    tilded_cache_data = np.reshape(tilded_cache_data,[-1,IMAGE_SIZE*IMAGE_SIZE]) #shape: [nbtodraw*nsamples,28x28]
    cache_data_tostack = np.reshape(cache_data,[-1,IMAGE_SIZE*IMAGE_SIZE,1]) #shape: [nbtodraw,28x28,1]
    # preprocess predictions
    preds = tf.reshape(predictions,[-1,IMAGE_SIZE*IMAGE_SIZE]) #shape: [nsamplesxnbsample,28x28]
    preds= preds.eval() #convert tensor to ndarray
    preds = np.split(preds, data_shape[0], 0) #shape: nbsamples*[nsample, 28x28]
    preds = np.stack(preds,axis=0) #shape: [nsample, nbsamples, 28x28]
    """
    preds_todraw = np.take(preds,idx,axis=0) #shape: [nbtodraw, nbsamples, 28x28]
    """
    preds_todraw = np.transpose(preds,[0,2,1]) #shape: [nbtodraw, 28x28, nsamples]
    if npixels!=300:
        shpe = np.shape(preds_todraw)
        cache = 0.5*np.ones([shpe[0],300-npixels,shpe[2]])
        preds_todraw[:,shpe[1]-300+npixels:,:] = cache
    # stacking all the images, first filter is the original, second is the cache, remaining are the predictions
    images = np.concatenate((original_data_tostack,cache_data_tostack,preds_todraw), axis=2) #shape: [nbtodraw, 28x28, 2+nbsamples]
    images = np.reshape(images,[data_shape[0],IMAGE_SIZE,IMAGE_SIZE,-1])*255.0 #shape: [nbtodraw, 28, 28, 2+nbsamples]
    return images.astype("int32")

def save_images(images,npixel,name_model):
    ROOT_DIR = "../../trained_models/part2/model_" + name_model
    DIR_NAME = os.path.join(ROOT_DIR,str(npixel))
    if not tf.gfile.Exists(DIR_NAME):
        tf.gfile.MkDir(DIR_NAME)
    data_shape = np.shape(images)
    for i in range(data_shape[0]):
        FILE_NAME = "example_" + str(i) + ".png"
        #fig = plt.figure(figsize=(20,40))
        fig = plt.figure()
        for j in range(8):
            plt.subplot(2,4,j+1)
            if j==0:
                plt.title("Original", fontsize=10)
            elif j==1:
                plt.title("cache", fontsize=10)
            elif j==2:
                plt.title("most probable", fontsize=10)
            else:
                plt.title("Sample " + str(j), fontsize=10)
            plt.imshow(images[i,:,:,j], cmap="gray", interpolation=None)
            plt.axis("on")
        DST = os.path.join(DIR_NAME, FILE_NAME)
        fig.savefig(DST)
        plt.close()

######################################## Main ########################################
def in_painting(model_archi,gt_data,cache_data):
    nn_model = model_archi["name"]
    # get weights path
    DST = "../../trained_models/part2/"
    NAME = "model_" + nn_model
    SUB_DST = os.path.join(os.path.join(DST,NAME),NAME)
    print("")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()
    print("Preparing variables and building model {}...".format(nn_model))
    ###### Create tf placeholder ######
    test_data_node = tf.placeholder(dtype = part2.data_type(), shape=(np.shape(gt_data)[0], np.shape(gt_data)[1]))
    cache_data_node = tf.placeholder(dtype = part2.data_type(), shape=(np.shape(cache_data)[0], np.shape(cache_data)[1]))
    ###### Build model and loss ######
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    test_logits, test_pred = build_model_part2.model_inpainting(cache_data_node,
                                                            name=nn_model,
                                                            cell=model_archi["cell"],
                                                            nlayers=model_archi["layers"],
                                                            nunits=model_archi["units"],
                                                            nsamples=nsamples,
                                                            training=phase_train)
    grtr_mean_Xentropy, grtr_samples_Xentropy = get_loss(logits=test_logits,targets=test_data_node[:,-300:],targets_GT=True)
    pred_mean_Xentropy, pred_samples_Xentropy = get_loss(logits=test_logits,targets=test_pred)
    saver = tf.train.Saver()
    print("Model {} built, took {:.4f}s".format(nn_model,time.time()-start_time))

    ###### Create a local session to run the training ######
    with tf.Session() as sess:
        print("\nStart testing...")
        #tf.global_variables_initializer().run()
        csv_path = DST + "/Perf/"
        csvfileTest = open(csv_path + "Xentropy_" + str(nn_model) + ".csv", 'w')
        Testwriter = csv.writer(csvfileTest, delimiter=';',)
        Testwriter.writerow(['Predict CE 300','Grount truth CE 300'])

        # Testing
        if not tf.gfile.Exists(SUB_DST + ".ckpt.meta"):
            raise Exception("no weights given")
        saver.restore(sess, SUB_DST + ".ckpt")
        # Compute and print results once training is done
        start_time = time.time()
        to_compute = [grtr_mean_Xentropy, grtr_samples_Xentropy,
                    pred_mean_Xentropy, pred_samples_Xentropy, test_pred]
        results = sess.run(to_compute, feed_dict={test_data_node: gt_data,
                                                cache_data_node: cache_data,
                                                phase_train: False})
        print("Testing done, took: {:.4f}s".format(time.time()-start_time))
        print("predicted Xent 300: {:.4f}, ground-truth Xent 300: {:.4f}".format(results[2],results[0]))
        # Save Xentropy
        Testwriter.writerow([results[0],results[2]])
        FILE_NAME = os.path.join(DST,"Perf/GT_Xentropy_" + str(nn_model) + ".mat")
        scipy.io.savemat(FILE_NAME, {'mat':results[1]})
        FILE_NAME = os.path.join(DST,"Perf/PR_Xentropy_" + str(nn_model) + ".mat")
        scipy.io.savemat(FILE_NAME, {'mat':results[3]})

        # in painting images and save
        print("\nStart inpainting...")
        for npixel in npixels:
            inpainting_images = process_images(gt_data, results[-1], npixel)
            NB_PIX = str(npixel) + "pixels"
            save_images(inpainting_images,NB_PIX,nn_model)
