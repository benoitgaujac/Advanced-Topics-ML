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
import tensorflow as tf

import build_model
import part2

WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
SEED = 66478  # Set to None for random seed.


######################################## Data processing ########################################
def get_cache_data_set(data,nsample=nsample):
    data_shape = np.shape(data)
    idx = np.random.randint(0,data_shape[0],nsample)
    samples = data[idx]
    return samples[:,:-300], idx

######################################## Utils functions ########################################
def get_loss(logits,targets,targets_GT=False):
    """
    logits: list of logits: 300x[10*nbsample,1]
    targets: targets pixels:
        - if targets_GT True: tensor of GT pixels [nbsample,300]
        - if targets_GT False: list of predicted pixels 300x[10*nbsample,1]
    """
    # Reshapping logits
    log = tf.stack(logits,axe=0) # log shape: [299,10*nbsample,1]
    log = tf.reshape(tf.transpose(log,perm=[1,0,2]),[-1,300]) # log shape: [10*nbsample,299]
    # Reshapping targets
    if not targets_GT:
        tar = tf.stack(targets,axe=0) # tar shape: [299,10*nbsample,1]
        tar = tf.reshape(tf.transpose(tar,perm=[1,0,2]),[-1,300]) # tar shape: [10*nbsample,299]
    elif targets_GT:
        tar = tf.tile(targets,[1,10]) # tar shape: [nbsample,299*10]
        tar = tf.reshape(tar,[-1,300]) # tar shape: [10*nbsample,299]
    loss_300 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar,logits=log))
    loss_28 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar[:,:28],logits=log[:,:28]))
    loss_10 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar[:,:10],logits=log[:,:10]))
    loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar[:,1],logits=log[:,1]))

    return [loss_300, loss_28, loss_10, loss_1]

def inpaint_images(data, idx, predictions, npixels):
    data_shape = orginal_data.get_shape()
    # Original
    original_data = data[idx] #shape: [nbsamples, 28x28]
    original_data_tostack = np.reshape(original_data,[-1,data_shape[1],1])
    # Cache data with ones
    cache_data = np.ones_like(orginal_data) #shape: [nbsamples, 28x28]
    cache_data[:,:-300] = original_data[:,-300] #shape: [nbsamples, 28x28]
    cache_data_tostack = np.reshape(cache_data,[-1,data_shape[1],1])
    # prediction (10 samples)
    preds = tf.stack(predictions,axe=0) #shape: [300, 10xnbsamples]
    preds = tf.reshape(preds,[300,data_shape[0],-1]) #shape: [300, nbsamples, 10]
    preds = tf.transpose(preds,perm=[1,0,2]) #shape: [nbsamples, 300, 10]
    preds_shape = preds.get_shape()
    mask = np.ones([preds_shape[0],preds_shape[1]-npixels,preds_shape[1]]) #shape: [nbsamples, 300-npixels, 10]
    preds[:,npixels:,:] = mask #shape: [nbsamples, 300, 10]
    list_ = [original_data for _ in range(10)] #shape: 10 x [nbsamples, 28x28]
    im_pred = np.stack(original_data, axis=0) #shape: [10, nbsamples, 28x28]
    im_pred = np.transpose(im_pred,(1,2,0)) #shape: [nbsamples, 28x28, 10]
    im_pred[:,-300:,:] = preds_

    images = np.stack([original_data_tostack,cache_data_tostack,im_pred ],axe=2) #shape: [nbsamples, 28x28, 12]
    images = np.reshape(images,[-1,IMAGE_SIZE,IMAGE_SIZE,12]) #shape: [nbsamples, 28, 28, 12]

    return images

def save_images(images,NB_PIX):
    data_shape = images.get_shape()
    for i in range(data_shape[0]):
        for j in range(data_shape[3]):
            fig=plt.figure()
            if j==0:
                plt.title("Original", fontsize=20)
                DIR = "./inpainting/example_" + str(i) + "/original.png"
            elif j==1:
                plt.title("cache", fontsize=20)
                DIR = "./inpainting/example_" + str(i) + "/cache.png"
            else:
                plt.title("inpainting " + str(NB_PIX), fontsize=20)
                DIR = "./inpainting/example_" + str(i) + "/sample_" + str(j) + "/" + str(NB_PIX) + ".png"
            plt.imshow(images[i,:,:,j], cmap=None, interpolation=None)
            plt.axis("off")
            fig.savefig(DIR)
            plt.close()
                

######################################## Main ########################################
def in_painting(model_archi,data,nsample=100):
    nn_model = model_archi["name"]
    cache_data, idx = get_cache_data_set(data,nsample=nsample)

    print("\nPreparing variables and building model {}...".format(nn_model))
    ###### Create tf placeholder ######
    test_data_node = tf.placeholder(dtype = data_type(), shape=(nsample, np.shape(cache_data)[1]))

    ###### Build model and loss ######
    test_logits, test_pred = build_model.model_inpainting(test_data_node,
                                                            name=nn_model,
                                                            cell=model_archi["cell"],
                                                            nlayers=model_archi["layers"],
                                                            nunits=model_archi["units"],
                                                            training=False)
    grtr_cross_entropy = get_loss(logits=test_logits,targets=test_data_node[:,1:],targets_GT=True)
    pred_cross_entropy = get_loss(logits=test_logits,targets=test_pred)

    ###### Create a local session to run the training ######
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        csvfileTest = open('Perf/test_' + str(nn_model) + '.csv', 'w')
        Testwriter = csv.writer(csvfileTest, delimiter=';',)
        Testwriter.writerow(['Predict CE 300','Grount truth CE 300',
                            'Predict CE 28','Grount truth CE 28',
                            'Predict CE 10','Grount truth CE 10'
                            'Predict CE 1','Grount truth CE 1'])
        # Testing
        WORK_DIRECTORY = "./models/model_" + str(nn_model) + ".ckpt"
        if not tf.gfile.Exists(WORK_DIRECTORY):
            raise Exception("no weights given")
        saver.restore(sess, WORK_DIRECTORY)

        # Compute and print results once training is done
        prediction_XE, GroundTruth_XE = sess.run([grtr_cross_entropy, pred_cross_entropy],
                                                    feed_dict={test_data_node: test_data,})
        test_acc = accuracy_logistic(test_pred,test_data[:,1:])
        print("\nTesting after {} epochs.".format(num_epochs))
        print("Predict Xent 300: {:.4f}, Ground truth Xent 300: {:.4f}".format(prediction_XE[0],GroundTruth_XE[0]))
        Testwriter.writerow([prediction_XE[0],GroundTruth_XE[0],
                            prediction_XE[1],GroundTruth_XE[1],
                            prediction_XE[2],GroundTruth_XE[2],
                            prediction_XE[3],GroundTruth_XE[3]])

        # in painting images and save
        idxtosave = np.random.randint(0,nsample,5)
        npixels = [1, 10, 28, 300]
        for npixel in npixels:
            inpainting_images = inpaint_images(data, idx[idxtosave], cache_data, test_pred, npixel)
            NB_PIX = str(npixel) + "pixels"
            save_images(inpainting_images,NB_PIX)
