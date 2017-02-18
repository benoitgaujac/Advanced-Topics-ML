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
nsamples = 10

######################################## Data processing ########################################
def get_cache_data_set(data,nsample=100):
    data_shape = np.shape(data)
    idx = np.random.randint(0,data_shape[0],nsample)
    samples = data[idx]
    return samples[:,:-300], idx

######################################## Utils functions ########################################
def get_loss(logits,targets,targets_GT=False):
    """
    logits: list of logits: 300x[nsamples*nbsample,1]
    targets: targets pixels:
        - if targets_GT True: tensor of GT pixels [nbsample,300]
        - if targets_GT False: list of predicted pixels 300x[nsamples*nbsample,1]
    """
    # Reshapping logits
    log = tf.stack(logits,axis=0) # log shape: [299,nsamples*nbsample,1]
    log = tf.reshape(tf.transpose(log,perm=[1,0,2]),[-1,300]) # log shape: [nsamples*nbsample,299]
    # Reshapping targets
    if not targets_GT:
        tar = tf.stack(targets,axis=0) # tar shape: [299,nsamples*nbsample,1]
        tar = tf.reshape(tf.transpose(tar,perm=[1,0,2]),[-1,300]) # tar shape: [nsamples*nbsample,299]
    elif targets_GT:
        tar = tf.tile(targets,[1,nsamples]) # tar shape: [nbsample,299*nsamples]
        tar = tf.reshape(tar,[-1,300]) # tar shape: [nsamples*nbsample,299]
    loss_300 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar,logits=log))
    loss_28 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar[:,:28],logits=log[:,:28]))
    loss_10 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar[:,:10],logits=log[:,:10]))
    loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar[:,1],logits=log[:,1]))

    return [loss_300, loss_28, loss_10, loss_1]

def inpaint_images(data, idx, predictions, npixels):
    data_shape = np.shape(data)
    nbtodraw = np.shape(idx)[0]
    # Original
    original_data = data[idx] #shape: [nbtodraw, 28x28]
    # tiling original as in build_model
    tilded_original_data = np.tile(original_data,(1,nsamples)) #shape: [nbtodraw, 28x28*nsamples]
    tilded_original_data = np.reshape(tilded_original_data,[nbtodraw*nsamples,-1]) #shape: [nbtodraw*nsamples,28x28]
    original_data_tostack = np.reshape(original_data,[nbtodraw,-1,1])
    # Cache data with ones
    cache_data = np.ones_like(original_data) #shape: [nbtodraw, 28x28]
    cache_data[:,:-300] = original_data[:,:-300] #shape: [nbtodraw, 28x28]
    tilded_cache_data = np.tile(cache_data,(1,nsamples)) #shape: [nbtodraw, 28x28*nsamples]
    tilded_cache_data = np.reshape(tilded_cache_data,[nbtodraw*nsamples,-1]) #shape: [nbtodraw*nsamples,28x28]
    cache_data_tostack = np.reshape(cache_data,[nbtodraw,-1,1])
    # prediction for nsamples samples
    # preprocess predictions
    preds = tf.stack(predictions,axis=1) #shape: [nsamplesxnbsample,300,1]
    preds = tf.reshape(preds,[data_shape[0]*nsamples,-1]) #shape: [nsamplesxnbsample,300]
    preds= preds.eval() #convert tensor to ndarray
    preds = np.split(preds, data_shape[0], 0) #shape: nbsamples*[nsample, 300]
    preds = np.stack(preds,axis=0) #shape: [nsample, nbsamples, 300]
    preds_todraw = np.take(preds,idx,axis=0) #shape: [nbtodraw, nbsamples, 300]
    preds_todraw = np.reshape(preds_todraw, [nbtodraw*nsamples,-1]) #shape: [nbtodraw*nbsamples, 300]
    # create images to inpaint
    im_pred = tilded_cache_data
    # replacing original by prediction
    if npixels==300:
        im_pred[:,-300:] = preds_todraw[:,:npixels] #shape: [nbtodraw*nsamples, 28x28]
    else:
        im_pred[:,-300:-300+npixels] = preds_todraw[:,:npixels]  #shape: [nbtodraw*nsamples, 28x28]
    im_pred = np.reshape(im_pred,[nbtodraw,nsamples,-1]) #shape: [nbtodraw,nsamples, 28x28]
    im_pred = np.transpose(im_pred,[0,2,1]) #shape: [nbtodraw, 28x28, nsamples]
    # stacking all the images, first filter is the original, second is the cache, remaining are the predictions
    images = np.concatenate((original_data_tostack,cache_data_tostack,im_pred), axis=2) #shape: [nbtodraw, 28x28, 2+nbsamples]
    images = np.reshape(images,[nbtodraw,IMAGE_SIZE,IMAGE_SIZE,-1])*255.0 #shape: [nbtodraw, 28, 28, 2+nbsamples]
    return images.astype("int32")

def save_images(images,NB_PIX):
    data_shape = np.shape(images)
    for i in range(data_shape[0]):
        DIR_NAME = "./inpainting/example_" + str(i)
        if not tf.gfile.Exists(DIR_NAME):
            tf.gfile.MkDir(DIR_NAME)
        for j in range(data_shape[3]):
            fig=plt.figure()
            if j==0:
                plt.title("Original", fontsize=20)
                FILE_NAME = "original.png"
                DST = os.path.join(DIR_NAME, FILE_NAME)
            elif j==1:
                plt.title("cache", fontsize=20)
                FILE_NAME = "cache.png"
                DST = os.path.join(DIR_NAME, FILE_NAME)
            else:
                plt.title("inpainting " + str(NB_PIX), fontsize=20)
                SUB_DIR = "sample_" + str(j-2)
                SUB_DIR_PATH = os.path.join(DIR_NAME,SUB_DIR)
                if not tf.gfile.Exists(SUB_DIR_PATH):
                    os.makedirs(SUB_DIR_PATH)
                FILE_NAME = str(NB_PIX) + ".png"
                FILE_NAME = os.path.join(SUB_DIR,FILE_NAME)
            DST = os.path.join(DIR_NAME, FILE_NAME)
            plt.imshow(images[i,:,:,j], cmap="gray", interpolation=None)
            plt.axis("on")
            fig.savefig(DST)
            plt.close()

def create_DST_DIT(name_model):
    NAME = "model_" + str(name_model) + ".ckpt"
    DIR = "models"
    SUB_DIR = os.path.join(DIR,NAME[:-5])
    if not tf.gfile.Exists(SUB_DIR):
        os.makedirs(SUB_DIR)
    DST = os.path.join(SUB_DIR,NAME)
    return DST

######################################## Main ########################################
def in_painting(model_archi,gt_data,cache_data):
    nn_model = model_archi["name"]
    # get weights path
    DST = create_DST_DIT(nn_model)

    print("\nPreparing variables and building model {}...".format(nn_model))
    ###### Create tf placeholder ######
    test_data_node = tf.placeholder(dtype = part2.data_type(), shape=(np.shape(gt_data)[0], np.shape(gt_data)[1]))
    cache_data_node = tf.placeholder(dtype = part2.data_type(), shape=(np.shape(cache_data)[0], np.shape(cache_data)[1]))
    ###### Build model and loss ######
    test_logits, test_pred = build_model.model_inpainting(cache_data_node,
                                                            name=nn_model,
                                                            cell=model_archi["cell"],
                                                            nlayers=model_archi["layers"],
                                                            nunits=model_archi["units"],
                                                            nsamples=nsamples,
                                                            training=False)
    grtr_cross_entropy = get_loss(logits=test_logits,targets=test_data_node[:,-300:],targets_GT=True)
    pred_cross_entropy = get_loss(logits=test_logits,targets=test_pred)

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    """
    vars_ =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    name_vars = [v.name for v in vars_]
    #pdb.set_trace()
    """

    ###### Create a local session to run the training ######
    with tf.Session() as sess:
        print("\nStart testing...")
        start_time = time.time()
        tf.global_variables_initializer().run()
        csvfileTest = open('Perf/test_' + str(nn_model) + '.csv', 'w')
        Testwriter = csv.writer(csvfileTest, delimiter=';',)
        Testwriter.writerow(['Predict CE 300','Grount truth CE 300',
                            'Predict CE 28','Grount truth CE 28',
                            'Predict CE 10','Grount truth CE 10'
                            'Predict CE 1','Grount truth CE 1'])
        # Testing
        if not tf.gfile.Exists(DST):
            raise Exception("no weights given")
        saver.restore(sess, DST)
        # Compute and print results once training is done
        predictions_pixel, prediction_XE, GroundTruth_XE = sess.run([test_pred, grtr_cross_entropy, pred_cross_entropy],
                                                    feed_dict={test_data_node: gt_data, cache_data_node: cache_data})
        print("Testing done, took: {:.4f}s".format(time.time()-start_time))
        print("predicted Xent 300: {:.4f}, ground-truth Xent 300: {:.4f}".format(prediction_XE[0],GroundTruth_XE[0]))
        Testwriter.writerow([prediction_XE[0],GroundTruth_XE[0],
                            prediction_XE[1],GroundTruth_XE[1],
                            prediction_XE[2],GroundTruth_XE[2],
                            prediction_XE[3],GroundTruth_XE[3]])

        # in painting images and save
        print("\nStart inpainting...")
        """
        test = gt_data[:,-300:]
        shpe = np.shape(test)
        test = np.tile(test,(1,nsamples))
        test = np.reshape(test,[shpe[0]*nsamples,-1])
        test_list = [test[:,i] for i in range(np.shape(test)[1])]
        """
        idxtosave = np.random.randint(0,np.shape(cache_data)[0],5)
        #npixels = [300,]
        npixels = [1, 10, 28, 300]
        for npixel in npixels:
            #inpainting_images = inpaint_images(gt_data, idxtosave, test_list, npixel)
            inpainting_images = inpaint_images(gt_data, idxtosave, predictions_pixel, npixel)
            NB_PIX = str(npixel) + "pixels"
            save_images(inpainting_images,NB_PIX)
