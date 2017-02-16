import gzip
import os
import sys
import time
import pdb

import numpy as np
from six.moves import urllib
import tensorflow as tf
import build_model
import part2
import csv
from sklearn.utils import shuffle


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 128


######################################## Data processing ########################################
def get_cache_data_set(data,nsample=nsample):
    data_shape = np.shape(data)
    idx = np.random.randint(0,data_shape[0],nsample)
    samples = data[idx]
    return samples[:,:-300]

######################################## Utils functions ########################################
def accuracy(predictions,labels):
    correct_prediction = (np.argmax(predictions, 1)==labels)
    return np.mean(correct_prediction)

def accuracy_logistic(predictions,labels):
    #pred_int = tf.to_int32(tf.rint(predictions))
    #lab_int = tf.to_int32(labels)
    pred_int = np.rint(predictions).astype(np.int32)
    lab_int = labels.astype(np.int32)
    correct_prediction = (pred_int==lab_int)
    #pdb.set_trace()
    return np.mean(correct_prediction)

def get_loss(logits,targets,targets_GT=False):
    """
    logits: list of logits: 299x[10*nbsample,1]
    targets: targets pixels:
        - if targets_GT True: tensor of GT pixels [nbsample,299]
        - if targets_GT False: list of predicted pixels 299x[10*nbsample,1]
    """
    # Reshapping logits
    log = tf.stack(logits,axe=0) # log shape: [299,10*nbsample,1]
    log = tf.reshape(tf.transpose(log,perm=[1,0,2]),[-1,299]) # log shape: [10*nbsample,299]
    # Reshapping targets
    if not targets_GT:
        tar = tf.stack(targets,axe=0) # tar shape: [299,10*nbsample,1]
        tar = tf.reshape(tf.transpose(tar,perm=[1,0,2]),[-1,299]) # tar shape: [10*nbsample,299]
    elif targets_GT:
        tar = tf.tile(targets,[0,10]) # tar shape: [nbsample,299*10]
        tar = tf.reshape(tar,[-1,299]) # tar shape: [10*nbsample,299]
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar,logits=log))

    return loss

######################################## Main ########################################
def in_painting(model_archi,data,nsample=100):
    nn_model = model_archi["name"]
    cache_data = get_cache_data_set(data,nsample=nsample)

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

        csvfileTest = open('Perf/Val_' + str(nn_model) + '.csv', 'w')
        Testwriter = csv.writer(csvfileTest, delimiter=';',)
        Testwriter.writerow(['Predict CE','Grount truth CE'])
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
        print("Predict Xent: {:.4f}, Ground truth Xent: {:.4f}".format(prediction_XE,GroundTruth_XE))
        Testwriter.writerow([rediction_XE,GroundTruth_XE])


if __name__ == '__main__':
    ###### Load and get data ######
    train_data, _, validation_data, _, test_data, _ = get_data()
    # Reshape data
    train_data = np.reshape(train_data,[-1,IMAGE_SIZE*IMAGE_SIZE])
    validation_data = np.reshape(validation_data,[-1,IMAGE_SIZE*IMAGE_SIZE])
    test_data = np.reshape(test_data,[-1,IMAGE_SIZE*IMAGE_SIZE])
    # Convert to binary
    print("Converting data to binary")
    train_data = binarize(train_data)
    validation_data = binarize(validation_data)
    test_data = binarize(test_data)
    # Shuffle train data
    np.random.shuffle(train_data)

    train_data = train_data[:2000]
    #validation_data = validation_data[:2000]

    options, arguments = parser.parse_args(sys.argv)
    # run for model
    if options.model not in models:
        for model_ in models.keys():
            main(models[model_],train_data, validation_data, test_data, options.mode)
    else:
        main(models[options.model],train_data, validation_data, test_data, options.mode)
