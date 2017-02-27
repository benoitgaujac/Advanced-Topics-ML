import gzip
import os
import sys
import time
import pdb

import numpy as np
import csv
from six.moves import urllib
from sklearn.utils import shuffle
import tensorflow as tf

import logging
logging.basicConfig(filename='out.log', level=logging.DEBUG)

import build_model
import inpainting


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 2048
BATCH_SIZE_EVAL = 2048
nsample = 100
nsamples = 11

num_epochs = 100
epochs_per_checkpoint = 2

from_pretrained_weights = False

from optparse import OptionParser
parser = OptionParser()
parser.add_option('-m', '--model', action='store', dest='model',
    help="NN models in {Onelinear,OneHidden,TwoHidden,Conv1}")
parser.add_option('-s', '--mode', action='store', dest='mode',
    help="train, test or inpainting mode")

######################################## Models architectures ########################################
gru1l32u = {"name": "gru1l32u", "cell": "GRU", "layers": 1, "units":32, "init_learning_rate": 0.01}
gru1l64u = {"name": "gru1l64u", "cell": "GRU", "layers": 1, "units":64, "init_learning_rate": 0.005}
gru1l128u = {"name": "gru1l128u", "cell": "GRU", "layers": 1, "units":128, "init_learning_rate": 0.005}
gru3l32u = {"name": "gru3l32u", "cell": "GRU", "layers": 3, "units":32, "init_learning_rate": 0.01}

models = {"gru1l32u":gru1l32u, "gru1l64u":gru1l64u,
        "gru1l128u": gru1l128u, "gru3l32u": gru3l32u}

#models = {"gru1l32u":gru1l32u, "gru1l64u":gru1l64u,
#        "gru1l128u": gru1l128u}

######################################## Data processing ########################################
def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    return tf.float32

def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def get_data():
    # download the data id needed
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def get_cache_data_set(data,nsample=100):
    data_shape = np.shape(data)
    idx = np.random.randint(0,data_shape[0],nsample)
    samples = data[idx]
    return samples[:,:-300], idx

######################################## Utils functions ########################################
def accuracy(predictions,labels):
    correct_prediction = (np.argmax(predictions, 1)==labels)
    return np.mean(correct_prediction)

def accuracy_logistic(predictions,labels):
    pred_int = np.rint(predictions).astype(np.int32)
    lab_int = labels.astype(np.int32)
    correct_prediction = (pred_int==lab_int)
    return np.mean(correct_prediction)

def binarize(images, threshold=0.1):
    return (threshold < images).astype("float32")

def create_DST_DIT(name_model):
    NAME = "model_" + str(name_model) + ".ckpt"
    DIR = "models"
    SUB_DIR = os.path.join(DIR,NAME[:-5])
    if not tf.gfile.Exists(SUB_DIR):
        os.makedirs(SUB_DIR)
    DST = os.path.join(SUB_DIR,NAME)
    return DST

def get_batches(images, batch_size=BATCH_SIZE):
    batches = []
    X = shuffle(images)
    for i in range(int(X.shape[0]/batch_size)+1):
        if i<int(X.shape[0]/batch_size):
            X_batch = X[i * batch_size: (i + 1) * batch_size]
        else:
            X_batch = X[-batch_size:]
        batches.append(X_batch)
    return batches

######################################## Main ########################################
def main(model_archi,train_data, validation_data, test_data, mode_):
    nn_model = model_archi["name"]
    # Create weights DST dir
    DST = create_DST_DIT(nn_model)

    train_size = train_data.shape[0]

    start_time = time.time()
    print("\nPreparing variables and building model {}...".format(nn_model))
    ###### Create tf placeholder ######
    data_node = tf.placeholder(dtype=data_type(), shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS))

    ###### Build model and loss ######
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    logits = build_model.model(data_node,   name=nn_model,
                                            cell=model_archi["cell"],
                                            nlayers=model_archi["layers"],
                                            nunits=model_archi["units"],
                                            training=phase_train)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                            targets=data_node[:,1:],
                                            logits=logits))
    ###### Create varaible for batch ######
    batch = tf.Variable(0, dtype=data_type())
    ###### CLearning rate decay ######
    learning_rate = tf.train.exponential_decay(
                    model_archi["init_learning_rate"],  # Base learning rate.
                    batch * BATCH_SIZE,                 # Current index into the dataset.
                    5*train_size,                       # Decay step.
                    0.90,                               # Decay rate.
                    staircase=True)
    ###### Optimizer ######
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=batch)
    ###### Predictions for the current training minibatch ######
    prediction = tf.sigmoid(logits)
    ###### Saver ######
    saver = tf.train.Saver()

    ###### Create a local session to run the training ######
    with tf.Session() as sess:
        # Training
        if mode_!="test":
            # Opening csv file
            csvfileTrain = open('Perf/Training_' + str(nn_model) + '.csv', 'w')
            Trainwriter = csv.writer(csvfileTrain, delimiter=';',)
            Trainwriter.writerow(['Num Epoch', 'Time', 'Training loss', 'Validation loss'])
            # Load pre trained model if exist
            if not tf.gfile.Exists(DST + ".meta") or not from_pretrained_weights:
                sess.run(tf.global_variables_initializer(), feed_dict={phase_train: True})
            else:
                saver.restore(sess, DST)
                # Reinitialize learning rate
                tf.variables_initializer([batch,]).run()
                learning_rate = tf.train.exponential_decay(
                                model_archi["init_learning_rate"],  # Base learning rate.
                                batch * BATCH_SIZE,                 # Current index into the dataset.
                                5*train_size,                       # Decay step.
                                0.85,                               # Decay rate.
                                staircase=True)
            # initialize performance indicators
            loss_history = [10000.0,]
            best_train_loss, best_eval_loss = 10000.0, 10000.0
            best_train_acc, best_eval_acc = 0.0, 0.0
            #training loop
            print("\nStart training {}...".format(nn_model))
            logging.info("Start training {}...".format(nn_model))
            for epoch in range(num_epochs):
                start_time = time.time()
                train_loss, train_acc = 0.0, 0.0
                print("")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                Batches = get_batches(train_data, BATCH_SIZE)
                for batch_ in Batches:
                    feed_dict = {data_node: batch_, phase_train: True}
                    # Run the optimizer to update weights.
                    sess.run(optimizer, feed_dict=feed_dict)
                    l, lr, predictions = sess.run([loss, learning_rate, prediction], feed_dict=feed_dict)
                    # Update average loss and accuracy
                    train_loss += l / len(Batches)
                    train_acc += accuracy_logistic(predictions,batch_[:,1:]) / len(Batches)
                if train_acc>best_train_acc:
                    best_train_acc = train_acc
                if train_loss<best_train_loss:
                    best_train_loss = train_loss
                # Print info for previous epoch
                print("Epoch {} done, took {:.2f}s, learning rate: {:.2f}e-3".format(epoch,time.time()-start_time,lr*1000))
                logging.info("Epoch {} done, took {:.2f}s, learning rate: {:.2f}e-3".format(epoch,time.time()-start_time,lr*1000))
                print("Epoch loss: {:.4f}, Best train loss: {:.4f}, Best train accuracy: {:.2f}%".format(
                                                            train_loss,best_train_loss,best_train_acc*100))
                logging.info("Epoch loss: {:.4f}, Best train loss: {:.4f}, Best train accuracy: {:.2f}%".format(
                                                            train_loss,best_train_loss,best_train_acc*100))
                # Perform evaluation
                if epoch % epochs_per_checkpoint==0:
                    eval_loss_, eval_acc = 0.0, 0.0
                    eval_Batches = get_batches(validation_data, BATCH_SIZE_EVAL)
                    for eval_batch in eval_Batches:
                        ev_loss, ev_pred = sess.run([loss, prediction], feed_dict={
                                                                data_node: eval_batch,
                                                                phase_train: False})
                        eval_loss_ += ev_loss / len(eval_Batches)
                        eval_acc += accuracy_logistic(ev_pred,eval_batch[:,1:]) / len(eval_Batches)
                    if eval_acc>best_eval_acc:
                        best_eval_acc = eval_acc
                        saver.save(sess,DST)
                    if eval_loss_<best_eval_loss:
                        best_eval_loss = eval_loss_
                    print("Validation loss: {:.4f}, Best validation loss: {:.4f}, Best validation accuracy: {:.2f}%".format(
                                                                                    eval_loss_,best_eval_loss,best_eval_acc*100))
                    logging.info("Validation loss: {:.4f}, Best validation loss: {:.4f}, Best validation accuracy: {:.2f}%".format(
                                                                                    eval_loss_,best_eval_loss,best_eval_acc*100))
                    sys.stdout.flush()
                # Writing csv file with results and saving models
                Trainwriter.writerow([epoch + 1, time.time() - start_time, train_loss, eval_loss_])
        # Testing
        csvfileTest = open('Perf/test_' + str(nn_model) + '.csv', 'w')
        Testwriter = csv.writer(csvfileTest, delimiter=';',)
        Testwriter.writerow(['Test loss'])
        if not tf.gfile.Exists(DST+ ".meta"):
            raise Exception("no weights given")
        saver.restore(sess, DST)
        # Compute and print results once training is done
        test_loss, test_acc = 0.0, 0.0
        test_Batches = get_batches(test_data, BATCH_SIZE_EVAL)
        for test_batch in test_Batches:
            tst_loss, tst_pred = sess.run([loss, prediction], feed_dict={
                                                    data_node: test_batch,
                                                    phase_train: False})
            test_loss += tst_loss / len(test_Batches)
            test_acc += accuracy_logistic(tst_pred,test_batch[:,1:]) / len(test_Batches)
        print("\nTesting after {} epochs.".format(num_epochs))
        print("Test loss: {:.4f}, Test acc: {:.2f}%".format(test_loss,test_acc*100))
        logging.info("\nTest loss: {:.4f}, Test acc: {:.2f}%".format(test_loss,test_acc*100))
        Testwriter.writerow([test_loss])

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

    #train_data = train_data[:3000]

    options, arguments = parser.parse_args(sys.argv)
    if options.model not in models.keys():
        raise Exception("Invalide model name")
    else:
        if options.mode!="inpainting":
            main(models[options.model],train_data, validation_data, test_data, options.mode)
        else:
            # pixel in-painting
            cache_data, idx = get_cache_data_set(test_data,nsample=nsample)
            inpainting.in_painting(models[options.model],test_data[idx],cache_data)
