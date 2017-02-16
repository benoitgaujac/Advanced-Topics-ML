import gzip
import os
import sys
import time
import pdb

import numpy as np
from six.moves import urllib
import tensorflow as tf
import build_model
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

num_epochs = 11
epochs_per_checkpoint = 5

from optparse import OptionParser
parser = OptionParser()
parser.add_option('-m', '--model', action='store', dest='model',
    help="NN models in {Onelinear,OneHidden,TwoHidden,Conv1}")
parser.add_option('-s', '--mode', action='store', dest='mode',
    help="train, test or inpainting mode")

######################################## Models architectures ########################################
#lstm1l32u = {"name": "lstm1l32u", "cell": "LSTM", "layers": 1, "units":32}
#lstm1l64u = {"name": "lstm1l64u", "cell": "LSTM", "layers": 1, "units":64}
#lstm1l128u = {"name": "lstm1l128u", "cell": "LSTM", "layers": 1, "units":128}
#lstm3l32u = {"name": "lstm3l32u", "cell": "LSTM", "layers": 3, "units":32}
gru1l32u = {"name": "gru1l32u", "cell": "GRU", "layers": 1, "units":32}
gru1l64u = {"name": "gru1l64u", "cell": "GRU", "layers": 1, "units":64}
gru1l128u = {"name": "gru1l128u", "cell": "GRU", "layers": 1, "units":128}
gru3l32u = {"name": "gru3l32u", "cell": "GRU", "layers": 3, "units":32}
"""
models = {"lstm1l32u": lstm1l32u,"lstm1l64u":lstm1l64u, "lstm1l128u": lstm1l128u,
        "lstm3l32u":lstm3l32u, "gru1l32u":gru1l32u, "gru1l64u":gru1l64u,
        "gru1l128u": gru1l128u, "gru3l32u": gru3l32u}
"""
models = {"gru1l32u":gru1l32u, "gru1l64u": gru1l64u,
        "gru1l128u": gru1l128u , "gru3l32u": gru3l32u}

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

def binarize(images, threshold=0.1):
    return (threshold < images).astype("float32")

######################################## Main ########################################
def main(model_archi,train_data, validation_data, test_data, mode_):
    nn_model = model_archi["name"]
    train_size = train_data.shape[0]

    print("\nPreparing variables and building model {}...".format(nn_model))
    ###### Create tf placeholder ######
    train_data_node = tf.placeholder(dtype=data_type(),
                                    shape=(BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS))
    eval_data_node = tf.placeholder(dtype = data_type(),
                                    shape=(np.shape(validation_data)[0], IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS))
    test_data_node = tf.placeholder(dtype = data_type(),
                                    shape=(np.shape(test_data)[0], IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS))

    ###### Build model and loss ######
    with tf.variable_scope(nn_model) as scope:
        # Training
        logits = build_model.model(train_data_node, name=nn_model,
                                                    cell=model_archi["cell"],
                                                    nlayers=model_archi["layers"],
                                                    nunits=model_archi["units"],
                                                    training=True)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                targets=train_data_node[:,1:],
                                                logits=logits))
        scope.reuse_variables()
        # Validation and testing
        eval_logits = build_model.model(eval_data_node, name=nn_model,
                                                        cell=model_archi["cell"],
                                                        nlayers=model_archi["layers"],
                                                        nunits=model_archi["units"],
                                                        training=False)
        eval_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                targets=eval_data_node[:,1:],
                                                logits=eval_logits))

        #scope.reuse_variables()
        test_logits = build_model.model(test_data_node, name=nn_model,
                                                        cell=model_archi["cell"],
                                                        nlayers=model_archi["layers"],
                                                        nunits=model_archi["units"],
                                                        training=False)
        test_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                targets=test_data_node[:,1:],
                                                logits=test_logits))

    ###### Create varaible for batch ######
    batch = tf.Variable(0, dtype=data_type())

    ###### CLearning rate decay ######
    learning_rate = tf.train.exponential_decay(
                    0.005,                # Base learning rate.
                    batch * BATCH_SIZE,  # Current index into the dataset.
                    5*train_size,          # Decay step.
                    0.98,                # Decay rate.
                    staircase=True)

    ###### Optimizer ######
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=batch)

    ###### Predictions for the current training minibatch ######
    train_prediction = tf.sigmoid(logits)

    ###### Predictions for the validation ######
    eval_prediction = tf.sigmoid(eval_logits)

    ###### Predictions for the test ######
    test_prediction = tf.sigmoid(test_logits)

    ###### Saver ######
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

    ###### Create a local session to run the training ######
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Opening csv file
        if mode_!="test":
            csvfileTrain = open('Perf/Training_' + str(nn_model) + '.csv', 'w')
            Trainwriter = csv.writer(csvfileTrain, delimiter=';',)
            Trainwriter.writerow(['Num Epoch', 'Time', 'Training loss', 'Validation loss'])
        csvfileTest = open('Perf/Val_' + str(nn_model) + '.csv', 'w')
        Testwriter = csv.writer(csvfileTest, delimiter=';',)
        Testwriter.writerow(['Test error'])

        # Training
        if mode_!="test":
            # initialize performance indicators
            best_train_loss, best_eval_loss = 10000.0, 10000.0
            best_train_acc, best_eval_acc = 0.0, 0.0
            #training loop
            print("Start training...")
            for epoch in range(num_epochs):
                start_time = time.time()
                train_loss, train_acc = 0.0, 0.0
                print("")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                n_step = int(float(train_size) / BATCH_SIZE)
                for step in range(n_step):
                    # Note that we could use better randomization across epochs.
                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
                    feed_dict = {train_data_node: batch_data,}
                    # Run the optimizer to update weights.
                    sess.run(optimizer, feed_dict=feed_dict)
                    l, lr, predictions = sess.run([loss, learning_rate, train_prediction], feed_dict=feed_dict)
                    # Update average loss and accuracy
                    train_loss += l / n_step
                    train_acc += accuracy_logistic(predictions,batch_data[:,1:]) / n_step
                # Print info for previous epoch
                print("Epoch {} done, took {:.2f}s, learning rate: {:.2f}e-4".format(epoch,time.time()-start_time,lr*10000))
                if train_acc>best_train_acc:
                    best_train_acc = train_acc
                if train_loss<best_train_loss:
                    best_train_loss = train_loss
                print("Epoch loss: {:.4f}, Best train loss: {:.4f}, Best train accuracy: {:.2f}%".format(
                                                            train_loss,best_train_loss,best_train_acc*100))
                # Perform evaluation
                if epoch % epochs_per_checkpoint==0:
                    ev_loss, eval_pred = sess.run([eval_loss, eval_prediction],
                                    feed_dict={eval_data_node: validation_data,})
                    eval_acc = accuracy_logistic(eval_pred,validation_data[:,1:])
                    if eval_acc>best_eval_acc:
                        best_eval_acc = eval_acc
                        saver.save(sess,"models/model_" + str(nn_model) + ".ckpt")
                    if ev_loss<best_eval_loss:
                        best_eval_loss = ev_loss
                    print("Validation loss: {:.4f}, Best validation loss: {:.4f}, Best validation accuracy: {:.2f}%".format(
                                                                                    ev_loss,best_eval_loss,best_eval_acc*100))
                    sys.stdout.flush()
                # Writing csv file with results and saving models
                Trainwriter.writerow([epoch + 1, time.time() - start_time, train_loss, eval_loss])
        # Testing
        WORK_DIRECTORY = "./models/model_" + str(nn_model) + ".ckpt"
        if not tf.gfile.Exists(WORK_DIRECTORY):
            raise Exception("no weights given")
        saver.restore(sess, WORK_DIRECTORY)
        scope.reuse_variables()
        # Compute and print results once training is done
        tst_loss, test_pred = sess.run([test_loss, test_prediction], feed_dict={
                                                        test_data_node: test_data,})
        test_acc = accuracy_logistic(test_pred,test_data[:,1:])
        print("\nTesting after {} epochs.".format(num_epochs))
        print("Test loss: {:.4f}, Test acc: {:.2f}%, Test err: {:.2f}%".format(tst_loss,test_acc*100, 100 - test_acc*100))
        Testwriter.writerow([1 - test_acc,test_acc])


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
    if options.mode!="inpainting":
        # run for model
        if options.model not in models:
            for model_ in models.keys():
                main(models[model_],train_data, validation_data, test_data, options.mode)
        else:
            main(models[options.model],train_data, validation_data, test_data, options.mode)
    elif option.mode=="inpainting":
        sfve
    else:
        raise Exception("mode must be train, test or inpainting")
