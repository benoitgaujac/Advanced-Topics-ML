import gzip
import os
import sys
import time
import pdb

import numpy as np
import csv
from six.moves import urllib
import tensorflow as tf
from sklearn.utils import shuffle

import build_model

import logging



SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 256
BATCH_SIZE_EVAL = 256

num_epochs = 100
epochs_per_checkpoint = 2

from_pretrained_weights = False

from optparse import OptionParser
parser = OptionParser()
parser.add_option('-m', '--model', action='store', dest='model',
    help="NN models in {Onelinear,OneHidden,TwoHidden,Conv1}")
parser.add_option('-s', '--mode', action='store', dest='mode',
    help="testing or training mode")

######################################## Models architectures ########################################
lstm1l32u = {"name": "lstm1l32u", "cell": "LSTM", "layers": 1, "units":32, "init_learning_rate": 0.01}
lstm1l64u = {"name": "lstm1l64u", "cell": "LSTM", "layers": 1, "units":64, "init_learning_rate": 0.005}
lstm1l128u = {"name": "lstm1l128u", "cell": "LSTM", "layers": 1, "units":128, "init_learning_rate": 0.005}
lstm3l32u = {"name": "lstm3l32u", "cell": "LSTM", "layers": 3, "units":32, "init_learning_rate": 0.01}
gru1l32u = {"name": "gru1l32u", "cell": "GRU", "layers": 1, "units":32, "init_learning_rate": 0.01}
gru1l64u = {"name": "gru1l64u", "cell": "GRU", "layers": 1, "units":64, "init_learning_rate": 0.001}
gru1l128u = {"name": "gru1l128u", "cell": "GRU", "layers": 1, "units":128, "init_learning_rate": 0.001}
gru3l32u = {"name": "gru3l32u", "cell": "GRU", "layers": 3, "units":32, "init_learning_rate": 0.01}
#models = {"lstm1l32u": lstm1l32u,"lstm1l64u":lstm1l64u, "lstm1l128u": lstm1l128u,
#        "lstm3l32u":lstm3l32u, "gru1l32u":gru1l32u, "gru1l64u":gru1l64u,
#        "gru1l128u": gru1l128u, "gru3l32u": gru3l32u}
models = {"lstm1l64u":lstm1l64u, "lstm3l32u":lstm3l32u,
        "gru1l32u":gru1l32u, "gru1l64u":gru1l64u,
        "gru3l32u": gru3l32u}


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

def get_batches(images, labels, batch_size=BATCH_SIZE):
    batches = []
    X, y = shuffle(images, labels)
    for i in range(int(X.shape[0]/batch_size)+1):
        if i<int(X.shape[0]/batch_size):
            X_batch = X[i * batch_size: (i + 1) * batch_size]
            y_batch = y[i * batch_size: (i + 1) * batch_size]
        else:
            X_batch = X[-batch_size:]
            y_batch = y[-batch_size:]
        batches.append([X_batch, y_batch])
    return batches

######################################## Main ########################################
def main(model_archi,train_data, train_labels, validation_data, validation_labels, test_data, test_labels, mode_):


    nn_model = model_archi["name"]
    # Create weights dst DIR
    DST = create_DST_DIT(nn_model)
    print(DST)

    train_size = train_labels.shape[0]
    print("\nPreparing variables and building model {}...".format(nn_model))
    ###### Create tf placeholder ######
    train_data_node = tf.placeholder(data_type(),shape=(BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data_node = tf.placeholder(data_type(),shape=(BATCH_SIZE_EVAL, IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS))
    eval_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE_EVAL,))

    ###### Build model and loss ######
    with tf.variable_scope(nn_model) as scope:
        # Training
        logits = build_model.model(train_data_node, name=nn_model,
                                                    cell=model_archi["cell"],
                                                    nlayers=model_archi["layers"],
                                                    nunits=model_archi["units"],
                                                    training=True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=train_labels_node, logits=logits))

        scope.reuse_variables()
        # Validation and testing
        eval_logits = build_model.model(eval_data_node, name=nn_model,
                                                        cell=model_archi["cell"],
                                                        nlayers=model_archi["layers"],
                                                        nunits=model_archi["units"],
                                                        training=False)
        eval_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=eval_labels_node, logits=eval_logits))

    ###### Create varaible for batch ######
    batch = tf.Variable(0, dtype=data_type())
    ###### CLearning rate decay ######
    learning_rate = tf.train.exponential_decay(
                    model_archi["init_learning_rate"],  # Base learning rate.
                    batch * BATCH_SIZE,                 # Current index into the dataset.
                    5*train_size,                       # Decay step.
                    0.85,                               # Decay rate.
                    staircase=True)

    ###### Optimizer ######
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=batch)
    """
    ###### Optimizer ######
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=batch)
    """
    ###### Predictions for the current training minibatch ######
    train_prediction = tf.nn.softmax(logits)

    ###### Predictions for the validation ######
    eval_prediction = tf.nn.softmax(eval_logits)

    ###### Saver ######
    #saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver = tf.train.Saver()

    ###### Create a local session to run the training ######
    with tf.Session() as sess:
        # Training
        if mode_!="test":
            # opening csv file
            csvfileTrain = open('Perf/Training_' + str(nn_model) + '.csv', 'w')
            Trainwriter = csv.writer(csvfileTrain, delimiter=';',)
            Trainwriter.writerow(['Num Epoch', 'Time', 'Training loss', 'Training accuracy', 'Validation loss','Validation accuracy'])

            # Load pre trained models
            if not tf.gfile.Exists(DST) or not from_pretrained_weights:
                tf.global_variables_initializer().run()
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
            # Training
            # initialize performance indicators
            loss_history = []
            best_train_loss, best_eval_loss = 10000.0, 10000.0
            best_train_acc, best_eval_acc = 0.0, 0.0
            #training loop
            print("Start training...".format(nn_model))
            logging.info("Start training...".format(nn_model))
            for epoch in range(num_epochs):
                start_time = time.time()
                train_loss, train_acc = 0.0, 0.0
                """
                # init learning rate
                lr = model_archi["init_learning_rate"]
                """
                print("")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                Batches = get_batches(train_data, train_labels, BATCH_SIZE)
                for batch_ in Batches:
                    feed_dict={train_data_node: batch_[0],
                            train_labels_node: batch_[1]}
                    """
                    feed_dict={train_data_node: batch_[0],
                            train_labels_node: batch_[1],
                            learning_rate: lr}
                    """
                    # Run the optimizer to update weights.
                    sess.run(optimizer, feed_dict=feed_dict)
                    """
                    l, predictions = sess.run([loss, train_prediction], feed_dict=feed_dict)
                    """
                    l, lr, predictions = sess.run([loss, learning_rate, train_prediction], feed_dict=feed_dict)
                    # Update average loss and accuracy
                    train_loss += l / len(Batches)
                    train_acc += accuracy(predictions,batch_[1]) / len(Batches)
                if train_acc>best_train_acc:
                    best_train_acc = train_acc
                if train_loss<best_train_loss:
                    best_train_loss = train_loss
                # Print info for previous epoch
                print("Epoch {} done, took {:.2f}s, learning rate: {:.2f}e-3".format(epoch,time.time()-start_time,lr*1000))
                logging.info("Epoch {} done, took {:.2f}s, learning rate: {:.2f}e-3".format(epoch,time.time()-start_time,lr*1000))
                print("loss: {:.4f}, Best train loss: {:.4f}".format(train_loss,best_train_loss))
                print("acc: {:.3f}%, Best train acc: {:.3f}%".format(train_acc*100,best_train_acc*100))
                logging.info("loss: {:.4f}, Best train loss: {:.4f}".format(train_loss,best_train_loss))
                logging.info("acc: {:.3f}%, Best train acc: {:.3f}%".format(train_acc*100,best_train_acc*100))

                """
                # update learning: learning_rate<-learning_rate/2 if no improvement over last 3 epochs
                eps = float(best_train_loss)/50
                fct = 2
                if epoch>49:
                    fct=10
                loss_history.append(train_loss)
                if len(loss_history)>5:
                    loss_history.pop(0)
                if best_train_loss - eps < min(loss_history):
                    lr = float(lr)/fct
                """

                # Perform evaluation
                if epoch % epochs_per_checkpoint==0:
                    eval_loss_, eval_acc = 0.0, 0.0
                    eval_Batches = get_batches(validation_data, validation_labels, BATCH_SIZE_EVAL)
                    for eval_batch in eval_Batches:
                        ev_loss, ev_pred = sess.run([eval_loss, eval_prediction], feed_dict={
                                                                eval_data_node: eval_batch[0],
                                                                eval_labels_node: eval_batch[1]})
                        eval_loss_ += ev_loss / len(eval_Batches)
                        eval_acc += accuracy(ev_pred,eval_batch[1]) / len(eval_Batches)
                    if eval_acc>best_eval_acc:
                        best_eval_acc = eval_acc
                        saver.save(sess,DST)
                    if eval_loss_<best_eval_loss:
                        best_eval_loss = eval_loss_
                    print("Val loss: {:.4f}, Best val loss: {:.4f}".format(eval_loss_,best_eval_loss))
                    print("Val acc: {:.2f}%, Best val acc: {:.2f}%".format(eval_acc*100,best_eval_acc*100))
                    logging.info("Val loss: {:.4f}, Best val loss: {:.4f}".format(eval_loss_,best_eval_loss))
                    logging.info("Val acc: {:.2f}%, Best val acc: {:.2f}%".format(eval_acc*100,best_eval_acc*100))
                    sys.stdout.flush()
                # Writing csv file with results and saving models
                Trainwriter.writerow([epoch + 1, time.time() - start_time,
                                    train_loss, train_acc, eval_loss_, eval_acc])

        # Testing
        csvfileTest = open('Perf/test_' + str(nn_model) + '.csv', 'w')
        Testwriter = csv.writer(csvfileTest, delimiter=';',)
        Testwriter.writerow(['Test loss', 'Test accuracy'])
        """
        if not tf.gfile.Exists(DST):
            raise Exception("no weights given")
        saver.restore(sess, DST)
        """
        vars_ =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        name_vars = [v.name for v in vars_]
        # Compute and print results once training is done
        test_loss, test_acc = 0.0, 0.0
        test_Batches = get_batches(test_data, test_labels, BATCH_SIZE_EVAL)
        for test_batch in test_Batches:
            tst_loss, tst_pred = sess.run([eval_loss, eval_prediction], feed_dict={
                                                        eval_data_node: test_batch[0],
                                                        eval_labels_node: test_batch[1]})
            test_loss += tst_loss / len(test_Batches)
            test_acc += accuracy(tst_pred,test_batch[1]) / len(test_Batches)
        print("\nTesting after {} epochs.".format(num_epochs))
        print("Test loss: {:.4f}, Test acc: {:.2f}%".format(test_loss,test_acc*100))
        logging.info("\nTest loss: {:.4f}, Test acc: {:.2f}%".format(test_loss,test_acc*100))
        Testwriter.writerow([test_loss,test_acc])
        #tf.reset_default_graph()
        sess.close()

if __name__ == '__main__':
    logging.basicConfig(filename='out.log', level=logging.DEBUG)

    ###### Load and get data ######
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = get_data()
    train_data = np.reshape(train_data,[-1,IMAGE_SIZE*IMAGE_SIZE])
    validation_data = np.reshape(validation_data,[-1,IMAGE_SIZE*IMAGE_SIZE])
    test_data = np.reshape(test_data,[-1,IMAGE_SIZE*IMAGE_SIZE])
    # Convert to binary
    print("Converting data to binary")
    train_data = binarize(train_data)
    validation_data = binarize(validation_data)
    test_data = binarize(test_data)
    # shuffl data
    train_data, train_labels = shuffle(train_data, train_labels, random_state=SEED)

    #train_data = train_data[:1000]
    #train_labels = train_labels[:1000]

    options, arguments = parser.parse_args(sys.argv)
    # run for model
    if options.model not in models.keys():
        if options.mode=="train":
            for model_ in models.keys():
                main(models[model_],train_data, train_labels, validation_data, validation_labels, test_data, test_labels,options.mode)
        else:
            raise Exception("You have to give one unique existing model to test")
    else:
        main(models[options.model],train_data, train_labels, validation_data, validation_labels, test_data, test_labels,options.mode)
