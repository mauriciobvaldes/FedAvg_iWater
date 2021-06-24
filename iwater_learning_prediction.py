# -*- coding: utf-8 -*-


"""This code is for tensorflow 1.x while the default install of tensorflow is 2.x. You will need to:
    pip uninstall tensorflow
    pip install tensorflow==1.15
This code runs the local training on the sensor"""

import os.path
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from tensorflow.contrib import layers
import random

import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Removing unnecessary warnings


# =============================================================================
# client = MongoClient("mongodb://username:password")
# mydb=client["iWater"]
# #mycollection = mydb["test"]
# mycollection = mydb["iWater_node_01"]
# measure_type=['TC1_D','SA_A','OS_D', 'TC2_C','PH_C','RX_C','TC3_A','CN_A',]
# measure_t=['PH_C','RX_C']
# =============================================================================

class Datafortrain:
    def __init__(self, m_type, num_clients, client_number, data, data2):
        self.LSTM_steps = 108
        result_bins, df_in_train, num_of_bins, df_raw, Xdf1_train, Xdf_TC_train, Xdf1_test, Xdf_TC_test, Ydf_test, \
        Ydf_train, df_X_pre, df_TC_pre, Xdf_train, my_last_time = init_to_db(m_type, self.LSTM_steps, num_clients,
                                                                             client_number, data, data2)
        self.mtype = m_type
        self.bins = result_bins
        self.num_of_bins = num_of_bins
        self.X_train = Xdf1_train
        self.TC_train = Xdf_TC_train
        self.Y_train = Ydf_train
        self.X_test = Xdf1_test
        self.TC_test = Xdf_TC_test
        self.Y_test = Ydf_test
        self.X_for_predict = df_X_pre
        self.TC_for_predict = df_TC_pre
        self.mytime = my_last_time
        self.output = "init"

    def __call__(self, num_clients, client_number, data, data2):
        self.LSTM_steps = 108
        result_bins, df_in_train, num_of_bins, df_raw, Xdf1_train, Xdf_TC_train, Xdf1_test, Xdf_TC_test, Ydf_test, \
        Ydf_train, df_X_pre, df_TC_pre, Xdf_train, my_last_time = init_to_db(self.mtype, self.LSTM_steps, num_clients,
                                                                             client_number, data, data2)
        self.bins = result_bins
        self.num_of_bins = num_of_bins
        self.X_train = Xdf1_train
        self.TC_train = Xdf_TC_train
        self.Y_train = Ydf_train
        self.X_test = Xdf1_test
        self.TC_test = Xdf_TC_test
        self.Y_test = Ydf_test
        self.X_for_predict = df_X_pre
        self.TC_for_predict = df_TC_pre
        self.mytime = my_last_time
        self.output = "call"
        return self


def first(the_iterable, condition=lambda x: True):
    # getting the first element that satisfies the condition
    for i in the_iterable:
        if condition(i):
            return i


def get_data_from_db(my_measure_type, history_length_in_days, num_steps, num_clients, client_number, data):
    """ Fucntion for downsampling to multiple pseudo-clients from the total dataset."""
    df_mytype_withtime = data

    # Put the correct data for only the current client to be handled
    df_mytype_withtime = df_mytype_withtime.drop(
        df_mytype_withtime.index[
            [i for i in range(df_mytype_withtime.shape[0]) if not i % num_clients == client_number]])

    df_mytype = df_mytype_withtime['value']
    df_time = df_mytype_withtime['timestamp_sensor'].tail(1)
    for ddf_time in df_time:
        my_last_time = ddf_time

    output_bins, bins = pd.qcut(df_mytype, 6, retbins=True, duplicates='drop')

    df_new = []
    # num_of_bins=len(bins)-1
    for i in df_mytype:
        j = first(range(1, len(bins)), lambda k: i <= bins[k]) - 1

        df_new.append(j)  # df_new is the whole time series of the quantized (from 0 to 5) where 5 is the bin number -1

    df_X, df_label = data_stacking(df_new, num_steps)  # To prepare the training data, such that each row of df_X is a
    # time series of length num_steps, df_label is the label
    df_X = np.reshape(df_X, (len(df_X), num_steps, 1))

    return bins, df_X, df_label, df_mytype, df_new, my_last_time


def data_stacking(data_in, series_length):
    """Preparing data for training, from a time series of length L to L - series_length number of series with length:
    series length """

    # preparing data for training,
    # from a serie with length L to L-series_length number of series with length series_length
    X = []
    Y = []
    print(len(data_in))
    for i in range(0, len(data_in) - series_length, 1):
        sequence = data_in[i:i + series_length]
        # label = data_in[i + series_length].item(0)
        label = data_in[i + series_length]
        # print("label")
        # print(label)
        X.append(sequence)
        Y.append(label)
    # print("convert done")
    # print(Y)
    return X, Y


def data_slicing_binning(data_in, num_of_bins):
    """Discretizing the values using a quantile-cut"""
    cats, mybins = pd.qcut(data_in, num_of_bins, retbins=True, duplicates='drop')
    return cats, mybins


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def data_preparation(my_type, num_steps, num_clients, client_number, data, data2):
    """Reading data and orgainizing it for training."""
    window_size_max = 365  # max 365 days
    TC_result_bins, df_TC_train, df_TC_label_train, df_TC_raw, df_TC_full, _ = get_data_from_db('TC1_D',
                                                                                                window_size_max,
                                                                                                num_steps,
                                                                                                num_clients,
                                                                                                client_number, data2)
    num_of_TC_bins = len(TC_result_bins) - 1

    result_bins, df_X_train, df_label_train, df_raw, df_full, my_last_time = get_data_from_db(my_type, window_size_max,
                                                                                              num_steps,
                                                                                              num_clients,
                                                                                              client_number, data)
    num_of_bins = len(result_bins) - 1

    df_X_train = df_X_train / float(
        num_of_bins)  # normalization, update the values of df_X from [0,binnumber-1] to [0,1]
    # df_RX_train=df_RX_train/float(20.0)
    df_TC_train = df_TC_train / float(num_of_TC_bins)
    df_in_train = np.concatenate((df_X_train, df_TC_train), axis=2)  # combine X data and temperature data
    # df_RX_in_train=np.concatenate((df_RX_train,df_TC_train),axis=2)

    print(len(df_X_train))
    # plt.plot(df_raw)
    # plt.show()

    df_full2 = df_full.copy()
    for i in range(0, num_of_bins):  # This is to force the onehot process below has the exact same number of classes
        # as the bin number
        df_full2.append(i)
    # Y_modified = tf.one_hot(df_label_train,20)
    Y_onehot = pd.get_dummies(df_full2)
    Y_onehot = Y_onehot[num_steps:-num_of_bins]

    df_pre = [xvalue / float(num_of_bins) for xvalue in df_full]  # normalization of the data for prediction
    df_TC_pre = [xvalue / float(num_of_bins) for xvalue in df_TC_full]  # df_TC_full/float(num_of_bins)

    return result_bins, df_in_train, Y_onehot, num_of_bins, df_raw, df_pre[-num_steps:], df_TC_pre[
                                                                                         -num_steps:], my_last_time


def init_to_db(m_type, num_steps, num_clients, client_number, data, data2):
    """Splitting the data for training and testing, in current implementation of code we have 10% testing data"""
    result_bins, df_in_train, Y_onehot, num_of_bins, df_raw, df_X_pre, df_TC_pre, my_last_time = data_preparation(
        m_type, num_steps, num_clients, client_number, data, data2)
    df_X_pre = np.reshape(df_X_pre, (1, len(df_X_pre), 1))  # normalized input data for prediction
    df_TC_pre = np.reshape(df_TC_pre, (1, len(df_TC_pre), 1))

    Xdf_train, Xdf_test, Ydf_train, Ydf_test = train_test_split(df_in_train, Y_onehot, test_size=0.1, shuffle=True)

    # shuffle the dataset and split for training and validation
    Xdf1_train = np.reshape(Xdf_train[:, :, 0], (len(Xdf_train[:, :, 0]), num_steps, 1))
    Xdf_TC_train = np.reshape(Xdf_train[:, :, 1], (len(Xdf_train[:, :, 0]), num_steps, 1))
    Xdf1_test = np.reshape(Xdf_test[:, :, 0], (len(Xdf_test[:, :, 0]), num_steps, 1))
    Xdf_TC_test = np.reshape(Xdf_test[:, :, 1], (len(Xdf_test[:, :, 1]), num_steps, 1))

    return result_bins, df_in_train, num_of_bins, df_raw, Xdf1_train, Xdf_TC_train, Xdf1_test, Xdf_TC_test, Ydf_test, \
           Ydf_train, df_X_pre, df_TC_pre, Xdf_train, my_last_time


def db_to_MLmodel(datafortrain, client_number, B, E, n, save_model=True):
    """Executes local training from the prepared data, on local sensors. Also saves the local models"""
    batch_size = B
    num_steps = datafortrain.LSTM_steps
    tf.reset_default_graph()
    keep_prob = tf.placeholder("float")

    x_ = tf.placeholder(tf.float32, [None, num_steps, 1], name='input_placeholder')
    x_TC_ = tf.placeholder(tf.float32, [None, num_steps, 1], name='input_placeholder')

    rnn_inputs = tf.unstack(x_, axis=1)
    rnn_inputs_TC = tf.unstack(x_TC_, axis=1)

    num_of_states = [50, 50]  # Use [50] for only one lstm layer.

    regularizer = layers.l1_regularizer(0.0002)

    with tf.variable_scope("PH", regularizer=regularizer):
        cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_of_states]  # LSTM layer

        stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        rnn_outputs, state = tf.nn.static_rnn(stacked_rnn_cell, rnn_inputs, dtype=tf.float32)

        output_st = rnn_outputs[-1]  # get the last state output

        output_reshape = tf.reshape(output_st, [-1, num_of_states[-1]])

        rnn_outputs_drop = tf.nn.dropout(output_reshape, keep_prob)

    with tf.variable_scope("TC", regularizer=regularizer):
        cells_TC = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_of_states]
        stacked_rnn_cell_TC = tf.nn.rnn_cell.MultiRNNCell(cells_TC)
        rnn_outputs_TC, state_TC = tf.nn.static_rnn(stacked_rnn_cell_TC, rnn_inputs_TC, dtype=tf.float32)
        output_st_TC = rnn_outputs_TC[-1]
        output_TC_reshape = tf.reshape(output_st_TC, [-1, num_of_states[-1]])
        rnn_TC_outputs_drop = tf.nn.dropout(output_TC_reshape, keep_prob)


    W = weight_variable([num_of_states[-1], datafortrain.num_of_bins])
    b = bias_variable([datafortrain.num_of_bins])
    W_TC = weight_variable([num_of_states[-1], datafortrain.num_of_bins])

    # combine the rnn results of m_type and TC
    y_predict = tf.nn.softmax(tf.matmul(rnn_outputs_drop, W) + tf.matmul(rnn_TC_outputs_drop, W_TC) + b)

    y_ = tf.placeholder("float", [None, datafortrain.num_of_bins])

    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y_predict)) #cross entropy
    # Defining loss
    reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    reg_constant = 1.0
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
    loss = cross_entropy + reg_constant * reg_losses
    train_step = tf.train.AdamOptimizer(n).minimize(loss)  # n=5e-4 (Learningrate??)
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_, 1))  # check matching
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    pred = tf.argmax(y_predict, 1)

    # check the percentage that the difference between the prediction and the actual is less than 1 category
    close_prediction = tf.less_equal(tf.abs(tf.argmax(y_predict, 1) - tf.argmax(y_, 1)), 1)
    near_accuracy = tf.reduce_mean(tf.cast(close_prediction, "float"))

    init = tf.global_variables_initializer()
    epoch = E
    db_line = {"sensor": datafortrain.mtype,
               "timestamp_sensor": datafortrain.mytime
               }  # Preparing the string to insert prediction result into db

    with tf.Session() as sess:

        data_train_size = datafortrain.X_train.shape[0]
        sess.run(init)
        saver = tf.train.Saver()
        best_test_acc = 0
        writing_meta = True
        client_number = str(client_number)
        if save_model and os.path.exists(
                'saved_model_iWater_model' + datafortrain.mtype + 'global' + '.meta'):  # check if we have saved the
            # previous ML model, if yes then load it as initial model

            new_saver = tf.train.import_meta_graph('saved_model_iWater_model' + datafortrain.mtype + 'global' + '.meta')

            new_saver.restore(sess, 'saved_model_iWater_model' + datafortrain.mtype + 'global')

            writing_meta = False
            init = tf.global_variables_initializer()
            print("loaded global model for client: " + client_number)
            train_var = tf.trainable_variables()

            # v = sess.run(train_var[1])
            # print(v)
            # v = sess.run(train_var[2])
            # print(v)
            # v = sess.run(train_var[3])
            # print(v)
            # v = sess.run(train_var[4])
            # print(v)
            # v = sess.run(train_var[5])
            # print(v)
            # v = sess.run(train_var[6])
            # print(v)  # will show you your variable.

        for ep in range(epoch + 1):
            p_loss = 0
            p_correct = 0
            batch = int(data_train_size / batch_size + 0.5)
            index = np.random.permutation(batch)
            for i in range(batch):
                train_index = index[i]
                xx = datafortrain.X_train[train_index * batch_size:(train_index + 1) * batch_size]
                xx_TC = datafortrain.TC_train[train_index * batch_size:(train_index + 1) * batch_size]
                yy = datafortrain.Y_train[train_index * batch_size:(train_index + 1) * batch_size]

                train_step.run(session=sess, feed_dict={x_: xx, x_TC_: xx_TC, y_: yy,
                                                        keep_prob: 0.8})
            train_accuracy = accuracy.eval(session=sess,
                                           feed_dict={x_: datafortrain.X_train, x_TC_: datafortrain.TC_train,
                                                      y_: datafortrain.Y_train, keep_prob: 1.0})
            train_near_accuracy = near_accuracy.eval(session=sess,
                                                     feed_dict={x_: datafortrain.X_train, x_TC_: datafortrain.TC_train,
                                                                y_: datafortrain.Y_train, keep_prob: 1.0})
            print("step %d, train_accuracy %g, near_accuracy %g" % (ep, train_accuracy, train_near_accuracy))

            if ep % 1 == 0:
                train_accuracy = accuracy.eval(session=sess,
                                               feed_dict={x_: datafortrain.X_test, x_TC_: datafortrain.TC_test,
                                                          y_: datafortrain.Y_test, keep_prob: 1.0})
                train_near_accuracy = near_accuracy.eval(session=sess,
                                                         feed_dict={x_: datafortrain.X_test,
                                                                    x_TC_: datafortrain.TC_test,
                                                                    y_: datafortrain.Y_test, keep_prob: 1.0})
                print("step %d, tests_accuracy %g, tests_near_accuracy %g" % (ep, train_accuracy, train_near_accuracy))
                if best_test_acc <= train_accuracy and save_model:
                    try:  # if accuracy improved, then save the model
                        best_test_acc = train_accuracy
                        saved_path = saver.save(sess, './saved_model_iWater_model' + datafortrain.mtype + client_number,
                                                write_meta_graph=writing_meta)
                        print('model saved in {}'.format(saved_path))
                    except:
                        print("saving error")
        # predict the probability of X in each bins at the next time slot
        # If there is saved model, then load that model
        if save_model:
            new_saver = tf.train.import_meta_graph(
                'saved_model_iWater_model' + datafortrain.mtype + client_number + '.meta')

            new_saver.restore(sess, 'saved_model_iWater_model' + datafortrain.mtype + client_number)
        train_accuracy = accuracy.eval(session=sess,
                                       feed_dict={x_: datafortrain.X_test, x_TC_: datafortrain.TC_test,
                                                  y_: datafortrain.Y_test, keep_prob: 1.0})
        train_near_accuracy = near_accuracy.eval(session=sess,
                                                 feed_dict={x_: datafortrain.X_test, x_TC_: datafortrain.TC_test,
                                                            y_: datafortrain.Y_test, keep_prob: 1.0})
        print("Final, tests_accuracy %g, tests_near_accuracy %g" % (train_accuracy, train_near_accuracy))
        prediction = y_predict.eval(session=sess,
                                    feed_dict={x_: datafortrain.X_for_predict, x_TC_: datafortrain.TC_for_predict,
                                               keep_prob: 1.0})
        print(prediction)
        db_line['test_accuracy'] = train_accuracy.item()
        for i in range(datafortrain.num_of_bins):
            db_line['bin_' + str(i)] = [datafortrain.bins[i], datafortrain.bins[i + 1]]
            db_line['prob_' + str(i)] = prediction[0, i].item()

    return db_line
