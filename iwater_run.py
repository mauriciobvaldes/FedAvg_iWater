# -*- coding: utf-8 -*-

"""This is the code run from the server, which calls the iwater_learning_prediction code for local training"""
import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import os
from pymongo import MongoClient
from datetime import datetime
from datetime import timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from iwater_learning_prediction import Datafortrain
from iwater_learning_prediction import db_to_MLmodel
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import layers
import matplotlib.pyplot as plt


def scale_model_weights(weight, scalar):
    """Function for scaling a models weights for federated averaging"""

    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(w_final):
    """Return the sum of the listed scaled weights"""

    counter = 0
    for scaled_client_weights in w_final:

        if counter == 0:
            avg_grad = scaled_client_weights
            counter += 1
        else:
            avg_grad += scaled_client_weights
    return np.array(avg_grad)


def create_global_model(num_clients, length_of_data_list, length_of_total_data, m_type):
    """Aggregates all the weights and biases from saved files of the local models"""
    #TODO: Fix this function, a lot of things are still hard coded which shouldn't be.
    scale_list = []  # Creating placeholder lists for scaling, as well as weights and biases.
    w_LSTM_list = []  # These will be filled later.
    b_LSTM_list = []
    w_TC_LSTM_list = []
    b_TC_LSTM_list = []
    w_LSTM_list1 = []
    b_LSTM_list1 = []
    w_TC_LSTM_list1 = []
    b_TC_LSTM_list1 = []
    w_dense_list = []
    b_dense_list = []
    w_TC_dense_list = []

    for i in range(num_clients):
        # saver = tf.train.import_meta_graph('saved_model_iWater_modelRX_C' + str(i) + '.meta')
        # delete the current graph
        tf.reset_default_graph()

        # import the graph from the file
        imported_graph = tf.train.import_meta_graph('saved_model_iWater_model' + m_type + str(i) + '.meta')
        # run the session
        with tf.Session() as sess:
            # restore the saved vairable
            imported_graph.restore(sess, 'saved_model_iWater_model' + m_type + str(i))

            # list_of_variables = tf.trainable_variables()
            # print(list_of_variables)

            w_LSTM, b_LSTM, w_TC_LSTM, b_TC_LSTM, w_LSTM1, b_LSTM1, w_TC_LSTM1, b_TC_LSTM1, w_dense, b_dense, w_TC_dense = sess.run(
                ['PH/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0', 'PH/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',

                 'TC/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0', 'TC/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',

                 'PH/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0', 'PH/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0',

                 'TC/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0', 'TC/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0',

                 'Variable:0', 'Variable_1:0', 'Variable_2:0'])  # Getting all variables from the loaded model.
            w_LSTM_list.append(w_LSTM)
            b_LSTM_list.append(b_LSTM)
            w_TC_LSTM_list.append(w_TC_LSTM)
            b_TC_LSTM_list.append(b_TC_LSTM)
            w_LSTM_list1.append(w_LSTM1)
            b_LSTM_list1.append(b_LSTM1)
            w_TC_LSTM_list1.append(w_TC_LSTM1)
            b_TC_LSTM_list1.append(b_TC_LSTM1)
            w_dense_list.append(w_dense)
            b_dense_list.append(b_dense)
            w_TC_dense_list.append(w_TC_dense)  # Creating lists for holding all variables for all clients.

        scale = length_of_data_list[i] / length_of_total_data
        scale_list.append(scale)  # Creating a list for scale factors for all clients.

    w_LSTM_final = []  # Initializing the final lists to be used.
    b_LSTM_final = []
    w_TC_LSTM_final = []
    b_TC_LSTM_final = []
    w_LSTM_final1 = []
    b_LSTM_final1 = []
    w_TC_LSTM_final1 = []
    b_TC_LSTM_final1 = []
    w_dense_final = []
    b_dense_final = []
    w_TC_dense_final = []

    for i in range(num_clients):
        #  Creating the final scaled variable lists.
        w_LSTM_final.append(scale_model_weights(np.array(w_LSTM_list[i]), scale_list[i]))
        b_LSTM_final.append(scale_model_weights(np.array(b_LSTM_list[i]), scale_list[i]))
        w_TC_LSTM_final.append(scale_model_weights(np.array(w_TC_LSTM_list[i]), scale_list[i]))
        b_TC_LSTM_final.append(scale_model_weights(np.array(b_TC_LSTM_list[i]), scale_list[i]))

        w_LSTM_final1.append(scale_model_weights(np.array(w_LSTM_list1[i]), scale_list[i]))
        b_LSTM_final1.append(scale_model_weights(np.array(b_LSTM_list1[i]), scale_list[i]))
        w_TC_LSTM_final1.append(scale_model_weights(np.array(w_TC_LSTM_list1[i]), scale_list[i]))
        b_TC_LSTM_final1.append(scale_model_weights(np.array(b_TC_LSTM_list1[i]), scale_list[i]))

        w_dense_final.append(scale_model_weights(np.array(w_dense_list[i]), scale_list[i]))
        w_TC_dense_final.append(scale_model_weights(np.array(w_TC_dense_list[i]), scale_list[i]))
        b_dense_final.append(scale_model_weights(np.array(b_dense_list[i]), scale_list[i]))

    # Scaled weights: First input are the scaled weights from the first client, second input are the scaled weights
    # from the second client and so on...
    w_LSTM_final = np.array(w_LSTM_final)  # Converting to np arrays for better summation.
    b_LSTM_final = np.array(b_LSTM_final)
    w_TC_LSTM_final = np.array(w_TC_LSTM_final)
    b_TC_LSTM_final = np.array(b_TC_LSTM_final)

    w_LSTM_final1 = np.array(w_LSTM_final1)
    b_LSTM_final1 = np.array(b_LSTM_final1)
    w_TC_LSTM_final1 = np.array(w_TC_LSTM_final1)
    b_TC_LSTM_final1 = np.array(b_TC_LSTM_final1)

    w_dense_final = np.array(w_dense_final)
    w_TC_dense_final = np.array(w_TC_dense_final)
    b_dense_final = np.array(b_dense_final)

    # Sum all the weights
    w_LSTM_final, b_LSTM_final, w_TC_LSTM_final, b_TC_LSTM_final = sum_scaled_weights(w_LSTM_final), sum_scaled_weights(
        b_LSTM_final), sum_scaled_weights(w_TC_LSTM_final), sum_scaled_weights(b_TC_LSTM_final)

    w_LSTM_final1, b_LSTM_final1, w_TC_LSTM_final1, b_TC_LSTM_final1 = sum_scaled_weights(
        w_LSTM_final1), sum_scaled_weights(
        b_LSTM_final1), sum_scaled_weights(w_TC_LSTM_final1), sum_scaled_weights(b_TC_LSTM_final1)

    w_dense_final, w_TC_dense_final, b_dense_final = sum_scaled_weights(w_dense_final), sum_scaled_weights(
        w_TC_dense_final), sum_scaled_weights(b_dense_final)

    return w_dense_final, w_TC_dense_final, b_dense_final, w_LSTM_final, b_LSTM_final, w_TC_LSTM_final, b_TC_LSTM_final, w_LSTM_final1, b_LSTM_final1, w_TC_LSTM_final1, b_TC_LSTM_final1


def save_global_model(w_dense_final, w_TC_dense_final, b_dense_final, w_LSTM_final, b_LSTM_final, w_TC_LSTM_final,
                      b_TC_LSTM_final, w_LSTM_final1, b_LSTM_final1, w_TC_LSTM_final1, b_TC_LSTM_final1, m_type, B, n):
    """Saves global model meta-file"""

    tf.reset_default_graph()

    saver = tf.train.import_meta_graph(
        'saved_model_iWater_model' + m_type + '0' + '.meta')  # Restores arbitrary graph from local client to be used
    # as a template, it has the correct "form".

    op1 = tf.trainable_variables()[0].assign(w_LSTM_final)  # Creating the operations to be called in session.
    op2 = tf.trainable_variables()[1].assign(b_LSTM_final)  # These will only be executed in the session in this
    # version of tensorflow. (No eager execution).
    op3 = tf.trainable_variables()[4].assign(w_TC_LSTM_final)  # Assigns the FedAvg values to the model.
    op4 = tf.trainable_variables()[5].assign(b_TC_LSTM_final)
    op5 = tf.trainable_variables()[2].assign(w_LSTM_final1)
    op6 = tf.trainable_variables()[3].assign(b_LSTM_final1)
    op7 = tf.trainable_variables()[6].assign(w_TC_LSTM_final1)
    op8 = tf.trainable_variables()[7].assign(b_TC_LSTM_final1)
    op9 = tf.trainable_variables()[8].assign(w_dense_final)
    op10 = tf.trainable_variables()[9].assign(b_dense_final)
    op11 = tf.trainable_variables()[10].assign(w_TC_dense_final)

    init = tf.global_variables_initializer()

    # Replaces weights and biases with the global weights and biases

    with tf.Session() as sess:
        sess.run(init)
        sess.run(op1)
        sess.run(op2)
        sess.run(op3)
        sess.run(op4)
        sess.run(op5)
        sess.run(op6)
        sess.run(op7)
        sess.run(op8)
        sess.run(op9)
        sess.run(op10)
        sess.run(op11)

        """v = sess.run(tf.trainable_variables()[4])
        print(v)"""
        try:
            saved_path = saver.save(sess, './saved_model_iWater_model' + m_type + 'global',
                                    write_meta_graph=True)  # Saves global model as meta-file
            print('model saved in {}'.format(saved_path))
        except:
            print("saving error")
    pass


def get_data(m_type):
    """Getting data from mongodb server. Needs password so here on github you will get data from file instead. Saves
    as csv files """
    my_measure_type = m_type
    history_length_in_days = 365
    client = MongoClient("mongodb://username:password")
    mydb = client["iWater"]
    # mycollection = mydb["test"]
    mycollection = mydb["iWater_node_01"]
    time_now = datetime.now()
    d = timedelta(days=history_length_in_days)
    time_sliding_window_start = time_now - d  # check at most d days int db
    # UPDATE_RONG
    time_change_date = datetime(2019, 12, 18)  # This is the date when the sensor was deployed in the lake
    time_sliding_window_start = max(time_sliding_window_start, time_change_date)
    # ENDUPDATE_RONG
    cursor_mytype = mycollection.find(
        {'$and': [{'sensor': my_measure_type}, {'timestamp_sensor': {'$gt': time_sliding_window_start}}]})
    cursor_TC1_D = mycollection.find(
        {'$and': [{'sensor': 'TC1_D'}, {'timestamp_sensor': {'$gt': time_sliding_window_start}}]})
    # cursor_mytype=mycollection.find()
    # print(datetime.strftime(time_sliding_window_start,"%Y-%m-%d %H:%M:%S"))
    df_mytype_withtime = pd.DataFrame(list(cursor_mytype))[['value', 'timestamp_sensor']]
    df_mytype_withtime.to_csv(r'C:\Users\youruser\yourfolder\X_data.csv')  # Change to your specific folder
    df_TC1_D_withtime = pd.DataFrame(list(cursor_TC1_D))[['value', 'timestamp_sensor']]
    df_TC1_D_withtime.to_csv(r'C:\Users\youruser\yourfolder\TC1_D_data.csv')  # Change to your specific folder
    pass


def FederatedAveraging(B, E, R, n, K, m_type, name_of_data_file, data, data2):
    """Executing the FedAvg algorithm, running local training and calling helpfunctions to create global models.
    Tweak these parameters as you wish when calling the function"""
    # B: Local batch-size
    # E: Local # of epochs + 1
    # R: Number of global rounds
    # n: Learning rate
    # K: # of clients
    # m_type: 'RX_C' or other.

    num_clients = K
    length_of_data_list = []
    length_of_total_data = 0
    global_accuracy = 0

    client_number_list = []
    client_data_list = []

    data_for_plot_x = [0]
    data_for_plot_y = [0]

    for client_number in range(num_clients):
        tf.reset_default_graph()
        client_data = Datafortrain(m_type, num_clients, client_number, data, data2)  # Preparing data
        db_line = db_to_MLmodel(client_data, client_number, B, E, n)  # Feed data to ML model
        length_of_data = len(client_data.X_train)
        length_of_data_list.append(length_of_data)  # This is used to create the scale-factors (n_k/n) in FedAvg
        length_of_total_data += length_of_data
        client_number_list.append(client_number)
        client_data_list.append(client_data)
        global_accuracy += db_line['test_accuracy'] / num_clients

    print(global_accuracy, 'round' + str(1))
    data_for_plot_x.append(1)
    data_for_plot_y.append(global_accuracy)

    w_dense_final, w_TC_dense_final, b_dense_final, w_LSTM_final, b_LSTM_final, w_TC_LSTM_final, b_TC_LSTM_final, w_LSTM_final1, b_LSTM_final1, w_TC_LSTM_final1, b_TC_LSTM_final1 = create_global_model(
        num_clients, length_of_data_list, length_of_total_data, m_type)  # Scales and sums weights according to FedAvg

    save_global_model(w_dense_final, w_TC_dense_final, b_dense_final, w_LSTM_final, b_LSTM_final, w_TC_LSTM_final,
                      b_TC_LSTM_final, w_LSTM_final1, b_LSTM_final1, w_TC_LSTM_final1, b_TC_LSTM_final1, m_type, B,
                      n)  # Saves new global model as meta-file

    for i in range(2, R + 1):

        print('round number: ', i)

        global_accuracy = 0

        # decay_steps = R
        # decay_rate = 0.9
        # drop = 0.7
        # epochs_drop = 2
        # decayed_n = n * drop**(np.floor((i)/epochs_drop)) #step decay
        # decayed_n = n * decay_rate**(i/decay_steps) #exponential decay as implemented in keras
        # decayed_n = n * (1. / (1. + (i / R))) #time based learning decay, maybe try step based or exponential
        # print('decayed_n: ', decayed_n)

        for client_number in client_number_list:
            db_line = db_to_MLmodel(client_data_list[client_number], client_number, B, E, n)
            global_accuracy += db_line[
                                   'test_accuracy'] / num_clients

        print(global_accuracy, 'round' + str(i))

        data_for_plot_x.append(i)
        data_for_plot_y.append(global_accuracy)

        w_dense_final, w_TC_dense_final, b_dense_final, w_LSTM_final, b_LSTM_final, w_TC_LSTM_final, b_TC_LSTM_final, w_LSTM_final1, b_LSTM_final1, w_TC_LSTM_final1, b_TC_LSTM_final1 = create_global_model(
            num_clients, length_of_data_list, length_of_total_data,
            m_type)  # Scales and sums weights according to FedAvg

        save_global_model(w_dense_final, w_TC_dense_final, b_dense_final, w_LSTM_final, b_LSTM_final, w_TC_LSTM_final,
                          b_TC_LSTM_final, w_LSTM_final1, b_LSTM_final1, w_TC_LSTM_final1, b_TC_LSTM_final1, m_type, B,
                          n)  # Saves new global model as meta-file

    data_for_plot = np.array([data_for_plot_x, data_for_plot_y])
    np.save(name_of_data_file, data_for_plot)  # Saves numpy array used for plotting global accuracy


m_type = 'TC1_D'  # Choose type of measurement here. Without access to the database you will only have TC1_D available
# get_data(m_type)  # Only if you have password to the database

m_type_data = pd.read_csv(r'C:\Users\youruser\yourfolder\X_data.csv')  # Change to your folder, reads data from file
TC_data = pd.read_csv(r'C:\Users\youruser\yourfolder\TC1_D_data.csv')  # Change to your folder, reads data from file

FederatedAveraging(6, 0, 50, 0.0005, 5, m_type, m_type + '2lstm_B6_E1_R50_n0005_K5', m_type_data, TC_data)  # Runs the
# program, choose parameters here.

# measure_type=['TC1_D','SA_A','OS_D', 'TC2_C','PH_C','RX_C','TC3_A','CN_A',]
