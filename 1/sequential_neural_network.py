"""Module containing function for defining, running and testing
the model (sequential neural network)
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from utils import get_data, save_data

import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'


from process_dataset import (processData, encodeData, encodeLabel, decodeLabel)


def get_model():
    """Defines and returns Sequential NN model

    Parameters
    ----------

    Returns
    -------
        model: keras Sequential object
    """

    ## MODEL PARAMETERS
    
    input_size = 10  # 2^10 > 1000
    first_dense_layer_nodes = 512  # first hidden layer
    second_dense_layer_nodes = 256  # first hidden layer
    third_dense_layer_nodes = 4  # output layer, no of classes of classification problem
    drop_out = 0.3  # Dropout Rate, adjusting for overfitting

    # 'Model' is a data structure used by keras, a way to 
    # organize(stack) the layers.
    # Dense in Keras means fully connected layer.
    # Keras has 2 ways for defining an architecture of neural nodes,
    # 1. Sequential : linear stack of layers
    # 2. functional API : for completely arbitrary architecture
    # for our purpose, approach 1 suffices  
    model = Sequential()

    # Input Layer(First Layer): Dimensionality 900*10
    # 900 numbers (101-1000), 10 bits each

    # Hidden Layer:
    # Weights: 10 * 256 (10 bits, 256 units in the first layer)
    # Dimensions from output of first layer = 900 * 256
    # Note : hidden layer units shouldn't be too absurd (too large or too small)
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))

    # Activation : Functions sigmoid/relu for hidden layers
    # To bring non linearity
    # Why not sigmoid?
    model.add(Activation('relu'))

    # Why dropout?
    # Dropout is a technique where randomly selected neurons are ignored during training.
    # The effect is that the network becomes less sensitive to the specific weights of neurons.
    # This in turn results in a network that is capable of better generalization and is less likely to
    # overfit the training data.
    model.add(Dropout(drop_out))

    # Second Hidden Layer
    # 256 nodes, ReLU activated with same dropout
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
 
    # Output Layer
    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax?

    # Overview of the defined architecture
    model.summary()

    
    ## Network Compilation Step : learning process is configured
    
    # Loss function : feedback signal used for learning(updating) the weight tensors
    # categorical_crossentropy : generally used for many-class classification problem

    # Optimizer : algorithm to minimize loss function in the training phase.
    # determines how training proceeds

    model.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def run_model(model):
    """Executes the model

    Parameters
    ----------
        model: keras sequential.Sequential object

    Returns
    -------
        history: keras callbacks.History object
            dict object, containing data about everything that happened
            during training
            
    """

    # Parameters used in Training Phase

    # Epochs : each iteration over entire data
    num_epochs = 1000

    # Mini batch size, 
    # each epoch consists this many samples of data multiple times 
    # over to complete the entire dataset
    model_batch_size = 128

    # Validation split : data the model has never seen before, used after each 
    # epoch to validate predicted value calculated during the training phase
    # To tackle the problem of overfitting
    validation_data_split = 0.2

    # Tensorboard and Early Stopping : TF callbacks definition
    tb_batch_size = 32
    early_patience = 100
    tensorboard_cb = TensorBoard(
        log_dir='logs', batch_size=tb_batch_size, write_graph=True)
    earlystopping_cb = EarlyStopping(
        monitor='val_loss', verbose=1, patience=early_patience, mode='min')

    # Retrieve training dataset and respective labels
    dataset = get_data('training.csv')
    processedData, processedLabel = processData(dataset)

    # Training Phase
    
    history = model.fit(
        processedData,
        processedLabel,
        validation_split=validation_data_split,
        epochs=num_epochs,
        batch_size=model_batch_size,
        callbacks=[tensorboard_cb, earlystopping_cb])

    return history


def test_model(model):
    """Tests the accuracy of the model and outputs the result data
    into a pdf

    Parameters
    ----------
        model: keras sequential.Sequential object
    """
    wrong = 0
    right = 0

    testData = get_data('testing.csv')

    processedTestData = encodeData(testData['input'].values)
    processedTestLabel = encodeLabel(testData['label'].values)
    predictedTestLabel = []

    for i, j in zip(processedTestData, processedTestLabel):
        y = model.predict(np.array(i).reshape(-1, 10))
        predictedTestLabel.append(decodeLabel(y.argmax()))

        if j.argmax() == y.argmax():
            right = right + 1
        else:
            wrong = wrong + 1

    print("Errors: " + str(wrong), " Correct :" + str(right))

    print("Testing Accuracy: " + str(right / (right + wrong) * 100))

    output = {}
    output["input"] = testData['input']
    output["label"] = testData['label']
    output["predicted_label"] = predictedTestLabel

    save_data(output, 'output.csv', 'outputs')
