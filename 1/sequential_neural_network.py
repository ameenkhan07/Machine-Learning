"""Module containing function for defining, running and testing
the model (sequential neural network)
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from utils import get_data, save_data

from process_dataset import (processData, encodeData, encodeLabel, decodeLabel)

# MODEL PARAMETERS
input_size = 10  # 2^10 > 1000
drop_out = 0.2  # adjusting for overfitting

first_dense_layer_nodes = 256  # first hidden layer
# Note : hidden layer shouldn't be too absurd,
# not to large (overfitting), not to small
second_dense_layer_nodes = 4  # output layer


def get_model():
    """Returns Sequential NN model

    Parameters
    ----------

    Returns
    -------
        model: keras Sequential object
    """

    # Why do we need a model?
    # 'Model' is a data structure used by keras, a way to
    # organize the layers.

    # Why use Dense layer and then activation?
    # Dense layer means fully connected layer.

    # Why use sequential model with layers?
    model = Sequential()

    # First Hidden Layer
    # Input dimensionality 900*10
    # reason : 900 numbers (101-1000), 10 bits each
    # Weight of first hidden layer = 10 * 256 (10 bits, 256 units in the first layer)
    # Dimensions from first layer = 900 * 256
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))

    #  # Activation Layer
    # Activation : Fucntions sigmoid/ relu
    # Why Relu :relu is also used in regularization
    # Why not sigmoid?
    model.add(Activation('relu'))

    # Why dropout?
    # Dropout is a technique where randomly selected neurons are ignored during training.
    # The effect is that the network becomes less sensitive to the specific weights of neurons.
    # This in turn results in a network that is capable of better generalization and is less likely to
    # overfit the training data.
    model.add(Dropout(drop_out))

    # Second Layer, Output Layer
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax?

    model.summary()

    # Experiment with optimizers:
    # SGD, adagrad, adam etc.

    # Loss function
    # Cross Entropy: KL Divergence
    # Why use categorical_crossentropy?
    model.compile(
        optimizer='rmsprop',
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
    """
    validation_data_split = 0.2  # trying to tackle the problem of overfitting
    # difference between validation to testing
    # after every epoch we validate
    # validation loss and training loss
    # validation avoids over fitting

    # cross validation: validation data keeps rotating

    num_epochs = 10000  # experiment on this
    model_batch_size = 128
    # minibatch? : between the 2 approaches:
    # fitting whole data into the model,

    tb_batch_size = 32
    early_patience = 100

    tensorboard_cb = TensorBoard(
        log_dir='logs', batch_size=tb_batch_size, write_graph=True)
    earlystopping_cb = EarlyStopping(
        monitor='val_loss', verbose=1, patience=early_patience, mode='min')

    # Read Dataset
    dataset = get_data('training.csv')

    # Process Dataset
    processedData, processedLabel = processData(dataset)
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
