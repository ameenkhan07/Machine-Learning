"""Module containing function for defining, running and testing
the model (sequential neural network)
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import pandas as pd

from process_dataset import (
    processData, 
    encodeData, 
    encodeLabel, 
    decodeLabel
)

# MODEL PARAMETERS
input_size = 10  # 2^10 >1000
drop_out = 0.2  # adjusting for overfitting
first_dense_layer_nodes = 256  # first trainng layer, (hidden layer), hyperparameter
# if hidden layer shouldnt be too absurt, not to large (overfitting)
# not to small
second_dense_layer_nodes = 4  # output layer


def get_model():
    """
    """

    # Why do we need a model?
    # Why use Dense layer and then activation?
    # Why use sequential model with layers?
    model = Sequential()
    # input matrix dimensionality 900*10
    # reason : 900 input number, 10 bits of each numbers

    # weight of first hidden layer = 10*256
    # dimentionality of output will be 900*256

    # output matrix to activation function : sigmoid, relu etc
    # relu is also used in regularization
    # why not sigmoid?
    # variants fo relu, max(x, 0)

    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))

    # Why dropout?
    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax?

    model.summary()

    # SGD? optimizers
    #     Optimizers:
    # adagrad, adam etc.

    # Loss function
    # Cross Entropy: KL Divergence
    # Why use categorical_crossentropy?
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def run_model(model):
    """
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
    dataset = pd.read_csv('training.csv')

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
    """
    """
    wrong = 0
    right = 0

    TESTING_DATA = "./data/testing.csv"
    OUTPUT_DATA = "./data/output.csv"

    testData = pd.read_csv(TESTING_DATA)

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

    opdf = pd.DataFrame(output)
    opdf.to_csv(OUTPUT_DATA)
