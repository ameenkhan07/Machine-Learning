# from process_dataset import (processData, encodeData, encodeLabel, decodeLabel)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
import matplotlib.pyplot as plt
# from utils import get_data, save_data

import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'


def save_plot(history, filename):
    """Generates and saves the plot to output file

    Parameters
    ----------
        history: keras callbacks.History object
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pd.DataFrame(history)
    df.plot(subplots=True, grid=True, figsize=(10, 15))
    plt.savefig(os.path.join(OUTPUT_DIR, filename))


class SequentialNeuralNetwork:

    def __init__(self, *args):

        self.raw_data, self.raw_target = args[0], args[1]
        self.training_data, self.training_target = args[2], args[3]
        self.testing_data, self.testing_target = args[4], args[5]
        self.validation_data, self.validation_target = args[6], args[7]
        # self.learning_rate = 0.01
        # self.epochs = 1000
        # self.architecture = {
        #     'first_layer' : (512,'relu'),
        #     'second_layer': (256, 'relu'),
        #     'output_layer' : (4, 'softmax'),
        #     'dropout' : 0.3
        # }
        # self.optimzer = 'adadelta'
        # self.loss = 'categorical_crossentropy'

    def get_model(self):
        """Defines and returns Sequential NN model

        Parameters
        ----------

        Returns
        -------
            model: keras Sequential object
        """

        # MODEL PARAMETERS
        # print(self.training_data.shape[0])
        input_size = self.training_data.shape[0]  # 2^10 > 1000
        first_dense_layer_nodes = 2048  # first hidden layer
        # second_dense_layer_nodes = 1024  # first hidden layer
        third_dense_layer_nodes = 2  # output layer, no of classes of classification problem
        drop_out = 0.3  # Dropout Rate, adjusting for overfitting

        model = Sequential()

        # Input Layer(First Layer):
        model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
        model.add(Dropout(drop_out))

        # Second Hidden Layer
        # model.add(Dense(second_dense_layer_nodes))
        # model.add(Activation('relu'))
        # model.add(Dropout(drop_out))

        # Output Layer
        model.add(Dense(third_dense_layer_nodes))
        model.add(Activation('softmax'))

        # Overview of the defined architecture
        model.summary()

        # Network Compilation Step : learning process is configured
        model.compile(
            optimizer='sgd',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        return model

    def run_model(self, model):
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
        num_epochs = 10000

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
        # dataset = get_data('training.csv')
        # processedData, processedLabel = processData(dataset)
        processedData = self.training_data.transpose()
        processedLabel = np_utils.to_categorical(
            np.array(self.training_target.transpose()), 2)

        # Training Phase
        data = processedData
        target = processedLabel
        history = model.fit(
            data,
            target,
            validation_split=validation_data_split,
            epochs=num_epochs,
            batch_size=model_batch_size,
            callbacks=[tensorboard_cb, earlystopping_cb])

        return history

    def test_model(self, model):
        """Tests the accuracy of the model and outputs the result data
        into a pdf

        Parameters
        ----------
            model: keras sequential.Sequential object
        """

        wrong = 0
        right = 0

        # testData = get_data('testing.csv')

        processedTestData = self.testing_data.transpose()
        processedTestLabel = self.testing_target.transpose()
        predictedTestLabel = []

        for i, j in zip(processedTestData, processedTestLabel):
            y = model.predict(
                np.array(i).reshape(-1, self.testing_data.shape[0]))
            # predictedTestLabel.append(decodeLabel(y.argmax()))

            if j.argmax() == y.argmax():
                right = right + 1
            else:
                wrong = wrong + 1

        print("Errors: " + str(wrong), " Correct :" + str(right))

        print("Testing Accuracy: " + str(right / (right + wrong) * 100))

        # output = {}
        # output["input"] = testData['input']
        # output["label"] = testData['label']
        # output["predicted_label"] = predictedTestLabel

        # save_data(output, 'output.csv', 'outputs')
