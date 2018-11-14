import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adam


class SequentialNeuralNetwork:

    def __init__(self, *args, **kwargs):
        self.train_data, self.train_tar, self.train_labels = args[0], args[1], args[2]
        self.epochs = 200
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'

    def get_model(self):
        """
        """

        drop_out = 0.2
        first_dense_layer_nodes = 64
        third_dense_layer_nodes = 10

        model = Sequential()

        model.add(Dense(first_dense_layer_nodes,
                        input_dim=self.train_data.shape[1]))
        model.add(Activation('sigmoid'))
        model.add(Dropout(drop_out))

        model.add(Dense(third_dense_layer_nodes))
        model.add(Activation('softmax'))

        model.summary()

        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=['accuracy'])

        return model

    def run_model(self, model):
        """
        """
        # NN Execution params
        validation_data_split = 0.2
        model_batch_size = 128
        tb_batch_size = 64
        early_patience = 100

        tensorboard_cb = TensorBoard(
            log_dir='logs', batch_size=tb_batch_size, write_graph=True)
        earlystopping_cb = EarlyStopping(
            monitor='val_loss', verbose=1, patience=early_patience)

        # print(self.train_data.shape, self.train_labels.shape)
        history = model.fit(self.train_data, self.train_labels,
                            validation_split=validation_data_split,
                            epochs=self.epochs, batch_size=model_batch_size,
                            callbacks=[tensorboard_cb, earlystopping_cb]
                            )

        return(history)

    def test_model(self, model, test_data, test_label):
        """Computes the loss based on the input you pass it
        """
        return(model.evaluate(test_data, test_label, batch_size=128, verbose=1))
