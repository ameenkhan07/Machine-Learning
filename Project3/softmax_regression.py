import numpy as np
import math


class SoftmaxRegression:

    def __init__(self, *args, **kwargs):
        self.train_data, self.train_tar, self.train_labels = self._add_bias(
            args[0]), args[1], args[2]
        self.val_data, self.val_tar, self.val_labels = self._add_bias(
            args[3]), args[4], args[5]
        self.lmbda = 0.01
        self.learning_rate = 0.01
        self.minibatch_size = 20
        self.weight = []

    def _add_bias(self, data):
        """Adding Bias term to dataset
        """
        return(np.hstack((np.zeros(shape=(data.shape[0], 1), dtype='float') + 1, data)))

    def _one_hot_encoding(self, t):
        """Converting one hot vector to corresponding digit.
        """
        return(np.argmax(t, axis=1))

    def get_pred_data(self, data, weight=[]):
        """
        """
        # Get model weight if none passes in params
        if not len(weight):
            weight = self.weight
            data = self._add_bias(data)
        return(self._one_hot_encoding(np.dot(data, weight)))

    def _get_accuracy(self, y, t):
        count = sum([1 for i in range(len(y)) if y[i] == t[i]])
        return(float(count)/len(y))

    def _get_cross_entropy_loss(self, X, w, y):
        """Cross Entropy Cost function for optimising our weights
        """
        loss = 0
        t = np.dot(X, w)
        for i in range(len(t)):
            loss += -1 * np.dot(y[i], np.log(t[i].T))
        return(float(loss)/len(t))

    def _softmax(self, X, weight):
        """N-class softmax function used in logistic regression
        Returns class probabilites.
        """
        activation = np.dot(X, weight)
        p_matrix = []
        # arranging weight vectors as matrix for probability
        for i in range(len(activation)):
            exp_summation = sum([np.exp(j) for j in activation[i]])
            _list = [(float(np.exp(k))/exp_summation) for k in activation[i]]
            p_matrix.append(_list)
        return(np.array(p_matrix))

    def get_sgd_solution(self, verbose=0):
        """Return Training, Validation and Testing accuracy using softmax regression
        """
        X, t = self.train_data, self.train_labels
        epochs, mb_size = 200, self.minibatch_size
        early_stopping_error = 0.1  # Used

        # Initialise random weights
        weight = np.random.rand(X.shape[1], len(t[0]))

        loss_list, train_acc_list, val_acc_list,  = [], [], []

        # Gradient Descent
        for itr in range(epochs):

            # Minibatch weight updation
            mb_pos = 0
            for i in range(X.shape[0]//mb_size):
                mb_next = mb_pos+mb_size

                # Probability matrix for the given batch
                y = self._softmax(X[mb_pos:mb_next], weight)

                # Gradient of cross entropy error
                gradient = np.dot(X[mb_pos:mb_next].T,
                                  (y - t[mb_pos:mb_next]))/mb_size

                # Regularization, avoid regularizing bias term
                bias_term = gradient[0]
                gradient += ((self.lmbda / mb_size) * weight)
                gradient[0] = bias_term

                # Weight update with learning rate
                weight -= self.learning_rate * gradient

                # Cross entropy loss of mini-batch of data
                loss = self._get_cross_entropy_loss(X[mb_pos:mb_next],
                                                    weight, t)
                # Update
                mb_pos = mb_next

            # Track cross entropy loss of this iteration
            loss_list.append(loss)

            # Training Accuracy
            train_pred = self.get_pred_data(X, weight)
            train_acc_list.append(
                self._get_accuracy(self.train_tar, train_pred))
            # Validation Accuracy
            val_pred = self.get_pred_data(self.val_data, weight)
            val_acc_list.append(
                self._get_accuracy(self.val_tar, val_pred))

            if verbose:
                print(f'-------Iteration : {itr}, LOSS : {loss}---------')

            # Early Stopping
            if np.abs(loss) < early_stopping_error or math.isnan(loss):
                break

        # Save the weights after model has been trained
        self.weight = weight

        return(loss_list, train_acc_list, val_acc_list)
