import numpy as np
import math


class SoftmaxRegression:

    def __init__(self, *args, **kwargs):
        self.train_data, self.train_tar, self.train_labels = self._add_bias(
            args[0]), args[1], args[2]
        self.val_data, self.val_tar, self.val_labels = self._add_bias(
            args[3]), args[4], args[5]
        self.test_data, self.test_tar, self.test_labels = self._add_bias(
            args[6]), args[7], args[8]
        self.lmbda = 0.01
        self.alpha = 0.01
        self.minibatch_size = 20

    def _add_bias(self, data):
        """Adding Bias term to dataset
        """
        return(np.hstack((np.zeros(shape=(data.shape[0], 1), dtype='float') + 1, data)))

    def _one_hot_encoding(self, t):
        """Converting one hot vector to corresponding digit.
        """
        return(np.argmax(t, axis=1))

    def _get_entropy_loss(self, X, w, y):
        """Cross Entropy Cost function for optimising our model weights
        """
        loss = 0
        t = np.dot(X, w)
        for i in range(len(t)):
            loss += -1 * np.dot(y[i], np.log(t[i].T))
        return(float(loss)/len(t))

    def _softmax(self, X, weight):
        """
        """
        t = np.dot(X, weight)
        prob_matrix = []
        for i in range(len(t)):
            prob_vector = []
            sum_exp = 0
            for j in t[i]:
                sum_exp += np.exp(j)
            for k in t[i]:
                prob_vector.append(float(np.exp(k))/sum_exp)
            prob_matrix.append(prob_vector)
        return(np.array(prob_matrix))

    def _get_accuracy(self, y, t):
        """
        """
        count = sum([1 for i in range(len(y)) if y[i] == t[i]])
        return(float(count)/len(y))

    def get_sgd_solution(self):
        """Return Training, Validation and Testing accuracy using softmax regression
        """
        X, y = self.train_data, self.train_labels
        epochs, mb_size = 200, self.minibatch_size
        max_error = 0.1

        # Initialise random weights
        weight = np.random.rand(X.shape[1], len(y[0]))

        loss_list = []
        train_acc_list, val_acc_list, test_acc_list = [], [], []
        pred_test_list = []

        # Minibatch Stochastic Descent
        for iteration in range(epochs):

            mb_pos = 0
            for i in range(X.shape[0]//mb_size):
                mb_next = mb_pos+mb_size

                # Get Gradient
                out_probs = self._softmax(
                    X[mb_pos:mb_next], weight)

                grad = (1.0/mb_size) * np.dot(X[mb_pos:mb_next].T,
                                              (out_probs - y[mb_pos:mb_next]))
                g0 = grad[0]
                grad += ((self.lmbda * weight) / mb_size)
                grad[0] = g0

                weight -= self.alpha * grad

                # calculate the magnitude of the gradient and check for convergence
                loss = self._get_entropy_loss(
                    X[mb_pos:mb_next], weight, y)

                mb_pos = mb_next

            # Track loss of this iteration
            loss_list.append(loss)

            # Training Accuracy
            pred_train = self._one_hot_encoding(np.dot(X, weight))
            train_acc_list.append(
                self._get_accuracy(self.train_tar, pred_train))
            # Validation Accuracy
            pred_val = self._one_hot_encoding(
                np.dot(self.val_data, weight))
            val_acc_list.append(
                self._get_accuracy(self.val_tar, pred_val))

            # Testing Accuracy
            pred_test = self._one_hot_encoding(
                np.dot(self.test_data, weight))
            test_acc_list.append(
                self._get_accuracy(self.test_tar, pred_test))
            pred_test_list.append(pred_test)

            print(f'-------Iteration : {iteration}, LOSS : {loss}---------')

            # Early Stopping exit
            if np.abs(loss) < max_error or math.isnan(loss):
                break

        return(loss_list, train_acc_list, val_acc_list, test_acc_list, pred_test_list)
