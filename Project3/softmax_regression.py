import numpy as np
import math


class SoftmaxRegression:

    def __init__(self, *args):
        self.train_data, self.train_tar, self.train_labels = args[0], args[1], args[2]
        self.validation_data, self.validation_tar, self.validation_labels = args[
            3], args[4], args[4]

    def _calculate_entropy_loss(self, X, w, y):
        """
        """
        loss = 0
        t = np.dot(X, w)
        for i in range(len(t)):
            loss += -1 * np.dot(y[i], np.log(t[i].T))
        return(float(loss)/len(t))

    def _softmax(self, X, theta):
        """
        """
        t = np.dot(X, theta)
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

    def add_ones(self, X):
        """Adding Bias term
        """
        return(np.hstack((np.zeros(shape=(X.shape[0], 1), dtype='float') + 1, X)))

    def _one_hot_encoding(self, t):
        """Reverse binary encoding to decimal
        """
        return(np.argmax(t, axis=1))

    def accuracy(self, y, t):
        """
        """
        count = sum([1 for i in range(len(y)) if y[i]==t[i]])
        return(float(count)/len(y))

    def output_accuracy():
        """
        """
        pass

    def get_sgd_solution(self):
        """
        """
        X, y = self.train_data, self.train_labels
        valid_dataset, valid_labels = self.validation_data, self.validation_labels
        raw_train_labels, raw_valid_labels = self.train_tar, self.validation_tar
        X = self.add_ones(X)  # Bias term

        epochs = 200
        batch_size = 20
        # Initialise random weights
        theta = np.random.rand(X.shape[1], len(y[0]))

        lmbda, alpha = 0.01, 0.01
        max_error = 0.1
        # loss = 10

        err_iteration, train_accuracy, validation_accuracy = [], [], []

        # SGD
        for iteration in range(epochs):
            start = 0
            for i in range(X.shape[0]//batch_size):
                out_probs = self._softmax(
                    X[start:start+batch_size], theta)
                grad = (1.0/batch_size) * \
                    np.dot(X[start:start+batch_size].T,
                           (out_probs - y[start:start+batch_size]))
                g0 = grad[0]
                grad += ((lmbda * theta) / batch_size)
                grad[0] = g0
                theta -= alpha * grad

                # calculate the magnitude of the gradient and check for convergence
                loss = self._calculate_entropy_loss(
                    X[start:start+batch_size], theta, y)
                start += batch_size

            err_iteration.append(loss)

            # ???
            pred_output_train = np.dot(X, theta)
            train_accuracy.append(
                self.accuracy(raw_train_labels, self._one_hot_encoding(pred_output_train)))
            # ???
            pred_output_valid = np.dot(self.add_ones(valid_dataset), theta)
            validation_accuracy.append(
                self.accuracy(raw_valid_labels, self._one_hot_encoding(pred_output_valid)))

            print(f' Iteration : {iteration}, LOSS : {loss}')
            # Early Stopping
            if np.abs(loss) < max_error or math.isnan(loss):
                break

        # Output Training Accuracy
        # Output Validation Accuracy
        # Output Testing Accuracy

        return(theta, err_iteration, train_accuracy, validation_accuracy)
