import numpy as np
from math import e


class LogisticRegression:

    def __init__(self, *args):
        self.raw_data, self.raw_target = args[0], args[1]
        self.training_data, self.training_target = args[2], args[3]
        self.testing_data, self.testing_target = args[4], args[5]
        self.learning_rate = args[6]
        self.learning_rate = 0.01
        self.la = 2
        self.W = np.random.rand(self.training_data.shape[0])

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _gradient(self, weight, X, Y):
        """Returns cost
        """
        # Reshape arrays
        weight, X, Y = np.matrix(weight), np.matrix(X), np.matrix(Y)

        feature_len = int(weight.ravel().shape[1])  # Number of features
        res = np.zeros(feature_len)

        error = self._sigmoid(X * weight.T) - Y.T
        for i in range(0, feature_len):
            r = np.multiply(error, X[:, i])
            res[i] = np.sum(r)/len(X)
        # temp =  e**(np.dot(X,weight))/(1.+e**(np.dot(X,weight)))
        return res

    def get_sgd_solution(self):
        """Return prediction accuracy for logistic regression
        """
        W_Now = self.W
        for i in range(0, len(self.training_data[0])):
            delta_W = self._gradient(
                W_Now, self.training_data.T, self.training_target)
            La_delta_W = np.dot(self.la, W_Now)  # Delta va
            delta_W_Reg = np.add(delta_W, La_delta_W)
            W_Next = W_Now - (np.dot(self.learning_rate, delta_W_Reg))
            W_Now = W_Next
            # print(i, 'New updates :', W_Now)

        theta = np.matmul(self.testing_data.T, W_Now)
        hypothesis = np.rint(self._sigmoid(theta))
        diff = np.subtract(hypothesis, self.testing_target)
        non_zeros = np.count_nonzero(diff)
        accuracy = ((len(diff) - non_zeros)/len(diff))*100
        # print(accuracy)

        return(accuracy)
