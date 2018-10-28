# import numpy as np
import math
from sklearn.cluster import KMeans
from utils import *
from linear_regression import *

# Linear Regression Variables
TrainingPercent = 80  # Data Split for Training Data
ValidationPercent = 10  # Data Split for Validation Data
TestPercent = 10  # Data Split for Testing Data

M = 10
C_Lambda = 0.03
learningRate = 0.01

# Prep Data
# raw_data, raw_target = get_data_features('hod', operation='concat')
# print('HOD CONCAT : ', raw_data.shape, raw_target.shape)
# raw_data, raw_target = get_data_features('hod', operation='subtract')
# print('HOD SUBTRACT : ', raw_data.shape, raw_target.shape)
# raw_data, raw_target = get_data_features('gsc', operation='concat')
# print('GSC CONCAT : ', raw_data.shape, raw_target.shape)
raw_data, raw_target = get_data_features('gsc', operation='subtract')
print('GSC SUBTRACT : ', raw_data.shape, raw_target.shape)

# Data split into training/validation/testing
training_ratio = math.floor(raw_data.shape[0]*.8)
val_test_ratio = math.floor(raw_data.shape[0]*.9)
training_data, training_target = raw_data[:
                                          training_ratio], raw_target[:training_ratio]
validation_data, validation_target = raw_data[training_ratio:
                                              val_test_ratio], raw_target[training_ratio:val_test_ratio]
testing_data, testing_target = raw_data[val_test_ratio:], raw_target[val_test_ratio:]
raw_data, training_data, testing_data, validation_data = raw_data.transpose(
), training_data.transpose(), testing_data.transpose(), validation_data.transpose()
print('DATA SPLIT : ', training_data.shape,
      testing_data.shape, validation_data.shape)
print('TARGET SPLIT : ', training_target.shape,
      testing_target.shape, validation_target.shape)

# Initialise Linear Regression
M = 10
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(training_data))
Mu = kmeans.cluster_centers_
BigSigma = GenerateBigSigma(raw_data)
print('MU : ', Mu.shape, ' SIGMA MATRIX : ', BigSigma.shape)

W = np.array([0]*M)
print(W.shape)

TRAINING_PHI = GetPhiMatrix(training_data, Mu, BigSigma)
TEST_PHI = GetPhiMatrix(testing_data, Mu, BigSigma)
VAL_PHI = GetPhiMatrix(validation_data, Mu, BigSigma)
print('TRAINING PHI Shape: ', TRAINING_PHI.shape)
print('TEST PHI Shape: ', TEST_PHI.shape)
print('VALIDATION PHI Shape: ', VAL_PHI.shape)


L_Erms_TR, L_Erms_Val, L_Erms_Test, L_Accuracy_Test = get_sgd_solution(
    TRAINING_PHI, TEST_PHI, VAL_PHI, W, training_data, training_target,
    testing_data, testing_target, validation_data, validation_target)

print(f"M = {M} \nLambda  = {C_Lambda}\neta={learningRate}")
print("E_rms Training   = " + str(np.around(min(L_Erms_TR), 5)))
print("E_rms Validation = " + str(np.around(min(L_Erms_Val), 5)))
print("E_rms Testing    = " + str(np.around(min(L_Erms_Test), 5)))
print("Testing Accuracy = " + str(np.around(min(L_Erms_Test), 5)))