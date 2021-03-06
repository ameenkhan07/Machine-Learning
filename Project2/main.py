import math
from preprocess import *
from linear_regression import *
from logistic_regression import *
from sequential_neural_network import *
import numpy as np

# Linear Regression Variables
TrainingPercent = 80  # Data Split for Training Data
ValidationPercent = 10  # Data Split for Validation Data
TestPercent = 10  # Data Split for Testing Data

print('UBITname      = ameenmoh')
print('Person Number = 50288968')
print('----------------------------------------------------')

# Preprocess Data
# raw_data, raw_target = get_data_features('hod', operation='concat')
# print('HOD CONCATENATED DATASET')
# raw_data, raw_target = get_data_features('hod', operation='subtract')
# print('HOD SUBTRACTED DATASET')
raw_data, raw_target = get_data_features('gsc', operation='concat')
print('GSC CONCATENATED DATASET')
# raw_data, raw_target = get_data_features('gsc', operation='subtract')
# print('GSC SUBTRACTED DATASET')
print('----------------------------------------------------')

# Data split into training/validation/testing
training_ratio = math.floor(raw_data.shape[0]*.8)
val_test_ratio = math.floor(raw_data.shape[0]*.9)

training_data = raw_data[:training_ratio].transpose()
training_target = raw_target[:training_ratio]
validation_data = raw_data[training_ratio:val_test_ratio].transpose()
validation_target = raw_target[training_ratio:val_test_ratio]
testing_data = raw_data[val_test_ratio:].transpose()
testing_target = raw_target[val_test_ratio:]
raw_data = raw_data.transpose()

print('DATA SPLIT : ', training_data.shape,
      testing_data.shape, validation_data.shape)
print('TARGET SPLIT : ', training_target.shape,
      testing_target.shape, validation_target.shape)

# Hyperparameter Initialization
M = 10
C_Lambda = 0.03
learningRate = 0.1

# Initialise Linear Regression
linear_regression = LinearRegression(
    raw_data, raw_target,
    training_data, training_target,
    testing_data, testing_target,
    validation_data, validation_target,
    M, C_Lambda, learningRate
)
# Execute Linear Regression
L_Erms_TR, L_Erms_Val, L_Erms_Test, L_Accuracy_Test = linear_regression.get_sgd_solution()
print("-----------------Linear Regression (SGD)---------------------")
print(f"M = {M} \nLambda  = {C_Lambda}\neta={learningRate}")
print("E_rms Training   = " + str(np.around(min(L_Erms_TR), 5)))
print("E_rms Validation = " + str(np.around(min(L_Erms_Val), 5)))
print("E_rms Testing    = " + str(np.around(min(L_Erms_Test), 5)))
print("Testing Accuracy = " + str(np.around(min(L_Erms_Test), 5)))

# Initialise Logistic Regression
logistic_regression = LogisticRegression(
    raw_data, raw_target,
    training_data, training_target,
    testing_data, testing_target,
    learningRate
)
# Execute Logistic Regression
print("-----------------Logistic Regression (SGD)---------------------")
print(f"\neta={learningRate}")
accuracy = logistic_regression.get_sgd_solution()
print(f"Accuracy : {accuracy}")


# Initialise Seuquential Neural Network
nn = SequentialNeuralNetwork(
    raw_data, raw_target,
    training_data, training_target,
    testing_data, testing_target
)
# Execute Neural Network model
print("-----------------Neural Netowrk (SGD)---------------------")
model = nn.get_model()
history = nn.run_model(model)
save_plot(history.history, 'keras.png')
# Test Accuracy
accuracy = nn.test_model(model)
print("Testing Accuracy: " + accuracy)
