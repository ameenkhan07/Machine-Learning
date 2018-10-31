import math
from preprocess import *
from linear_regression import *
from logistic_regression import *

# Linear Regression Variables
TrainingPercent = 80  # Data Split for Training Data
ValidationPercent = 10  # Data Split for Validation Data
TestPercent = 10  # Data Split for Testing Data

M = 10
C_Lambda = 0.03
learningRate = 0.01

# Preprocess Data
# raw_data, raw_target = get_data_features('hod', operation='concat')
# print('HOD CONCAT : ', raw_data.shape, raw_target.shape)
raw_data, raw_target = get_data_features('hod', operation='subtract')
# print('HOD SUBTRACT : ', raw_data.shape, raw_target.shape)
# raw_data, raw_target = get_data_features('gsc', operation='concat')
# print('GSC CONCAT : ', raw_data.shape, raw_target.shape)
# raw_data, raw_target = get_data_features('gsc', operation='subtract')
# print('GSC SUBTRACT : ', raw_data.shape, raw_target.shape)

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

# Initialise Linear Regression
linear_regression = LinearRegression(
    raw_data, raw_target,
    training_data, training_target,
    testing_data, testing_target,
    validation_data, validation_target
)

# Execute Linear Regression
L_Erms_TR, L_Erms_Val, L_Erms_Test, L_Accuracy_Test = linear_regression.get_sgd_solution()
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
    validation_data, validation_target
)
# Execute Linear Regression
accuracy = logistic_regression.get_sgd_solution()
print(f"Accuracy : {accuracy}")

