from process_data import *
from softmax_regression import *

print('----------------------------------------------------')
print('UBITname      = ameenmoh')
print('Person Number = 50288968')
print('----------------------------------------------------')

# MNIST Data
train_data, train_tar, train_labels, validation_data, validation_tar, validation_labels, test_data, test_tar, test_labels = get_MNIST_data()
# get USPS data, for testing the mode
print('\nMNIST DATA SUMMARY\n')
print(f'Training Data : {len(train_data)}, TRAINING LABELS : {train_labels.shape}')
print(f'Validation Data : {len(validation_data)}, Validation LABELS : {validation_labels.shape}')
print(f'Testing Data : {len(test_data)}, Testing LABELS : {test_labels.shape}')

# USPS Data
# USPSMat, USPSTar = get_USPS_data()
# print(f'USPS Data : {len(USPSMat)}, USPS LABELS : {USPSTar}')



# Implementation of Logistic Regression
print("-----------------Linear Regression (SGD)---------------------")
# weights_lr, err_iteration_lr, train_accuracy_l54321`, validation_accuracy_lr = train_log_regression(
#     train_data, train_labels, validation_data, validation_labels, train_tar, validation_tar)
sr = SoftmaxRegression(
    train_data, train_tar, train_labels,
    validation_data, validation_tar, validation_labels)

weights_sr, err_iteration_sr, train_accuracy_sr, validation_accuracy_sr = sr.get_sgd_solution()

# pred_output_train_lr = np.dot(add_ones(training_data), weights_lr)
# print("Training Set Accuracy - Logistic Regression: ", sr.accuracy(training_tar, one_hot_encoding(pred_output_train_lr)))
# pred_output_valid_lr = np.dot(add_ones(validation_data), weights_lr)
# print("Validation Set Accuracy - Logistic Regression: ", sr.accuracy(validation_tar, sr.one_hot_encoding(pred_output_valid_lr)))
# pred_output_test_lr = np.dot(add_ones(test_data), weights_lr)
# print("Test Set Accuracy - Logistic Regression: ", sr.accuracy(test_tar, one_hot_encoding(pred_output_test_lr)))
