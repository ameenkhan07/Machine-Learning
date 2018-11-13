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
print(
    f'Training Data : {len(train_data)}, TRAINING LABELS : {train_labels.shape}')
print(
    f'Validation Data : {len(validation_data)}, Validation LABELS : {validation_labels.shape}')
print(f'Testing Data : {len(test_data)}, Testing LABELS : {test_labels.shape}')

# USPS Data
# USPSMat, USPSTar = get_USPS_data()
# print(f'USPS Data : {len(USPSMat)}, USPS LABELS : {USPSTar}')

# Implementation of Logistic Regression
print("-----------------Linear Regression (SGD)---------------------")
sr = SoftmaxRegression(
    train_data, train_tar, train_labels,
    validation_data, validation_tar, validation_labels,
    test_data, test_tar, test_labels)
loss_list, train_acc_list, val_acc_list, test_acc_list = sr.get_sgd_solution()
print('Training Accuracy : ', train_acc_list[-1])
print('Validation Accuracy : ', val_acc_list[-1])
print('Testing Accuracy : ', test_acc_list[-1])
