from process_data import *
from softmax_regression import *
from neural_network import *

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
USPSMat, USPSTar, USPSLabel = get_USPS_data()
print('\nUSPS DATA SUMMARY\n')
print(f'USPS Data : {USPSMat.shape}, USPS Label : {USPSTar.shape}')

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


# Update training dataset, appended validation dataset to it
# for benefit of following models
train_data = np.concatenate((train_data, validation_data), axis=0)
train_tar = np.concatenate((train_tar, validation_tar), axis=0)
train_labels = np.concatenate((train_labels, validation_labels), axis=0)

print("----------------------Neural Network------------------------")

# Implementation of Neural Network
nn = SequentialNeuralNetwork(train_data, train_tar, train_labels)
model = nn.get_model()
nn.run_model(model)
# Testing on MNIST data
mnist_accuracy, mnist_accuracy_loss = nn.test_model(
    model, test_data, test_labels)
print(f'MNIST Accuracy : {mnist_accuracy*100}')
# Testing on USPS data
usps_loss, usps_accuracy = nn.test_model(model, USPSMat, USPSLabel)
print(f'USPS Accuracy : {usps_accuracy*100}')
