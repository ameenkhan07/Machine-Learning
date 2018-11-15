from process_data import *
from softmax_regression import *
from neural_network import *
from random_forest import *
from svm import *
from utils import *

print('\n----------------------------------------------------\n')
print('UBITname      = ameenmoh')
print('Person Number = 50288968')
print('\n----------------------------------------------------\n')

# MNIST Data
mnist_data = get_MNIST_data()
train_data, train_tar, train_labels = mnist_data[0], mnist_data[1], mnist_data[2]
validation_data, validation_tar, validation_labels = mnist_data[3], mnist_data[4], mnist_data[5]
test_data, test_tar, test_labels = mnist_data[6], mnist_data[7], mnist_data[8]

print('\nMNIST DATA SUMMARY\n')
print(
    f'Training Data : {len(train_data)}, TRAINING LABELS : {train_labels.shape}')
print(
    f'Validation Data : {len(validation_data)}, Validation LABELS : {validation_labels.shape}')
print(f'Testing Data : {len(test_data)}, Testing LABELS : {test_labels.shape}')

# USPS Test Data
USPSMat, USPSTar, USPSLabel = get_USPS_data()
print('\nUSPS TEST DATA SUMMARY\n')
print(f'USPS Data : {USPSMat.shape}, USPS Label : {USPSTar.shape}')

print('\n----------------------------------------------------\n')

# Implementation of Logistic Regression
print("\n--------------Softmax Logistic Regression (SGD)-----------------\n")
sr = SoftmaxRegression(train_data, train_tar, train_labels,
                       validation_data, validation_tar, validation_labels)
loss_list, train_acc_list, val_acc_list = sr.get_sgd_solution(verbose=0)

# Plot Training and Validation Accuracy
# print('Training Accuracy : ', train_acc_list[-1])
# print('Validation Accuracy : ', val_acc_list[-1])

pred_test = sr.get_pred_data(test_data)
acc = get_confusion_matrix(
    pred_test, test_tar, 'Softmax Logistic Regression (MNIST)')
print(f'Logistic Regression Accuracy for MNIST: {acc}')

pred_test_usps = sr.get_pred_data(USPSMat)
acc_usps = get_confusion_matrix(
    pred_test_usps, USPSTar, 'Softmax Logistic Regression (USPS)')
print(f'Logistic Regression Accuracy for USPS : {acc_usps}')

# Append validation dataset to training dataset
train_data = np.concatenate((train_data, validation_data), axis=0)
train_tar = np.concatenate((train_tar, validation_tar), axis=0)
train_labels = np.concatenate((train_labels, validation_labels), axis=0)

print("\n------------------Deep Neural Network------------------------\n")

# Implementation of Neural Network
nn = SequentialNeuralNetwork(train_data, train_tar, train_labels)
model = nn.get_model()
nn.run_model(model)
pred_test = nn.get_predicted_data(model, test_data)
acc = get_confusion_matrix(pred_test, test_tar, 'Neural Network (MNIST)')
print(f'Neural Network Accuracy for MNIST: {acc}')

# Testing on USPS data
pred_test_usps = nn.get_predicted_data(model, USPSMat)
acc = get_confusion_matrix(pred_test_usps, USPSTar, 'Neural Network (USPS)')
print(f'Neural Network Accuracy for USPS: {acc}')

print("\n----------------------Random Forest------------------------\n")

# Implementation of Random Forest
rf = RandomForest(train_data, train_tar, train_labels)
classifier = rf.get_rf_classifier()
pred_test = rf.get_pred_data(classifier, test_data)
acc = get_confusion_matrix(pred_test, test_tar, 'Random Forest (MNIST)')
print(f'Random Forest Accuracy for MNIST: {acc}')

# Testing on USPS data
pred_test_usps = rf.get_pred_data(classifier, USPSMat)
acc = get_confusion_matrix(pred_test_usps, USPSTar, 'Random Forest (USPS)')
print(f'Random Forest Accuracy for USPS: {acc}')

print("\n------------------Support Vector Machine---------------------\n")

# Implementation of SVM
svc = SupportVectorClassifier(train_data, train_tar, train_labels)
classifier = svc.get_svc()
pred_test = svc.get_pred_data(classifier, test_data)
acc = get_confusion_matrix(
    pred_test, test_tar, 'Support Vector Machine (MNIST)')
print(f'Support Vector Maching Accuracy for MNIST: {acc}')

# Testing on USPS data
pred_test_usps = svc.get_pred_data(classifier, USPSMat)
acc = get_confusion_matrix(pred_test_usps, USPSTar,
                           'Support Vector Machine (USPS)')
print(f'Support Vector Maching Accuracy for USPS: {acc}')
