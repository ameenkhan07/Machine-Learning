from process_data import *
from softmax_regression import *
from neural_network import *
from random_forest import *
from svm import *
from utils import *
from collections import Counter


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

print("\n-----------------Combining Models (Majority Vote)-----------------------\n")
# Create 4 subsample datasets and targets from the original training data
# print(len(train_data), len(train_tar), len(train_labels))
tr_data_samples, tr_tar_samples, tr_label_samples = [], [], []
for i in range(4):
    indexes = np.random.randint(0, train_data.shape[0], 40000)
    tr_data_samples.append(train_data[indexes])
    tr_tar_samples.append([train_tar[i] for i in indexes])
    tr_label_samples.append(train_labels[indexes])

# Train models using the the training_samples

# 1. Softmax Logistic Regression
val_data, val_tar, val_lab = tr_data_samples[0][30001:
                                                ], tr_tar_samples[0][30001:], tr_label_samples[0][30001:]
tr_data, tr_tar, tr_lab = tr_data_samples[0][:
                                             30000], tr_tar_samples[0][:30000], tr_label_samples[0][:30000]
sr_sample = SoftmaxRegression(tr_data, tr_tar, tr_lab,
                              val_data, val_tar, val_lab)
loss_l, train_acc_l, val_acc_l = sr_sample.get_sgd_solution(verbose=0)

# 2. Deep Neural Network
nn_sample = SequentialNeuralNetwork(
    tr_data_samples[1], tr_tar_samples[1], tr_label_samples[1])
nn_model = nn_sample.get_model()
nn_sample.run_model(nn_model)

# 3. Random Forest
rf_sample = RandomForest(
    tr_data_samples[2], tr_tar_samples[2], tr_label_samples[2])
rf_sample_classifier = rf_sample.get_rf_classifier()

# 4. Support Vector Machine
svc_sample = SupportVectorClassifier(
    tr_data_samples[3], tr_tar_samples[3], tr_label_samples[3])
svc_sample_classifier = svc_sample.get_svc()

# Test the above models with MNIST data
sr_mnist_sample_pred = sr_sample.get_pred_data(test_data)
nn_mnist_sample_pred = nn_sample.get_predicted_data(nn_model, test_data)
rf_mnist_sample_pred = rf_sample.get_pred_data(
    rf_sample_classifier, test_data)
svc_mnist_sample_pred = svc_sample.get_pred_data(
    svc_sample_classifier, test_data)

print('\n\n----------------')
# print(len(sr_mnist_sample_pred), len(
#     nn_mnist_sample_pred), len(rf_mnist_sample_pred))

res_pred_mnist = []
for i, j, k, l in zip(sr_mnist_sample_pred, nn_mnist_sample_pred,
                      rf_mnist_sample_pred, svc_mnist_sample_pred):
    # print(i, j, k)
    _l = [i, j, k, l]
    res_pred_mnist.append(max(_l, key=_l.count))

# print(len(res_pred_mnist), res_pred_mnist[1:10])
res_mnist_acc = get_confusion_matrix(res_pred_mnist, test_tar,
                                     'Combined Model (MNIST)')
print(f'Combined Model Accuracy for MNIST: {res_mnist_acc}')


# Test the above models with USPS data
sr_usps_sample_pred = sr_sample.get_pred_data(USPSMat)
nn_usps_sample_pred = nn_sample.get_predicted_data(nn_model, USPSMat)
rf_usps_sample_pred = rf_sample.get_pred_data(
    rf_sample_classifier, USPSMat)
svc_usps_sample_pred = svc_sample.get_pred_data(
    svc_sample_classifier, USPSMat)

print('\n\n----------------')
# print(len(sr_usps_sample_pred), len(
#     nn_usps_sample_pred), len(rf_usps_sample_pred))

res_pred_usps = []
for i, j, k, l in zip(sr_usps_sample_pred, nn_usps_sample_pred,
                   rf_usps_sample_pred, svc_usps_sample_pred):
    # print(i, j, k)
    _l = [i, j, k, l]
    res_pred_usps.append(max(_l, key=_l.count))

# print(len(res_pred_usps), res_pred_usps[1:10])
res_usps_acc = get_confusion_matrix(res_pred_usps, USPSTar,
                                    'Combined Model (USPS)')
print(f'Combined Model Accuracy for USPS: {res_usps_acc}')
