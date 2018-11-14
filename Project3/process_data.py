import pickle
import gzip
import numpy as np
from keras.utils import np_utils

import os
from PIL import Image

data = {
    'mnist': 'data/mnist.pkl.gz',
    'usps': 'data/USPSdata/Numerals'
}


def encode(l):
    """Encode labels from 1-10 to binary alternatives
    """
    return((np.arange(10) == l[:, None]).astype(np.float32))

def _encode_label(labels):
    return np_utils.to_categorical(np.array(labels),10)

def get_MNIST_data():
    """
    """
    f = gzip.open(data['mnist'], 'rb')
    train_data, validation_data, test_data = pickle.load(
        f, encoding='latin1')
    f.close()
    # Getting Datasets and Targets
    train_tar, validation_tar, test_tar = train_data[1], validation_data[1], test_data[1]
    train_data, validation_data, test_data = train_data[0], validation_data[0], test_data[0]
    # Encoded Targets to labels
    train_labels, validation_labels, test_labels = encode(
        train_tar), encode(validation_tar), encode(test_tar)
    return(train_data, train_tar, train_labels,
           validation_data, validation_tar, validation_labels,
           test_data, test_tar, test_labels)


def get_USPS_data():
    """
    """
    USPSMat, USPSTar = [], []
    savedImg, labels = [], []
    for j in range(0, 10):
        curFolderPath = data['usps'] + '/' + str(j)
        imgs = os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg, 'r')
                img = img.resize((28, 28))
                # NORMALIZE
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)

    USPSLabel = _encode_label(USPSTar)
    return(np.asarray(USPSMat), np.asarray(USPSTar), np.asarray(USPSLabel))
