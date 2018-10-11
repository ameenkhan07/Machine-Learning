"""Module containing functions to process the data before and after
feeding into neural network
"""
import numpy as np
from keras.utils import np_utils

import os

# TF_CPP_MIN_LOG_LEVEL : Tensorflow environment variable
# Default: to 0 (all logs shown), 
# 1 to filter out INFO logs, 
# 2 to additionally filter out WARNING logs, 
# 3 to additionally filter out ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def processData(dataset):
    """Processes input data and label.
    
    Parameters
    ----------
        dataset : pandas.Dataframe
            Dataframe containing data and labels

    Returns
    -------
        processedData : numpy.ndarray
        processedLabel :  numpy.ndarray
    """

    # Why do we have to process?
    data = dataset['input'].values
    labels = dataset['label'].values

    processedData = encodeData(data)
    processedLabel = encodeLabel(labels)

    return processedData, processedLabel


def encodeData(data):
    """Returns data converted to its binary reporesentation
    in order to be able to fed into neural network as tensors.
    
    Parameters
    ----------
        data : numpy.ndarray

    Returns
    -------
        processedData : numpy.ndarray
    """
    processedData = []

    for dataInstance in data:
        # Why do we have number 10?
        processedData.append([dataInstance >> d & 1 for d in range(10)])

    return np.array(processedData)


def encodeLabel(labels):
    """Encodes string labels to int labels for benefit of NN

    Parameters
    ----------
        labels : numpy.ndarray

    Returns
    -------
        processedLabel : numpy.ndarray
    """

    processedLabel = []

    for labelInstance in labels:
        if (labelInstance == "fizzbuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif (labelInstance == "fizz"):
            # Fizz
            processedLabel.append([1])
        elif (labelInstance == "buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel), 4)


def decodeLabel(encodedLabel):
    """Decodes integer label back to equivalent string label.

    Parameters
    ----------
        encodedLabel : int

    Returns
    -------
        label : str
    """
    if encodedLabel == 0:
        return "other"
    elif encodedLabel == 1:
        return "fizz"
    elif encodedLabel == 2:
        return "buzz"
    elif encodedLabel == 3:
        return "fizzbuzz"
