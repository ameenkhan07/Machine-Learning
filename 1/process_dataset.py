"""Module containing functions to process the data before and after
feeding into neural network
"""

from keras.utils import np_utils
import numpy as np
import os

# Silence TF Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def processData(dataset):
    """Processes input data and label.
    
    Parameters
    ---------
        dataset : pandas dataframe
            Dataframe containing data and labels

    Returns
    -------
        processedData : list
        processedLabel :  list
    """

    # Why do we have to process?
    data = dataset['input'].values
    labels = dataset['label'].values

    processedData = encodeData(data)
    processedLabel = encodeLabel(labels)

    return processedData, processedLabel


def encodeData(data):
    """Returns data converted to its binary reporesentaiton
    
    Parameters
    ----------
        data : list[int]

    Returns
    -------
        processedData : numpy array
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
        data : list[str]

    Returns
    -------
        processedLabel : numpy array
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
