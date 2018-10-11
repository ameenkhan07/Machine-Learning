"""Module containing functions for creating training and testing data"""

from utils import save_data
from FizzBuzz import FizzBuzz

def createInputCSV(start, end, filename):
    """Creates data for a range of numbers and inserts it into a file.

    Parameters
    ----------
        start : int
            Starting index for the data to be created

        end : int
            Last position for the data to be created

        filename : string
            Name of the file to store data into

    """

    # Why list in Python?
    inputData = []
    outputData = []

    # Why do we need training Data?
    for i in range(start, end):
        inputData.append(i)
        outputData.append(FizzBuzz().func(i))

    # Why Dataframe?
    # Dataframes are pandas objects, which themselves are enhanced version of 
    # numpy structured arrays
    # DataFrames are inherently multidimensional arrays with 
    # attached row and column labels, and added support for heterogeneous 
    # types and missing data.
    dataset = {}
    dataset["input"] = inputData
    dataset["label"] = outputData

    save_data(dataset, filename)

    print(filename, "Created!")