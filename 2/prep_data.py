import numpy as np
import csv
import math


def GetTargetVector(filePath):
    """Returns target values, which is a vector
    """
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t


def GenerateRawData(filePath):
    """Returns datafrom the input file.
    """
    dataMatrix = []
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)
    # Remove unwanted columns
    dataMatrix = np.delete(dataMatrix, [5, 6, 7, 8, 9], axis=1)
    dataMatrix = np.transpose(dataMatrix)
    #print ("Data Matrix Generated..")
    return dataMatrix


def GenerateTrainingTarget(RawTarget, TrainingPercent=80):
    """Returns Targets for training for the given training percent.  
    """
    TrainingLen = int(math.ceil(len(RawTarget)*(TrainingPercent*0.01)))
    t = RawTarget[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t


def GenerateTrainingDataMatrix(rawData, TrainingPercent=80):
    """Returns feature values for the given training percent
    """
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:, 0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2


def GenerateValData(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:, TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")
    return dataMatrix


def GenerateValTargetVector(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t = rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t
