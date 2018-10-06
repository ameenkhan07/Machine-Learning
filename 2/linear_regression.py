import numpy as np
import math


def GenerateBigSigma(Data, MuMatrix, TrainingPercent, IsSynthetic):
    """
    """

    BigSigma = np.zeros((len(Data), len(Data)))
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT) * (TrainingPercent * 0.01))
    varVect = []
    for i in range(0, len(DataT[0])):
        vct = []
        for j in range(0, int(TrainingLen)):
            vct.append(Data[i][j])
        varVect.append(np.var(vct))

    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3, BigSigma)
    else:
        BigSigma = np.dot(200, BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma


def GetScalar(DataRow, MuRow, BigSigInv):
    """
    """

    R = np.subtract(DataRow, MuRow)
    T = np.dot(BigSigInv, np.transpose(R))
    L = np.dot(R, T)
    return L


def GetRadialBasisOut(DataRow, MuRow, BigSigInv):
    """
    """
    phi_x = math.exp(-0.5 * GetScalar(DataRow, MuRow, BigSigInv))
    return phi_x


def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent=80):
    """
    """
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT) * (TrainingPercent * 0.01))
    PHI = np.zeros((int(TrainingLen), len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for C in range(0, len(MuMatrix)):
        for R in range(0, int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI


def GetWeightsClosedForm(PHI, T, Lambda):
    """
    """
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0, len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T = np.transpose(PHI)
    PHI_SQR = np.dot(PHI_T, PHI)
    PHI_SQR_LI = np.add(Lambda_I, PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER = np.dot(PHI_SQR_INV, PHI_T)
    W = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W


def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent=80):
    """
    """
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT) * (TrainingPercent * 0.01))
    PHI = np.zeros((int(TrainingLen), len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for C in range(0, len(MuMatrix)):
        for R in range(0, int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI


def GetValTest(VAL_PHI, W):
    """
    """
    Y = np.dot(W, np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y


def GetErms(VAL_TEST_OUT, ValDataAct):
    """
    """
    sum = 0.0
    t = 0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range(0, len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]), 2)
        if (int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter += 1
    accuracy = (float((counter * 100)) / float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' + str(math.sqrt(sum / len(VAL_TEST_OUT))))
