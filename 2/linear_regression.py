import numpy as np
import math


def GenerateBigSigma(Data, TrainingPercent):
    """Generate and returns the covariance Matrix for the data. 
    Parameters:
    -----------
        Data : Raw data of 4 features
        TrainingPercent: Percent of raw data which is for training purposes
    Returns:
    -------
        BigSigma: Covariance
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
    BigSigma = np.dot(200, BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma


def GetScalar(DataRow, MuRow, BigSigInv):
    """Utility function for calculating the Radial Basis Function
    """

    R = np.subtract(DataRow, MuRow)
    T = np.dot(BigSigInv, np.transpose(R))
    L = np.dot(R, T)
    return L


def GetRadialBasisOut(DataRow, MuRow, BigSigInv):
    """Returns Gaussian Radial Basis Function for
    """
    phi_x = math.exp(-0.5 * GetScalar(DataRow, MuRow, BigSigInv))
    return phi_x


def GetWeightsClosedForm(PHI, T, Lambda):
    """Returns Moore-Penrose pseudo-inverse of the matrix phi with
    least squared regularization
    Parameters:
    ----------
        PHI: PHI matrix for the 
        T: Target values of the training data
        Lambda: Regularization Value
    """
    # Create Lambda identity matrix for vector operation
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
    """Computes and returns the design matrix
    Parameters:
    -----------
        Data:
        MuMatrix:
        BigSigma:
        TrainingPercent:
    Returns:
    ---------
        PHI: Design Matrix
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


def GetValTest(PHI, W):
    """Computes and returns the linear regression function
    Parameters:
    -----------
        PHI : M Basis Functions
        W : weight vector
    """
    Y = np.dot(W, np.transpose(PHI))
    ##print ("Test Out Generated..")
    return Y


def GetErms(VAL_TEST_OUT, ValDataAct):
    """Computes and returns ERMS and accuracy values
    rms for the output data
    """
    _sum, _counter = 0.0, 0
    for i in range(0, len(VAL_TEST_OUT)):
        _sum = _sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]), 2)
        if (int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            _counter += 1
    accuracy = (float((_counter * 100)) / float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(_sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' + str(math.sqrt(sum / len(VAL_TEST_OUT))))
