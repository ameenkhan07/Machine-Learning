import numpy as np
import math

learningRate = 0.01
epochs = 1000


def GenerateBigSigma(Data):
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
    # print(len(DataT), ".........", DataT[0])
    TrainingLen = math.ceil(len(DataT))
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


def GetPhiMatrix(Data, MuMatrix, BigSigma):
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
    TrainingLen = math.ceil(len(DataT))
    PHI = np.zeros((int(TrainingLen), len(MuMatrix)))
    # print(np.diag(BigSigma))
    # BigSigma[np.diag_indices_from(BigSigma)] +=1
    # print(np.diag(BigSigma))

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
    return (str(accuracy) + ',' + str(math.sqrt(_sum / len(VAL_TEST_OUT))))


def get_sgd_solution(TRAINING_PHI, TEST_PHI, VAL_PHI, W_Now, TrainingData,
                     TrainingTarget, TestData, TestDataAct, ValData, ValDataAct):
    """Computed weights for x datapoints iteratively updating, and returns the Erms for training, testing and validation data, as well as the testing accuracy.
    """
    # Gradient Descent Solution for Linear Regression
    La = 2
    # learningRate = 0.01
    L_Erms_Val, L_Erms_TR, L_Erms_Test, L_Accuracy_Test, W_Mat = [], [], [], [], []

    for i in range(0, epochs):

        # print (f'---------Iteration: {i} M{M} LR {learningRate} L :{C_Lambda}--------------')
        Delta_E_D = -np.dot(
            (TrainingTarget[i] - np.dot(np.transpose(W_Now), TRAINING_PHI[i])),
            TRAINING_PHI[i])
        La_Delta_E_W = np.dot(La, W_Now)
        Delta_E = np.add(Delta_E_D, La_Delta_E_W)
        Delta_W = -np.dot(learningRate, Delta_E)
        W_T_Next = W_Now + Delta_W
        W_Now = W_T_Next

        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT = GetValTest(TRAINING_PHI, W_T_Next)
        Erms_TR = GetErms(TR_TEST_OUT, TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))

        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT = GetValTest(VAL_PHI, W_T_Next)
        Erms_Val = GetErms(VAL_TEST_OUT, ValDataAct)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))

        #-----------------TestingData Accuracy---------------------#
        TEST_OUT = GetValTest(TEST_PHI, W_T_Next)
        Erms_Test = GetErms(TEST_OUT, TestDataAct)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        L_Accuracy_Test.append(float(Erms_Test.split(',')[0]))

    return ([L_Erms_TR, L_Erms_Val, L_Erms_Test, L_Accuracy_Test])
