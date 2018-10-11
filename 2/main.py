from sklearn.cluster import KMeans
import numpy as np

from prep_data import *
from linear_regression import *


<<<<<<< Updated upstream
# Constants used throughout the prog
learningRate = 0.1
C_Lambda = 0.3
=======
# Hyperparameters
learningRate = 0.01
C_Lambda =  0.1
M = 20 # Number of Radial Basis Func

# Data Split
>>>>>>> Stashed changes
TrainingPercent = 80 # Data Split for Training Data
ValidationPercent = 10 # Data Split for Validation Data
TestPercent = 10 # Data Split for Testing Data



def get_closed_form_solution(TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingData,
                             TrainingTarget, TestData, ValData):
    """Computes weights and returns the Erms for training, testing and validation data, as well as the testing accuracy.
    """
    TR_TEST_OUT = GetValTest(TRAINING_PHI, W)
    VAL_TEST_OUT = GetValTest(VAL_PHI, W)
    TEST_OUT = GetValTest(TEST_PHI, W)

    TrainingAccuracy = str(GetErms(TR_TEST_OUT, TrainingTarget))
    ValidationAccuracy = str(GetErms(VAL_TEST_OUT, ValDataAct))
    TestAccuracy = str(GetErms(TEST_OUT, TestDataAct))

    return ([TrainingAccuracy, ValidationAccuracy, TestAccuracy])


def get_sgd_solution(TRAINING_PHI, TEST_PHI, VAL_PHI, W_Now, TrainingData,
                     TrainingTarget, TestData, ValData):
    """Computed weights for x datapoints iteratively updating, and returns the Erms for training, testing and validation data, as well as the testing accuracy.
    """
    # Gradient Descent Solution for Linear Regression
    La = 2
    # learningRate = 0.01
    L_Erms_Val, L_Erms_TR, L_Erms_Test, L_Accuracy_Test, W_Mat = [], [], [], [], []

    for i in range(0, 400):

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


if __name__ == "__main__":

    # Fetch and Prepare Dataset
    RawTarget = GetTargetVector('./data/Querylevelnorm_t.csv')
    RawData = GenerateRawData('./data/Querylevelnorm_X.csv')

    # Prepare Training Data
    TrainingTarget = np.array(
        GenerateTrainingTarget(RawTarget, TrainingPercent))
    TrainingData = GenerateTrainingDataMatrix(RawData, TrainingPercent)
    # print('Training Target : ', TrainingTarget.shape)
    # print('Training Data : ', TrainingData.shape)

    # Prepare Validation Data
    ValDataAct = np.array(
        GenerateValTargetVector(RawTarget, ValidationPercent,
                                (len(TrainingTarget))))
    ValData = GenerateValData(RawData, ValidationPercent,
                              (len(TrainingTarget)))
    # print(ValDataAct.shape)
    # print(ValData.shape)

    # Prepare Test Data
    TestDataAct = np.array(
        GenerateValTargetVector(RawTarget, TestPercent,
                                (len(TrainingTarget) + len(ValDataAct))))
    TestData = GenerateValData(RawData, TestPercent,
                               (len(TrainingTarget) + len(ValDataAct)))
    # print(TestDataAct.shape)
    # print(TestData.shape)

    print('UBITname      = ameenmoh')
    print('Person Number = 50288968')
    print('----------------------------------------------------')
    print("------------------LeToR Data------------------------")
    print('----------------------------------------------------')

    # KMeans to get centroids, Mu for radial basis 
    kmeans = KMeans(
        n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
    Mu = kmeans.cluster_centers_

    # Get the covariance matrix
    BigSigma = GenerateBigSigma(RawData, TrainingPercent)

    # Initialise Radial Basis Function for 
    # Training/Testing/Validation data
    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
    TEST_PHI = GetPhiMatrix(TestData, Mu, BigSigma, 100)
    VAL_PHI = GetPhiMatrix(ValData, Mu, BigSigma, 100)

    # Closed Form Solutuion
    W = GetWeightsClosedForm(TRAINING_PHI, TrainingTarget, (C_Lambda))
    W_Now = np.dot(220, W)

    print('Mu Shape : ', Mu.shape)
    print('Big Sigma Shape : ',BigSigma.shape)
    print('W Shape : ',W.shape)
    print('PHI matrix (Training) Shape : ',TRAINING_PHI.shape)
    print('PHI matrix (Validation) Shape : ',VAL_PHI.shape)
    print('PHI matrix (Testing) Shape : ',TEST_PHI.shape)

    print("-------Closed Form with Radial Basis Function-------")
    print('----------------------------------------------------')

    # TrainingAccuracy, ValidationAccuracy, TestAccuracy = get_closed_form_solution(
    #     TRAINING_PHI, TEST_PHI, VAL_PHI, TrainingData, TrainingTarget,
    #     TestData, ValData)

<<<<<<< Updated upstream
    print(f"M = {M} \nLambda = {C_Lambda}")
    print("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
    print("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
    print("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
    print("Accuracy Testing    = " + str(float(TestAccuracy.split(',')[0])))
=======
    # print(f"M = {M} \nLambda = {C_Lambda}")
    # print("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
    # print("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
    # print("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
    # print("Accuracy Testing    = " + str(float(TestAccuracy.split(',')[0])))
>>>>>>> Stashed changes

    print("------------------SGD Solution----------------------")
    print('----------------------------------------------------')
    print('')
    print('----------------------------------------------------')
    print('-------------Please Wait for 2 mins!----------------')
    print('----------------------------------------------------')
    # learningRate = 5

    L_Erms_TR, L_Erms_Val, L_Erms_Test, L_Accuracy_Test = get_sgd_solution(
        TRAINING_PHI, TEST_PHI, VAL_PHI, W_Now, TrainingData, TrainingTarget,
        TestData, ValData)

    print('----------Gradient Descent Solution--------------------')
    print(f"M = {M} \nLambda  = {C_Lambda}\neta={learningRate}")
    print("E_rms Training   = " + str(np.around(min(L_Erms_TR), 5)))
    print("E_rms Validation = " + str(np.around(min(L_Erms_Val), 5)))
    print("E_rms Testing    = " + str(np.around(min(L_Erms_Test), 5)))
    print("Testing Accuracy = " + str(np.around(min(L_Erms_Test), 5)))
