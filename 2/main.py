from sklearn.cluster import KMeans
import numpy as np

from prep_data import *
from linear_regression import *

maxAcc = 0.0
maxIter = 0
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10
PHI = []
IsSynthetic = False

if __name__ == "__main__":

    # Fetch and Prepare Dataset
    RawTarget = GetTargetVector('./data/Querylevelnorm_t.csv')
    RawData   = GenerateRawData('./data/Querylevelnorm_X.csv',IsSynthetic)

    # Prepare Training Data
    TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
    TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
    print(TrainingTarget.shape)
    print(TrainingData.shape)

    # Prepare Validation Data
    ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
    ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
    print(ValDataAct.shape)
    print(ValData.shape)

    # Prepare Vaildation Data
    TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
    print(ValDataAct.shape)
    print(ValData.shape)

    # Prepare Test Data
    TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
    print(ValDataAct.shape)
    print(ValData.shape)


    ErmsArr = []
    AccuracyArr = []

    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
    Mu = kmeans.cluster_centers_

    BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
    W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
    TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
    VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


    print(Mu.shape)
    print(BigSigma.shape)
    print(TRAINING_PHI.shape)
    print(W.shape)
    print(VAL_PHI.shape)
    print(TEST_PHI.shape)


    TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
    VAL_TEST_OUT = GetValTest(VAL_PHI,W)
    TEST_OUT     = GetValTest(TEST_PHI,W)

    TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
    ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
    TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


    print ('UBITname      = XXXXXXXX')
    print ('Person Number = YYYYYYYY')
    print ('----------------------------------------------------')
    print ("------------------LeToR Data------------------------")
    print ('----------------------------------------------------')
    print ("-------Closed Form with Radial Basis Function-------")
    print ('----------------------------------------------------')
    print ("M = 10 \nLambda = 0.9")
    print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
    print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
    print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))

    print ('----------------------------------------------------')
    print ('--------------Please Wait for 2 mins!----------------')
    print ('----------------------------------------------------')


    W_Now        = np.dot(220, W)
    La           = 2
    learningRate = 0.01
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []

    for i in range(0,400):
        
        #print ('---------Iteration: ' + str(i) + '--------------')
        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next
        
        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        
        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
        Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
        
        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
        Erms_Test = GetErms(TEST_OUT,TestDataAct)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))

    print ('----------Gradient Descent Solution--------------------')
    print ("M = 15 \nLambda  = 0.0001\neta=0.01")
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))