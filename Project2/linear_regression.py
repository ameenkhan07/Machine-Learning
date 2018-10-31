import numpy as np
import math
from sklearn.cluster import KMeans

epochs = 1000


class LinearRegression:

    def __init__(self, *args):
        self.raw_data, self.raw_target = args[0], args[1]
        self.training_data, self.training_target = args[2], args[3]
        self.testing_data, self.testing_target = args[4], args[5]
        self.validation_data, self.validation_target = args[6], args[7]
        self.M = 10
        self.learning_rate = 0.01
        self.Mu = self.get_mu()
        self.BigSigma = self.GenerateBigSigma(self.raw_data)
        self.W = np.array([0]*self.M)
        self.TRAINING_PHI = self.GetPhiMatrix(self.training_data)
        self.TEST_PHI = self.GetPhiMatrix(self.testing_data)
        self.VAL_PHI = self.GetPhiMatrix(self.validation_data)

    def get_mu(self):
        """Return Mu Matrix based on kmeans clustering algo
        """
        kmeans = KMeans(n_clusters=self.M, random_state=0).fit(
            np.transpose(self.training_data))
        Mu = kmeans.cluster_centers_
        # print('MU : ', Mu.shape, ' SIGMA MATRIX : ', BigSigma.shape)
        return(Mu)

    def GenerateBigSigma(self, Data):
        """Generate and returns the covariance Matrix for the data. 
        Parameters:
        -----------
            Data : Raw data of 4 features
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

    def GetScalar(self, DataRow, MuRow, BigSigInv):
        """Utility function for calculating the Radial Basis Function
        """
        R = np.subtract(DataRow, MuRow)
        T = np.dot(BigSigInv, np.transpose(R))
        L = np.dot(R, T)
        return L

    def GetRadialBasisOut(self, DataRow, MuRow, BigSigInv):
        """Returns Gaussian Radial Basis Function for
        """
        phi_x = math.exp(-0.5 * self.GetScalar(DataRow, MuRow, BigSigInv))
        return phi_x

    def GetPhiMatrix(self, Data):
        """Computes and returns the design matrix
        Parameters:
        -----------
            Data:
        Returns:
        ---------
            PHI: Design Matrix
        """
        DataT = np.transpose(Data)
        TrainingLen = math.ceil(len(DataT))
        PHI = np.zeros((int(TrainingLen), len(self.Mu)))

        BigSigInv = np.linalg.inv(self.BigSigma)
        for C in range(0, len(self.Mu)):
            for R in range(0, int(TrainingLen)):
                PHI[R][C] = self.GetRadialBasisOut(
                    DataT[R], self.Mu[C], BigSigInv)
        #print ("PHI Generated..")
        return PHI

    def GetValTest(self, PHI, W):
        """Computes and returns the linear regression function
        Parameters:
        -----------
            PHI : M Basis Functions
            W : weight vector
        """
        Y = np.dot(W, np.transpose(PHI))
        ##print ("Test Out Generated..")
        return Y

    def GetErms(self, VAL_TEST_OUT, Target):
        """Computes and returns ERMS and accuracy values
        rms for the output data
        """
        _sum, _counter = 0.0, 0
        for i in range(0, len(VAL_TEST_OUT)):
            _sum = _sum + math.pow((Target[i] - VAL_TEST_OUT[i]), 2)
            if (int(np.around(VAL_TEST_OUT[i], 0)) == Target[i]):
                _counter += 1
        accuracy = (float((_counter * 100)) / float(len(VAL_TEST_OUT)))
        ##print ("Accuracy Generated..")
        ##print ("Validation E_RMS : " + str(math.sqrt(_sum/len(VAL_TEST_OUT))))
        return (str(accuracy) + ',' + str(math.sqrt(_sum / len(VAL_TEST_OUT))))

    def get_sgd_solution(self):
        """Computed weights for x datapoints iteratively updating, and returns the Erms for training, testing and validation data, as well as the testing accuracy.
        """
        # Gradient Descent Solution for Linear Regression
        La = 2
        L_Erms_Val, L_Erms_TR, L_Erms_Test, L_Accuracy_Test, W_Mat = [], [], [], [], []

        for i in range(0, epochs):

            # print (f'---------Iteration: {i} M{M} LR {learningRate} L :{C_Lambda}--------------')
            Delta_E_D = -np.dot(
                (self.training_target[i] -
                 np.dot(np.transpose(self.W), self.TRAINING_PHI[i])),
                self.TRAINING_PHI[i])
            La_Delta_E_W = np.dot(La, self.W)
            Delta_E = np.add(Delta_E_D, La_Delta_E_W)
            Delta_W = -np.dot(self.learning_rate, Delta_E)
            W_T_Next = self.W + Delta_W

            #-----------------TrainingData Accuracy---------------------#
            TR_TEST_OUT = self.GetValTest(self.TRAINING_PHI, W_T_Next)
            Erms_TR = self.GetErms(TR_TEST_OUT, self.training_target)
            L_Erms_TR.append(float(Erms_TR.split(',')[1]))

            #-----------------ValidationData Accuracy---------------------#
            VAL_TEST_OUT = self.GetValTest(self.VAL_PHI, W_T_Next)
            Erms_Val = self.GetErms(VAL_TEST_OUT, self.validation_target)
            L_Erms_Val.append(float(Erms_Val.split(',')[1]))

            #-----------------TestingData Accuracy---------------------#
            TEST_OUT = self.GetValTest(self.TEST_PHI, W_T_Next)
            Erms_Test = self.GetErms(TEST_OUT, self.testing_target)
            L_Erms_Test.append(float(Erms_Test.split(',')[1]))
            L_Accuracy_Test.append(float(Erms_Test.split(',')[0]))

        return ([L_Erms_TR, L_Erms_Val, L_Erms_Test, L_Accuracy_Test])
