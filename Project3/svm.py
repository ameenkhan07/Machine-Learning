from sklearn.svm import SVC


class SupportVectorClassifier():
    def __init__(self, *args, **kwargs):
        self.train_data, self.train_tar, self.train_labels = args[0], args[1], args[2]

    def get_svc(self):
        """
        """
        classifier = SVC(kernel='rbf', random_state=0)
        classifier.fit(self.train_data, self.train_tar)
        return classifier

    def get_pred_data(self, classifier, test_data):
        """Predicting the Test set results
        """
        return classifier.predict(test_data)
