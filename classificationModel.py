import numpy as np
from sklearn.svm import libsvm
from sklearn.svm import SVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
class Model(object):
	def __init__(self):
		self.model = None

	"""
	Initialize training of a new weighted linear regression model by choosing best parameters.
	Sets the trained model for this object.
	"""


	def train(self, traindata, trainLabels):
		# Cs = [0.001, 0.01, 0.1, 1, 10]
		# gammas = [0.001, 0.01, 0.1, 1]
		# params = {'C': Cs, 'gamma': gammas}
		self.model = SVC(kernel='linear')
		self.model.fit(traindata, trainLabels)




		"""
        Test the weighted SVM to predict labels of a given test data.
        Returns the result of prediction, and confidence behind the prediction
        """
	def test(self, testdata):
		if len(testdata) == 1:
			testdata = np.reshape(testdata, (1, -1))
		predictions = self.model.predict(testdata)
		return predictions

	"""
	Get summary of the model
	"""
	def getModelSummary(self):
		summary = '************************* Model   S U M M A R Y ************************\n'

		return summary