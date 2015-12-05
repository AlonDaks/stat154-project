from load_data import *
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_predict
from numpy import mean
from sklearn.svm import SVC


class Classifier:
	def __init__(self, X, y, C=1.0, kernel='rbf', degree=2):
		self.model = SVC(C=C, kernel=kernel, degree=degree)
        self.X = X
        self.y = y

    def train():
    	self.model.fit(self.X, self.y)

    def predict(self, new_data):
        return self.model.predict(new_data)