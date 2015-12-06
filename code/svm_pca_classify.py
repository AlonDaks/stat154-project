from load_data import *
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_predict
from numpy import mean
from sklearn.svm import SVC, LinearSVC
from scipy.sparse import csr_matrix
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
import numpy as np


class pcaClassifier():
    def __init__(self, X, y, num_comp, pca=IncrementalPCA(), C=1.0, kernel='linear', degree=2):
        self.model = SVC(C=C, kernel=kernel, degree=degree)
        self.pca=pca
        self.pca.set_params(n_components=num_comp)
        self.X = scale(X.astype(float))
        self.y = y
    def train(self):
        self.X = self.pca.fit_transform(self.X)
        self.model.fit(X, self.y)
    def predict(self, X):
        X = scale(X.astype(float))
        X = self.pca.transform(X)
        return self.model.predict(X)

