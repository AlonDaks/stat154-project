from load_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_predict
import math
from collections import OrderedDict
from numpy import mean

class Classifier:
    def __init__(self, X, y, max_features='auto', n_estimators=500):
        self.model = RandomForestClassifier(max_features=max_features,
                                            n_estimators=n_estimators,
                                            max_depth=10,
                                            oob_score=True, n_jobs=-1)
        self.X = X
        self.y = y

    def train(self):
        self.model.fit(self.X, self.y)

    def predict(self, new_data):
        return self.model.predict(new_data)
    
    def predict_proba(self, new_data):
        return self.model.predict_proba(new_data)
    
    def get_kth_prob(self, k, new_data):
        probs = self.predict_proba(new_data)
        return probs[:, k]

# Cross Validate the number of parameters used. Assume params is a list, X, y, n_trees are as in Classifier. 
def get_num_features(X, y, params, nfolds=10, n_trees=500):
    p = 0
    score = 0
    for num in params:
        rf = Classifier(X, y, max_features=num, n_estimators=n_trees)
        temp = cross_val_predict(rf.model, X, y, cv = nfolds, n_jobs=-1)
        temp = accuracy_score(y, temp)
        if temp > score:
            score = temp
            p = num
    return score, num

# OOB error rate plots for Random Forest Classifiers. numfeatures
def OOB_error_rates(nTrees, X, y, max_features = 'auto'):
    error_rates = OrderedDict()
    for i in nTrees:
        rf = Classifier(X = X, y = y, max_features=max_features, n_estimators=i)
        rf.train()
        oob_error = 1 - rf.model.oob_score_
        error_rates[i] = oob_error
    return error_rates



