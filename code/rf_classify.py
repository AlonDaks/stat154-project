from load_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from cross_validation import CrossValidate
import math


class Classifier:
    def __init__(self, X, y, max_features=None, n_estimators=500):
        if not max_features:
            max_features = int(math.sqrt(len(X)))
        self.model = RandomForestClassifier(max_features=max_features,
                                            n_estimators=n_estimators,
                                            oob_score=True)
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
    cv = CrossValidate(None, X, y)
    n = 0
    score = 0
    for p in params:
        rf = Classifier(X, y, p, n_trees)
        rf.train()
        cv.set_model(rf.model)
        temp = cv.get_cv_score(nfolds)
        if temp > score:
            score = temp
            n = p
    return score, n

