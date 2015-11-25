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
        self.model = RandomForestClassifier(max_features,
                                            max_features=max_features,
                                            n_estimators=n_estimators)
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
    
    def set_num_params(self, p):
        self.model.max_features = p
    
    def set_num_trees(self, n):
        self.model.n_estimators = n


    # Cross Validate the number of parameters used. Or use OOB error. Assume params is a list, X, y, n_trees are as in Classifier. Type is either cv or OOB
def get_num_features(X, y, params, nfolds=10, n_trees=500, type="cv"):
    scores = {}
    cv = CrossValidate(None, X, y)
    for p in params:
        rf = Classifier(X, y, p, n_trees)
        rf.train()
        cv.set_model(rf.model)
        scores[cv.get_cv_score(nfolds)] = p
        m2 = min(scores.keys())
    return scores[m2], m2

