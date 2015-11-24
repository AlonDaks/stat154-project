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


    # Cross Validate the number of parameters used. Or use OOB error. Assume params is a list, X, y, n_trees are as in Classifier. Type is either cv or OOB
def get_num_features(X, y, params, nfolds=10, n_trees=500, type="cv"):
    scores = {}
    if type == "OOB":
        for p in params:
            rf = Classifier(X, y, p, n_trees)
            rf.train()
            scores[rf.model.oob_score_] = p
        m1 = min(scores.keys())
        return scores[m1], m1
    else:
        cv = CrossValidate(rf.model, X, y)
        for p in params:
            rf = Classifier(X, y, p, n_trees)
            rf.train()
            cv.set_model(rf.model)
            scores[cv.get_cv_score(nfolds)] = p
        m2 = min(scores.keys())
        return scores[m2], m2
