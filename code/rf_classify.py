from load_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from cross_validation import CrossValidate

class Classifier:
    def __init__(self, X, y):
        self.model = RandomForestClassifier(n_estimators = 500)
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
def get_num_features(X, y, params, nfolds = 10, n_trees = 500, type = "cv"):
    rf = Classifier(X, y)
    if n_trees != 500:
        rf.set_num_trees(n_trees)
    scores = {}
    
    if type == "OOB":
        for p in params:
            rf.set_num_params(p)
            rf.train()
            scores[rf.model.oob_score_] = p
        m1 = min(scores.keys())
        return scores[m1], m1
    else:   
        cv = CrossValidate(rf.model, X, y)
        for p in params:
            rf.set_num_params(p)
            rf.train()
            cv.set_model(rf.model)
            scores[cv.get_cv_score(nfolds)] = p
        m2 = min(scores.keys())
        return scores[m2], m2

    
        
        
        
    
    
    

