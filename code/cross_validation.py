from load_data import *
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold, cross_val_score
from numpy import mean

class CrossValidate:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X 
        self.y = y
    
    def get_cv_score(self, cv):
        kf = self.make_folds(cv)
        scores = cross_val_score(self.model, self.X, self.y, cv = kf)
        return mean(scores)
        
    def make_folds(self, cv):
        return KFold(self.X.shape[0], n_folds = cv, shuffle = True)
    
    def set_model(self, model):
        self.model = model
        
    
