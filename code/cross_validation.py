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
        scores = cross_val_score(self.model, self.X, self.y, cv = cv, n_jobs = -1)
        return mean(scores)
    
    def set_model(self, model):
        self.model = model
        
    
