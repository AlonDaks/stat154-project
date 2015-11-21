import sklearn
from sklearn.ensemble import RandomForestClassifier


class Classifier:
    def __init__(self, X, y, n_trees):
        self.model = RandomForestClassifier(n_estimators = n_trees)
        self.X = X
        self.y = y 
        
    def train(self):
        self.model.fit(self.X, self.y)
        
    def predict(self, new_data):
        return self.model.predict(new_data)
    
        