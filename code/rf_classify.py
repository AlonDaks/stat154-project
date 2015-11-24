from load_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

class Classifier:
    def __init__(self, X, y, n_trees):
        self.model = RandomForestClassifier(n_estimators = n_trees)
        self.X = X
        self.y = y 
        
    def train(self):
        self.model.fit(self.X, self.y)
        
    def predict(self, new_data):
        return self.model.predict(new_data)
  

train_paths  = np.random.choice(document_paths('train'), size = 3000, replace=False)
test_paths  = np.random.choice(document_paths('train'), size = 500, replace=False)
X, words, vectorizer = featurize_documents(train_paths)
X, words = lemmatize_design_matrix(X, words)
y_train = get_labels(train_paths)

print 'classifying'
rf = Classifier(X, y_train, n_trees = 500)
rf.train()

X_test = lemmatize_design_matrix(vectorizer.transform(test_paths).toarray(), words)[0]
y_predicted = rf.predict(X_test)

print accuracy_score(y_predicted, get_labels(test_paths))