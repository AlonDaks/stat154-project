from load_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
import random
from rf_classify import Classifier
from cross_validation import CrossValidate


paths = document_paths('train')
train_paths = random.sample(paths, 5000)
remaining_paths = [p for p in paths if p not in train_paths]
test_paths = random.sample(remaining_paths, 500)

X, initial_words, vectorizer = featurize_documents(train_paths)
X, words = lemmatize_design_matrix(X, initial_words)

print("done with design matrix")

y_train = get_labels(train_paths)
y_test = get_labels(test_paths)

print 'classifying'
rf = Classifier(X, y_train, n_trees = 500)
rf.train()


X_test = lemmatize_design_matrix(vectorizer.transform(test_paths).toarray(), initial_words)[0]
y_predicted = rf.predict(X_test)

print accuracy_score(y_predicted, y_test)

rf_cross_validate = CrossValidate(X, y_train)

print(rf_cross_validate.cv_score(10))