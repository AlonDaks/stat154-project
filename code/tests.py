from load_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
import random
from rf_classify import Classifier
from cross_validation import CrossValidate
from time import time

paths = document_paths('train')
train_paths = random.sample(paths, 2000)
remaining_paths = [p for p in paths if p not in train_paths]
test_paths = random.sample(remaining_paths, 500)

start = time()
XX, initial_words, vectorizer = featurize_documents(train_paths)
print time() - start
X, words = lemmatize_design_matrix(XX, initial_words)
X, words = remove_numerals(X, words)

print("done with design matrix")

y_train = get_labels(train_paths)
y_test = get_labels(test_paths)

print 'classifying'
rf = Classifier(X, y_train, 'auto')

start = time()
rf.train()
print time() - start

X_test, words_test = lemmatize_design_matrix(
    vectorizer.transform(test_paths).toarray(), initial_words)
X_test = remove_numerals(X_test, words_test)[0]
y_predicted = rf.predict(X_test)

print accuracy_score(y_predicted, y_test)

rf_cross_validate = CrossValidate(rf.model, X, y_train)

start = time()
print(rf_cross_validate.get_cv_score(10))
print time() - start

y_prob = rf.predict_proba(X_test)


