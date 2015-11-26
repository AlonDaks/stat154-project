from load_data import *
from rf_classify import *
from cross_validation import *
from datetime import datetime
train_paths = document_paths("train")
XX, WORDS, vectorizer = featurize_documents(train_paths)
X, words = lemmatize_design_matrix(XX, WORDS)
X, words = remove_numerals(X, words)

y = get_labels(train_paths)

start = datetime.now()
rf = Classifier(X, y, 'auto', 500)
rf.train()
print datetime.now() - start

start = datetime.now()
rf.model.set_params(n_estimators=400)
rf.train()
print datetime.now() - start

start = datetime.now()
rf = Classifier(X, y, 'auto', 400)
rf.train()
print datetime.now() - start


