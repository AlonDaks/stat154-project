from load_data import *
from rf_classify import *
from cross_validation import *
train_paths = document_paths("train")
XX, WORDS, vectorizer = featurize_documents(train_paths)
X, words = lemmatize_design_matrix(XX, WORDS)
X, words = remove_numerals(X, words)
