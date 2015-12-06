import pickle
from os import listdir, path
import load_data as ld
from load_data import *
import rf_classify as rf 
import svm_classify as svm 
import sys
import numpy as np

(X, vectorizer) = pickle.load(open('pure_counts_df5.pkl'))
words = vectorizer.get_feature_names()

X = X.toarray()

#remove numerals and lemmatize all parts of speech
X, words = ld.remove_numerals(X, words)
X = ld.lemmatize_design_matrix(X, words)[0]
y = ld.get_labels(ld.document_paths('train'))

classes = ['child', 'history', 'religion', 'science']

validation_paths = [ld.training_path_by_class(c, 'validate') for c in classes]



model = pickle.load(open('svm.pkl'))

for i in range(len(classes)):
	kaggle_count_matrix = vectorizer.transform(validation_paths[i]).toarray()
	kaggle_count_matrix_lemmatized, words = ld.lemmatize_design_matrix(kaggle_count_matrix, vectorizer.get_feature_names())
	kaggle_count_matrix_lemmatized = ld.remove_numerals(kaggle_count_matrix_lemmatized, words)[0]

	predicted_labels = model.predict(kaggle_count_matrix_lemmatized)

	print '{0}: {1}'.format(classes[i], sum(predicted_labels == i) / float(len(predicted_labels)))
