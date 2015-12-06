import pickle
from sklearn.cross_validation import KFold
import numpy as np
import svm_classify as svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import load_data as ld
import load_data as Tokenizer
from sklearn.metrics import accuracy_score


DESIGN_MATRIX_PATH = 'pure_counts_df5.pkl'
X, vectorizer = pickle.load(open(DESIGN_MATRIX_PATH))

y = ld.get_labels(document_paths('train'))
words = vectorizer.get_feature_names()

X = X.toarray()
X, words = ld.remove_numerals(X, words)
X, words = lemmatize_design_matrix(X, words)

kf = KFold(X.shape[0], n_folds=5)
C = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 0, 1, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6]
kernels = ['rbf', 'poly', 'linear'] 
degrees = range(2, 5)

cross_validated_values = np.zeros((3, 2, len(C), len(kernels), 4))

for design_matrix_version in range(3):
	for lemmatize_version in range(2):
		for c in range(len(C)):
			for k in range(len(kernels)):
				if kernels[k] == 'poly':
					for d in len(degrees):
						current_model_accuracies = []
						for train_indices, test_indices in kf:
							X_train, X_test = X[train_indices,:], X[test_indices,:]
							y_train, y_test = y[train_indices], y[test_indices]
							if lemmatize_version == 0:
								X_train = ld.lemmatize_design_matrix(X_train, words, True)[0]
								X_test = ld.lemmatize_design_matrix(X_test, words, True)[0]
							elif lemmatize_version == 1:
								X_train = ld.lemmatize_design_matrix(X_train, words, False)[0]
								X_test = ld.lemmatize_design_matrix(X_test, words, False)[0]
							if design_matrix_version == 0:
								transformer = TfidfTransformer()
								X_train = transformer.fit_transform(X_train)
								X_test = transformer.transform(X_test)
							elif design_matrix_version == 1:
								X_train, X_test = X_train.astype(float), X_test.astype(float)
								X_train = normalize(X_train, axis=1, norm='l1')
								X_test = normalize(X_test, axis=1, norm='l1')
							model = svm.Classifier(X_train, y_train, C=C[c], kernel=kernels[k], degree = degrees[d])
							print design_matrix_version, lemmatize_version, c, k, "deg = %d"%d
							model.train()
							print "previous values worked"
							predicted_y = model.predict(X_test)
							current_model_accuracies.append(accuracy_score(y_test, predicted_y))
						cross_validated_values[design_matrix_version, lemmatize_version, c, k, d] = np.mean(np.array(current_model_accuracies))
				else: #if kernel is either linear or rbf we dont iterate over degree
					current_model_accuracies = []
					for train_indices, test_indices in kf:
						X_train, X_test = X[train_indices,:], X[test_indices,:]
						y_train, y_test = y[train_indices], y[test_indices]
						if lemmatize_version == 0:
							X_train = ld.lemmatize_design_matrix(X_train, words, True)[0]
							X_test = ld.lemmatize_design_matrix(X_test, words, True)[0]
						elif lemmatize_version == 1:
							X_train = ld.lemmatize_design_matrix(X_train, words, False)[0]
							X_test = ld.lemmatize_design_matrix(X_test, words, False)[0]
						if design_matrix_version == 0:
							transformer = TfidfTransformer()
							X_train = transformer.fit_transform(X_train)
							X_test = transformer.transform(X_test)
						elif design_matrix_version == 1:
							X_train, X_test = X_train.astype(float), X_test.astype(float)
							X_train = normalize(X_train, axis=1, norm='l1')
							X_test = normalize(X_test, axis=1, norm='l1')
						model = svm.Classifier(X_train, y_train, C=C[c], kernel=kernels[k])
						print (design_matrix_version, lemmatize_version, c, k)
						model.train()
						print "previous values worked"
						predicted_y = model.predict(X_test)
						current_model_accuracies.append(accuracy_score(y_test, predicted_y))
					cross_validated_values[design_matrix_version, lemmatize_version, c, k, 0] = np.mean(np.array(current_model_accuracies))


pickle.dump(cross_validated_values, open('svm_cross_validated_values.pkl', 'w+'))