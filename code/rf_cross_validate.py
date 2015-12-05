import pickle
from sklearn.cross_validation import KFold
import numpy as np
import rf_classify as rf 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import load_data as ld
from sklearn.metrics import accuracy_score

DESIGN_MATRIX_PATH = 'cv_design_matrix.pkl'
X, y, words = pickle.load(open(DESIGN_MATRIX_PATH))

X = X.toarray()
X, words = ld.remove_numerals(X, words)

kf = KFold(X.shape[0], n_folds=5)
M = range(10, 140, 10)

cross_validated_values = np.zeros((3, 3, len(M)))

for design_matrix_version in range(3):
	for lemmatize_version in range(2):
		for m in range(len(M)):
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
				if design_matrix_version == 1:
					transformer = TfidfTransformer()
					X_train = transformer.fit_transform(X_train)
					X_test = transformer.transform(X_test)
				elif design_matrix_version == 2:
					X_train, X_test = X_train.astype(float), X_test.astype(float)
					X_train = normalize(X_train, axis=1, norm='l1')
					X_test = normalize(X_test, axis=1, norm='l1')]
				model = rf.Classifier(X_train, y_train, max_features=M[m])
				model.train()
				predicted_y = model.predict(X_test)
				current_model_accuracies.append(accuracy_score(y_test, predicted_y))
			cross_validated_values[design_matrix_version, lemmatize_version, m] = np.mean(np.array(current_model_accuracies))


pickle.dump(cross_validated_values, open('cross_validated_values.pkl', 'w+'))

