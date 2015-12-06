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
X, words = ld.lemmatize_design_matrix(X, words)

kf = KFold(X.shape[0], n_folds=5)
C = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
kernels = ['poly', 'linear']
degrees = range(2, 5)

cross_validated_values = [[0] * 3, [0] * len(C), [0] * 2, [0] * 4]

for design_matrix_version in range(3):
    for c in range(len(C)):
        for k in range(len(kernels)):
            if kernels[k] == 'poly':
                for d in degrees:
                    current_model_accuracies = []
                    for train_indices, test_indices in kf:
                        X_train, X_test = X[train_indices, :], X[
                            test_indices, :]
                        y_train, y_test = y[train_indices], y[test_indices]
                        if design_matrix_version == 0:
                            transformer = TfidfTransformer()
                            X_train = transformer.fit_transform(X_train)
                            X_test = transformer.transform(X_test)
                        elif design_matrix_version == 1:
                            X_train, X_test = X_train.astype(
                                float), X_test.astype(float)
                            X_train = normalize(X_train, axis=1, norm='l1')
                            X_test = normalize(X_test, axis=1, norm='l1')
                        model = svm.Classifier(X_train,
                                               y_train,
                                               C=C[c],
                                               kernel=kernels[k],
                                               degree=degrees[d])
                        print design_matrix_version, lemmatize_version, c, k, "deg = %d" % d
                        model.train()
                        print "previous values worked"
                        predicted_y = model.predict(X_test)
                        current_model_accuracies.append(accuracy_score(
                            y_test, predicted_y))
                    cross_validated_values[design_matrix_version, c, k, d -
                                           1] = (np.mean(np.array(
                                               current_model_accuracies)), model)
            else:  #if kernel is either linear or rbf we dont iterate over degree
                current_model_accuracies = []
                for train_indices, test_indices in kf:
                    X_train, X_test = X[train_indices, :], X[test_indices, :]
                    y_train, y_test = y[train_indices], y[test_indices]
                    if design_matrix_version == 0:
                        transformer = TfidfTransformer()
                        X_train = transformer.fit_transform(X_train)
                        X_test = transformer.transform(X_test)
                    elif design_matrix_version == 1:
                        X_train, X_test = X_train.astype(float), X_test.astype(
                            float)
                        X_train = normalize(X_train, axis=1, norm='l1')
                        X_test = normalize(X_test, axis=1, norm='l1')
                    model = svm.Classifier(X_train,
                                           y_train,
                                           C=C[c],
                                           kernel=kernels[k])
                    print(design_matrix_version, lemmatize_version, c, k)
                    model.train()
                    print "previous values worked"
                    predicted_y = model.predict(X_test)
                    current_model_accuracies.append(accuracy_score(
                        y_test, predicted_y))
                cross_validated_values[design_matrix_version, c, k,
                                       0] = (np.mean(np.array(
                                           current_model_accuracies)), model)

pickle.dump(cross_validated_values, open('svm_cross_validated_values.pkl',
                                         'w+'))
