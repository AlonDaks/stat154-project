import pickle
from os import listdir, path
import load_data as ld
from load_data import Tokenizer
import rf_classify as rf 
import sys

KAGGLE_DATA_PATH = '../data/kaggle_in_class_data'

(X, words, vectorizer) = pickle.load(open('design_matrix.pkl'))

def train_kaggle_model():

	X = X.toarray()

	#remove numerals and lemmatize all parts of speech
	X, words = ld.remove_numerals(X, words)
	X = ld.lemmatize_design_matrix(X, words)[0]
	y = ld.get_labels(ld.document_paths('train'))

	model = rf.Classifier(X, y, max_features = 150, n_estimators = 1100)
	model.train()
	pickle.dump(model, open('rf.pkl', 'w+'), protocol = -1)

def classify_kaggle_documents():
	model = pickle.load(open('rf.pkl'))

	num_kaggle_files = len(listdir(KAGGLE_DATA_PATH)) - path.exists(KAGGLE_DATA_PATH + '.DS_STORE')
	kaggle_files = ['{0}/{1}.txt'.format(KAGGLE_DATA_PATH, i) for i in range(num_kaggle_files)]
	kaggle_count_matrix = vectorizer.transform(kaggle_files).toarray()
	kaggle_count_matrix_lemmatized, words = ld.lemmatize_design_matrix(kaggle_count_matrix, vectorizer.get_feature_names())
	kaggle_count_matrix_lemmatized = ld.remove_numerals(kaggle_count_matrix_lemmatized, words)[0]

	pickle.dump(kaggle_count_matrix_lemmatized, open('in_class_design_matrix.pkl', 'w+'), protocol=-1)
	predicted_labels = model.predict(kaggle_count_matrix_lemmatized)

	print 'id,category'
	for i in range(len(predicted_labels)):
		print '{0},{1}'.format(i, predicted_labels[i])

if __name__ == '__main__':
	if sys.argv[1] == 'train':
		train_kaggle_model()
	if sys.argv[1] == 'classify':
		classify_kaggle_documents()