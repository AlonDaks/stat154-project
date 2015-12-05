from sklearn import *
from load_data import *
from sklearn.metrics.ranking import roc_curve
import numpy as np
import pickle

def generate_cv_paths():
    child_paths = np.array(training_path_by_class('child', 'train'))
    history_paths = np.array(training_path_by_class('history', 'train'))
    religion_paths = np.array(training_path_by_class('religion', 'train'))
    science_paths = np.array(training_path_by_class('science', 'train'))
    random_child_indexes = np.random.choice(
        range(len(child_paths)),
        size=1600,
        replace=False)
    random_history_indexes = np.random.choice(
        range(len(history_paths)),
        size=1200,
        replace=False)
    random_religion_indexes = np.random.choice(
        range(len(religion_paths)),
        size=550,
        replace=False)
    random_science_indexes = np.random.choice(
        range(len(science_paths)),
        size=1650,
        replace=False)
    return np.concatenate((child_paths[
        random_child_indexes], history_paths[
            random_history_indexes], religion_paths[
                random_religion_indexes], science_paths[random_science_indexes]))

def generate_cv_objects(cv_paths, rmv=0.05):
    vectorizer = CountVectorizer(decode_error='replace',
                                 input='filename',
                                 stop_words=stop_words.STOP_WORDS,
                                 min_df=rmv,
                                 max_df=1-rmv,
                                 tokenizer=Tokenizer())
    return vectorizer.fit_transform(cv_paths), vectorizer.get_feature_names(), vectorizer

np.random.seed(11111)
cv_paths = generate_cv_paths()
y = get_labels(cv_paths)
X, words, vect = generate_cv_objects(cv_paths)
pickle.dump((X, y, words), open("cv_design_matrix.pkl", "w+"))
    
    
