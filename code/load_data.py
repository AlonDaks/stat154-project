from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from os import listdir
import re
from sklearn.feature_extraction.text import CountVectorizer
import stop_words

RELATIVE_DATA_PATH = '../data/'


def document_paths(data_set):
    if data_set == 'train':
        child_paths = training_path_by_class('child')
        history_paths = training_path_by_class('history')
        religion_paths = training_path_by_class('religion')
        science_paths = training_path_by_class('science')
        return child_paths + history_paths + religion_paths + science_paths
    if data_set == 'test':
        return [
            RELATIVE_DATA_PATH + 'test/' + i
            for i in listdir(RELATIVE_DATA_PATH + 'test/') if i != '.DS_Store'
        ]


def training_path_by_class(class_name):
    train_relative_paths = 'train/{0}/'.format(class_name)
    return [RELATIVE_DATA_PATH + train_relative_paths + i
            for i in listdir(RELATIVE_DATA_PATH + train_relative_paths)
            if i != '.DS_Store']


def featurize_documents(document_paths):
    vectorizer = CountVectorizer(decode_error='replace',
                                 input='filename',
                                 stop_words=stop_words.STOP_WORDS,
                                 tokenizer=Tokenizer())
    X = vectorizer.fit_transform(document_paths).toarray()
    return X, vectorizer.get_feature_names()


def get_labels(paths):
    labels = []
    for path in paths:
        label = None
        if 'child' in path:
            label = 1
        elif 'history' in path:
            label = 2
        elif 'religion' in path:
            label = 3
        else:
            label = 4
        labels.append(label)
    return np.array(labels)

def get_word_count_dictionary(X, words):
    return {words[i]: sum(X[:, i]) for i in range(X.shape[1])}


def lemmatize_design_matrix(X, words):
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(w) for w in words]
    merged_columns = {}
    bad_indecies = []
    for i in range(len(words)):
        if words[i] not in merged_columns:
            merged_columns[words[i]] = X[:, i]
        else:
            merged_columns[words[i]] += X[:, i]
            bad_indecies.append(words[i])
    for j in bad_indecies:
        words.remove(j)   
    return np.array(merged_columns.values()).T, words


class Tokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        doc = self.strip_gutenberg_header_footer(doc)
        return [self.wnl(self.strip(t)) for t in word_tokenize(doc)]

    def strip(self, word):
        return re.sub('[\W_]+', '', word)

    def strip_gutenberg_header_footer(self, doc):
        try:
            doc = re.split(
                '\*\*\* start of this project gutenberg ebook .* \*\*\*',
                doc)[1]
        except:
            pass
        try:
            doc = re.split(
                '\*\*\* end of this project gutenberg ebook .* \*\*\*', doc)[0]
        except:
            pass
        return doc
    
    
    
