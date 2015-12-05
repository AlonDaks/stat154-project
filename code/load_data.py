from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from os import listdir
import re
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer,\
    VectorizerMixin
import stop_words
from sklearn.decomposition import IncrementalPCA
from collections import OrderedDict
import pickle
from sklearn.decomposition.incremental_pca import IncrementalPCA
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer

RELATIVE_DATA_PATH = '../data/'


def document_paths(data_set):
    if data_set == 'train' or data_set == 'validate':
        child_paths = training_path_by_class('child', data_set)
        history_paths = training_path_by_class('history', data_set)
        religion_paths = training_path_by_class('religion', data_set)
        science_paths = training_path_by_class('science', data_set)
        return child_paths + history_paths + religion_paths + science_paths
    if data_set == 'test':
        return [
            RELATIVE_DATA_PATH + 'test/' + i
            for i in listdir(RELATIVE_DATA_PATH + 'test/') if i != '.DS_Store'
        ]
        

def training_path_by_class(class_name, data_set):
    if data_set == 'train':
        train_relative_paths = 'train/{0}/'.format(class_name)
    else:
        train_relative_paths = 'validate/{0}/'.format(class_name)
    return [RELATIVE_DATA_PATH + train_relative_paths + i
            for i in listdir(RELATIVE_DATA_PATH + train_relative_paths)
            if i != '.DS_Store']

# Word Feature Matrix processing.
def featurize_documents(document_paths):
    vectorizer = CountVectorizer(decode_error='replace',
                                 input='filename',
                                 stop_words=stop_words.STOP_WORDS,
                                 min_df=.05,
                                 max_df=.95,
                                 tokenizer=Tokenizer())
    X = vectorizer.fit_transform(document_paths)
    words = vectorizer.get_feature_names()
    return X, words, vectorizer

#Word Feature Matrix with stemming
def stem_featurize_docs(document_paths):
    vectorizer = CountVectorizer(decode_error='replace',
                                 input='filename',
                                 stop_words=stop_words.STOP_WORDS,
                                 min_df=.05,
                                 max_df=.95,
                                 tokenizer=Tokenizer())
    X = vectorizer.fit_transform(document_paths)
    words = vectorizer.get_feature_names()
    X, words = stem_design_matrix(X, words)
    X, words = remove_numerals(X, words)
    return X, words, vectorizer


#PCA analysis
def pca_feature_matrix(X, n_components):
    pca = IncrementalPCA(n_components = n_components)
    X = pca.fit_transform(X)
    return X

#TFIDF processing
def tfidf(X):
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)
    return X, transformer

def generate_design_matrix():
    train_paths = document_paths("train")
    X, words, vectorizer = featurize_documents(train_paths)
    pickle.dump((X, y, words, vectorizer, transformer), open('design_matrix.pkl', 'w+'))

def get_labels(paths):
    labels = []
    for path in paths:
        label = None
        if 'child' in path:
            label = 0
        elif 'history' in path:
            label = 1
        elif 'religion' in path:
            label = 2
        else:
            label = 3
        labels.append(label)
    return np.array(labels)

def remove_numerals(X, words):
    merged_columns = OrderedDict()
    for i in range(len(words)):
        if not re.match("\A\d*\Z", words[i]):
            merged_columns[words[i]] = X[:, i]
    return np.array(merged_columns.values()).T, merged_columns.keys()

#Power feature, if document contains numeral
def numeral_feature(X, words):
    numerals = np.zeros(X.shape[0])
    for i in range(len(words)):
        if re.match("[0-9]+", words[i]):
            numerals = np.add(X[:, i], numerals)
    return numerals

# Convert nltk tags to wordnet tags
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# text is a list that contains all text from a single document. Assume no non alpha-numeric text.
def tag_words(words):
    tags = pos_tag(words)
    tags = [(word, get_wordnet_pos(tag)) for (word, tag) in tags]
    return tags


def get_word_count_dictionary(X, words):
    return {words[i]: sum(X[:, i]) for i in range(X.shape[1])}


# Assume X is our design matrix, words is a list of our features.
def lemmatize_design_matrix(X, words, only_nouns=False):
    wnl = WordNetLemmatizer()
    if only_nouns:
        words = [wnl.lemmatize(w) for w in words]
    else:
        words = tag_words(words)
        words = [wnl.lemmatize(w, t) for w,t in words]
    merged_columns = OrderedDict()
    for i in range(len(words)):
        if words[i] not in merged_columns:
            merged_columns[words[i]] = X[:, i]
        else:
            merged_columns[words[i]] += X[:, i]
    return np.array(merged_columns.values()).T, merged_columns.keys()

# Stemming
def stem_design_matrix(X, words, type='snowball'):
    if type == 'snowball':
        stemmer = SnowballStemmer("english")
    elif type == 'porter':
        stemmer = PorterStemmer()
    else:
        stemmer = LancasterStemmer()
    words = [stemmer.stem(w) for w in words]
    merged_columns = OrderedDict()
    for i in range(len(words)):
        if words[i] not in merged_columns:
            merged_columns[words[i]] = X[:, i]
        else:
            merged_columns[words[i]] += X[:, i]
    return np.array(merged_columns.values()).T, merged_columns.keys()
    

# Assume that X is our numpy array design matrix, header is a list of our features
def write_to_csv(filename, X, header):
    s = filename + ".csv"
    np.savetxt(s,
               X,
               fmt="%d",
               delimiter=",",
               header=",".join(header),
               comments="")


class Tokenizer:
    def __call__(self, doc):
        doc = self.strip_gutenberg_header_footer(doc)
        return [self.strip(t) for t in word_tokenize(doc) if len(self.strip(t)) >= 2]

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
    