from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from os import listdir
import re
from sklearn.feature_extraction.text import TfidfVectorizer
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
    vectorizer = TfidfVectorizer(decode_error='replace',
                                 input='filename',
                                 stop_words=stop_words.STOP_WORDS,
                                 min_df=.05,
                                 max_df=.95,
                                 tokenizer=Tokenizer())
    X = vectorizer.fit_transform(document_paths).toarray()
    return X, vectorizer.get_feature_names(), vectorizer

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


def remove_numerals(X, words):
    merged_columns = {}
    for i in range(len(words)):
        if not re.match("\A\d*\Z", words[i]):
            merged_columns[words[i]] = X[:, i]
    return np.array(merged_columns.values()).T, merged_columns.keys()

# Convert nltk tags to wordnet tags
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
# text is a string that contains all text from a single document. Assume no non alpha-numeric text.
def tag_words(text):
    words = []
    for sentence in sent_tokenize(text):
        tokenized_sentence = word_tokenize(sentence)
        item = pos_tag(tokenized_sentence)
        words += [(i, get_wordnet_pos(tag)) for (i, tag) in item]
    return words

def get_word_count_dictionary(X, words):
    return {words[i]: sum(X[:, i]) for i in range(X.shape[1])}

# assume that words is the output from runnign tag_words on our features.
def lemmatize_design_matrix(X, words):
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(w, t) for w,t in words]
    merged_columns = {}
    for i in range(len(words)):
        if words[i] not in merged_columns:
            merged_columns[words[i]] = X[:, i]
        else:
            merged_columns[words[i]] += X[:, i]
    return np.array(merged_columns.values()).T, merged_columns.keys()


# Assume that X is our numpy array design matrix, header is a list of our features
def write_to_csv(filename, X, header):
    np.savetxt("design_matrix.csv",
               X,
               fmt="%d",
               delimiter=",",
               header=",".join(header),
               comments="")


class Tokenizer:
    def __call__(self, doc):
        doc = self.strip_gutenberg_header_footer(doc)
        return [self.strip(t) for t in word_tokenize(doc)]

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
