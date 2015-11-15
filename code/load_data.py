from nltk import word_tokenize
from os import listdir
import re
from sklearn.feature_extraction.text import CountVectorizer

RELATIVE_DATA_PATH = '../data/'

COMMON_ENGLISH_WORDS = [
    'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am',
    'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been',
    'but', 'by', 'can', 'cannot', 'could', 'dear', 'did', 'do', 'does',
    'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had',
    'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i',
    'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like',
    'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no',
    'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our',
    'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so',
    'some', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these',
    'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants', 'was', 'we',
    'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
    'will', 'with', 'would', 'yet', 'you', 'your'
]

COMMON_WORDS = ['project', 'gutenberg', 'ebook', 'title', 'author', 'release',
                'chapter']

ROMAN_NUMERALS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
                  'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii',
                  'xix', 'xx', 'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi',
                  'xxvii', 'xxviii', 'xxix', 'xxx', 'xxxi', 'xxxii', 'xxxiii',
                  'xxxiv', 'xxxv', 'xxxvi', 'xxxvii', 'xxxviii', 'xxxix', 'xl',
                  'xli', 'xlii', 'xliii', 'xliv', 'xlv', 'xlvi', 'xlvii',
                  'xlviii', 'xlix', 'l', 'li', 'lii', 'liii', 'liv', 'lv',
                  'lvi', 'lvii', 'lviii', 'lix', 'lx', 'lxi', 'lxii', 'lxiii',
                  'lxiv', 'lxv', 'lxvi', 'lxvii', 'lxviii', 'lxix', 'lxx',
                  'lxxi', 'lxxii', 'lxxiii', 'lxxiv', 'lxxv', 'lxxvi',
                  'lxxvii', 'lxxviii', 'lxxix', 'lxxx', 'lxxxi', 'lxxxii',
                  'lxxxiii', 'lxxxiv', 'lxxxv', 'lxxxvi', 'lxxxvii',
                  'lxxxviii', 'lxxxix', 'xc', 'xci', 'xcii', 'xciii', 'xciv',
                  'xcv', 'xcvi', 'xcvii', 'xcviii', 'xcix', 'c']

STOP_WORDS = COMMON_ENGLISH_WORDS + COMMON_WORDS + ROMAN_NUMERALS


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
                                 stop_words=STOP_WORDS,
                                 tokenizer=Tokenizer())
    X = vectorizer.fit_transform(document_paths).toarray()
    return X, vectorizer.get_feature_names()


class Tokenizer:
    def __call__(self, doc):
        return [self.strip(t) for t in word_tokenize(doc)]

    def strip(self, word):
        return re.sub('[\W_]+', '', word)
