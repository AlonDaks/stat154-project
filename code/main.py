from load_data import *
from rf_classify import *
from cross_validation import *
from time import time


# Random forest model.
rf = Classifier(X, y_train, 'auto', 500)

# Train the model. Time it.
start = time()
print "Training the model..."
rf.train()
print time() - start

# Reduce feature space using variable importance. Use default 'median' as threshold.
reduced_X, reduced_words = feature_selection(rf, X, words)

# Re-fit the model
rf = Classifier(reduced_X, y_train, 'auto', 500)
start = time()
print "Training the reduced model..."
rf.train()
print time() - start

# Cross-validate for number of features used at each plot. 
start = time()
score, p = get_num_features(X, y_train, params)
print time() - start


def document_paths(data_set):
    child_paths = training_path_by_class('child', data_set)
    history_paths = training_path_by_class('history', data_set)
    religion_paths = training_path_by_class('religion', data_set)
    science_paths = training_path_by_class('science', data_set)
    return child_paths + history_paths + religion_paths + science_paths
    
def training_path_by_class(class_name, data_set):
    if data_set == 'train':
        train_relative_paths = 'train/{0}/'.format(class_name)
    else:
        train_relative_paths = 'validate/{0}/'.format(class_name)
    return [RELATIVE_DATA_PATH + train_relative_paths + i
            for i in listdir(RELATIVE_DATA_PATH + train_relative_paths)
            if i != '.DS_Store']






