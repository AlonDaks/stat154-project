from load_data import *
from rf_classify import *
from cross_validation import *
from time import time
# Get the paths for our trainging set and test set.
train_paths = document_paths("train")
test_paths = document_paths("test")

# Process our inital design matrix. Time it for now just so we can see how long it takes
start = time()
print "Processing design matrix..."
XX, WORDS, vectorizer = featurize_documents(train_paths)
print time() - start

# Take our inital design matrix and features, lemmatize and remove numerals
start = time()
print "Lemmatizing and removing numerals..."
X, words = lemmatize_design_matrix(XX, WORDS)
X, words = remove_numerals(X, words)
print time() - start

# Get labels for train and test set
y_train = get_labels(train_paths)
y_test = get_labels(test_paths)

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
params = range(10, 110, 10)
score, p = get_num_features(reduced_X, y_train, params)




