import matplotlib as mpl
from math import sqrt
from rf_classify import Classifier
mpl.use('TKAgg')
from matplotlib import pyplot as plt
import pandas as pd  
import numpy as np
from collections import OrderedDict

# Histogram of the sum of frequencies of each word in X. Assume sums is the sum of the columns of our design matrix X
def histogram(sums, numbins = 100):
    plt.figure(figsize=(12, 9)) 
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    plt.xticks(fontsize=14)  
    plt.yticks(range(5000, 30001, 5000), fontsize=14)
    plt.xlabel("Word Features", fontsize=16)  
    plt.ylabel("Frequencies", fontsize=16)
    plt.hist(sums,  
         color="#3F5D7D", bins=numbins)
    plt.suptitle("Word Feature Frequencies", fontsize = 20) 
    plt.show()

# OOB error rate plots for Random Forest Classifiers. numfeatures
def OOB_error_rates(min, max, X, y, max_features):
    error_rates = OrderedDict()
    for i in range(min, max+1):
        rf = Classifier(X = X, y = y, max_features=max_features, n_estimators=i)
        rf.train()
        oob_error = 1 - rf.model.oob_score_
        error_rates[i] = oob_error
    return error_rates


        

    
    