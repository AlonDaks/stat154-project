import matplotlib as mpl
from math import sqrt
from rf_classify import Classifier
mpl.use('TKAgg')
from matplotlib import pyplot as plt
import pandas as pd  
import numpy as np
from collections import OrderedDict
from ggplot import *
from load_data import *
from sklearn.metrics.ranking import roc_curve
from ROC_analysis import get_binary_labels

# Histogram of the sum of frequencies of each word in X. Assume sums is the sum of the columns of our design matrix X
def histogram(sums, numbins = 100):
    plt.figure(figsize=(12, 9)) 
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    plt.xlim(0, len(sums)) 
    plt.xticks(fontsize=14)  
    plt.yticks(range(5000, 30001, 5000), fontsize=14)
    plt.xlabel("Word Features", fontsize=16)  
    plt.ylabel("Frequencies", fontsize=16)
    plt.hist(sums,  
         color="#3F5D7D", bins=numbins)
    plt.suptitle("Word Feature Frequencies", fontsize = 20) 
    plt.show()
    


# assume that errors is an orderedDictionary, 
def OOB_plot(errors, X, y, max_features = 'auto'):
    m = min(errors.keys())
    M = max(errors.keys())
    plt.plot(errors.keys(), errors.values())
    plt.xlim(m, M)
    plt.xlabel("Number of trees")
    plt.ylabel("OOB error rate")
    plt.show()

# Helper function for ROC plot below
def get_binary_labels(k, y):
    return [1 if pred == k else 0 for pred in y]

# ROC plot for kth class against all others. Assume y_prob is an np array with class probabilities
def ROC_plot(y_true, y_prob, k):
    y_true = get_binary_labels(k, y_true)
    pred_prob = []
    for i in range(y_prob.shape[0]):
        pred_prob.append(y_prob[i, k-1])
    fpr, tpr, _ = roc_curve(y_true, pred_prob)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    
    


    

    
    