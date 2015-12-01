import matplotlib as mpl
from math import sqrt
from rf_classify import Classifier
from sklearn.preprocessing.label import label_binarize
from ggplot.themes.element_target import legend_text
mpl.use('TKAgg')
from matplotlib import pyplot as plt
import pandas as pd  
import numpy as np
from collections import OrderedDict
from ggplot import *
from load_data import *
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from numpy import interp

color_list = [
    '#FFAAAA', 
    '#ff5b00',
    '#c760ff', 
    '#f43605', 
    '#00FF00',
    '#0000FF', 
    '#4c9085']

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

# Helper function for ROC plots below. Used in the legend.
def get_class_name(k):
    if k == 1:
        return 'Child'
    elif k == 2:
        return 'History'
    elif k == 3:
        return 'Religion'
    else:
        return 'Science'

# Helper function for ROC plots. Return pandas DataFrame with columns 'tpr', 'fpr' and 'class', used for the basic ROC plots.
def one_vs_all_ROC(y_true, y_prob):
    y_temp = get_binary_labels(1, y_true)
    pred_prob = y_prob[:, 1]
    fpr, tpr, _ = roc_curve(y_temp, pred_prob)
    classes = ["Child vs all \n (AUC = " + str(auc(fpr,tpr)) + ")"]*len(fpr)
    dat = pd.DataFrame(np.array([fpr.tolist(), tpr.tolist(), classes]).T, columns = ["fpr", "tpr","class"])
    for i in [2,3,4]:
        y_temp = get_binary_labels(i, y_true)
        pred_prob = y_prob[:, i-1]
        fpr, tpr, _ = roc_curve(y_temp, pred_prob)
        classes = [get_class_name(i) + " vs all \n (AUC = " + str(auc(fpr, tpr)) + ")"]*len(fpr)
        temp = pd.DataFrame(np.array([fpr.tolist(), tpr.tolist(), classes]).T, columns = ["fpr", "tpr","class"])
        dat = dat.append(temp)
    return dat

# Helper function for ROC plots. Return pandas DataFrame with columns 'tpr', 'fpr' and 'class'. Here, class 'Micro-average'.
def micro_avg_ROC(y_true, y_prob):
    y = label_binarize(y_true, classes = [1,2,3,4]) 
    fpr, tpr, _ = roc_curve(y.ravel(), y_prob.ravel())
    classes = ["Micro Average (AUC = " + str(auc(fpr, tpr)) + ")"] * len(fpr)
    return pd.DataFrame(np.array([fpr.tolist(), tpr.tolist(), classes]).T, columns = ["fpr", "tpr", "class"])

# Helper function for ROC plots. Return pandas DataFrame with columns 'tpr', 'fpr' and 'class'. Here, class is 'Macro-average'.
def macro_avg_ROC(y_true, y_prob):
    dat = one_vs_all_ROC(y_true, y_prob)
    all_fpr = np.unique(dat.loc[:,'fpr'])
    mean_tpr = interp(all_fpr.tolist(), dat.loc[:, "fpr"].tolist(), dat.loc[:, "tpr"].tolist())
    classes = ["Macro Average \n (AUC = " + str(auc(all_fpr, mean_tpr)) + ")"] * len(all_fpr)
    return pd.DataFrame(np.array([all_fpr.tolist(), mean_tpr.tolist(), classes]).T, columns = ["fpr", "tpr", "class"])
    
# Basic ROC plot for class k against the other three classes. Plot an ROC for each k. Assume y_prob is an np array with class probabilities
def Basic_ROC_plot(y_true, y_prob):
    dat = one_vs_all_ROC(y_true, y_prob)
    p = ggplot(aes(x = 'fpr', y = 'tpr', color = "class"), data = dat) + geom_step()
    p += xlab("False Positive Rate")
    p += ylab("True Positive Rate")
    p += geom_abline(intercept=0, slope=1, colour = 'black', linetype='dashed')
    p += ggtitle("Class against all others ROC Curves")
    return p
    
# Multi Class ROC curve with micro averaging.
def microAvg_ROC_plot(y_true, y_prob):
    dat = micro_avg_ROC(y_true, y_prob)
    p = ggplot(aes(x = 'fpr', y = 'tpr'), data = dat) + geom_line()
    p += geom_abline(intercept = 0, slope = 1, color = 'black', linetype = 'dashed')
    return p

# Mutli Class ROC curve with macro averaging.
def macroAvg_ROC_plot(y_true, y_prob):
    dat = macro_avg_ROC(y_true, y_prob)
    p = ggplot(aes(x = 'fpr', y = 'tpr'), data = dat) + geom_line()
    p += geom_abline(intercept = 0, slope = 1, color = 'black', linetype = 'dashed')
    return p
          
    


    

    
    