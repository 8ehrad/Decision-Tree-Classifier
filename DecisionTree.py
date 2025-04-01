#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 21:03:23 2023

@author: 13ehrad
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import category_encoders as ce
import matplotlib.pyplot as plt

import math
import time
import sys
import csv
import itertools
import openml
from memory_profiler import profile

import warnings
warnings.filterwarnings("ignore")

def get_adult_dataset():
    """ GET ADULT DATASET """
    # Get dataset by ID
    dataset = openml.datasets.get_dataset(1590) # Adult

    # Get dataset by name
    #dataset = openml.datasets.get_dataset('Fashion-MNIST')

    # Get the data itself as a dataframe (or otherwise)
    X_adult, y_adult, _, _ = dataset.get_data(dataset_format="dataframe")
    X_adult.dropna(inplace=True)
    y_adult = (X_adult['class'] == ">50K").astype(int)
    X_adult.drop(['class'], axis = 1, inplace=True)
    cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'native-country']
    ohe = ce.OneHotEncoder(cols=cols)
    X_adult_enc = pd.get_dummies(X_adult, columns=cols, drop_first=False)
    return X_adult_enc, y_adult

def get_wine_dataset():
    # Get dataset by ID
    dataset = openml.datasets.get_dataset(40691) # Wine
    # Get the data itself as a dataframe (or otherwise)
    X_wine, y_wine, _, _ = dataset.get_data(dataset_format="dataframe")
    X_wine.dropna(inplace=True)
    y_wine = X_wine['class'].astype(int)
    X_wine.drop(['class'], axis=1, inplace=True)
    return X_wine, y_wine

def sampling_dataset(X, y, sample_portion):
    # sample X and y in the same way
    sample_size = math.floor(len(X) * sample_portion)
    # index reset
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)
    # sampling
    X_sample = X.sample(sample_size)
    y_sample = y[X_sample.index]
    return X_sample, y_sample


def gini(y):
    """
    This function computes gini index for a given target y
    """
    y = np.array(y)
    # extract all classes
    classes = np.unique(y)
    
    g = 1
    for c in classes:
        g -= (y[y==c].shape[0] / y.shape[0])**2
    return g
    

def entropy(y):
    """
    This function computes entropy for a given target y
    """
    y = np.array(y)
    # extract all classes
    classes = np.unique(y)
    
    e = 0
    for c in classes:
        # add to entropy for all classes
        p = y[y==c].shape[0] / y.shape[0]
        if p:
            e -= p * np.log2(p)
    return e

def information_gain(y, mask, func = entropy):
    """
    This function computes information gain for a split
    """
    y, mask = np.array(y), np.array(mask)
    set1, set2 = y[mask], y[~mask]
    
    # First check if any of sets is empty
    if not set1.shape[0] or not set2.shape[0]:
        return 0 # information gain is 0
    
    ig = func(y) - set1.shape[0]/y.shape[0] * func(set1) \
        - set2.shape[0]/y.shape[0] * func(set2)
    return ig

def find_best_split(X, y, func = entropy):
    """
    This function find the best split for a given dataset and target
    based on information gain
    X : features 
    y : target
    """    
    
    X, y = np.array(X), np.array(y)
    ig_max = 0 # max information gain for a particular split
    ind_max = -1 # index of the column(variable) 
    val_max = -1 # split value of that feature
     
    # find best values to split from for each column 
    for j in range(X.shape[1]):
        # find unique values of this col and sort them
        vals = np.sort(np.unique(X[:,j]))
        for i in range(1, vals.shape[0]):   # ignore the first value 
            val = vals[i]
            mask = (X[:,j] >= val)
            ig = information_gain(y, mask, func)
            if ig > ig_max:
                ig_max = ig
                ind_max = j
                val_max = val
    # return maximum information gain, its according feature and value
    return ig_max, ind_max, val_max

def mode(y):
    """ Set the mode of a given set of labels """
    y = np.array(y).astype(int)
    return np.bincount(y).argmax()
    
class Node():
    """
    This class builds each node of the decision tree
    """
    def __init__(self, depth = 0, selected_feature = None, selected_value = None, c = False):
        self.left = None # left child 
        self.right = None # right child
        self.selected_feature = selected_feature # the feature splitting data
        self.selected_value = selected_value # selected value according to previous feature
        # Note that the default setting is selected feature >= selected value
        self.c = c # class for that node. if False, means that we are not at a leaf
        #self.mask = mask # a boolean array to indicate which data rows are present at this node
        self.depth = depth # depth of the node in tree

class DecisionTree():        
    """
    This class handles the implementation of a decision tree. It starts with a node
    as root, and is able to train and predict. 
    """
    def __init__(self, max_depth=1000, min_samples_split=2, min_information_gain=-1):
        
        self.root = Node()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
    
    def _travers_train(self, node, X, y, func=entropy):
        """ Recieves the data and continue forming tree from this node """
        X, y = np.array(X), np.array(y)

        # first check the end conditions 
        if np.unique(y).shape[0] == 1 or node.depth >= self.max_depth or \
            X.shape[0] < self.min_samples_split:
            # assign a class to this node and return 
            node.c = mode(y)
            return
        
        # compute max information  gain
        ig_max, ind_max, val_max = find_best_split(X, y, func)
        if ig_max < self.min_information_gain or ig_max == 0:
            node.c = mode(y)
            return 
        
        # now we need to continue splitting the feature space
        node.selected_feature = ind_max
        node.selected_value = val_max
        # form right and left children 
        mask = (X[:,ind_max] >= val_max)
        Xr, Xl, yr, yl = X[mask,:], X[~mask,:], y[mask], y[~mask]
        node.right = Node(node.depth+1)
        node.left = Node(node.depth+1)
        # call left and right children with according datasets
        self._travers_train(node.left, Xl, yl)
        self._travers_train(node.right, Xr, yr)
        return
    
    def _travers_predict(self, node, X):
        """ Recieves the data and labels them based on the trained tree """
        X = np.array(X)
        # check if we're in one of the leaves, we should predict
        if node.c is not False: # if a class is assigned to this node
            return np.full((X.shape[0]), node.c)
        
        # means we should split the dataset further 
        mask = (X[:, node.selected_feature] >= node.selected_value)
        # split dataset based on the selected feature and value
        Xr, Xl = X[mask], X[~mask]
        yl = self._travers_predict(node.left, Xl)
        yr = self._travers_predict(node.right, Xr)
        y = np.full((X.shape[0]), -1)
        if yr is not None:
            y[mask] = yr
        if yl is not None:
            y[~mask] = yl
        return y
    # @profile
    def train(self, X, y, func=entropy):
        """ Train the decision tree on a dataset X with labels y """
        # call a second function to form the tree recursively
        self._travers_train(self.root, X, y, func)
        return
    
    def predict(self, X):
        """ Predict the labels for a given dataset X """
        # call a second function to search the tree recursively
        y = self._travers_predict(self.root, X)
        return y

def compare_data_size(X, y):
    """ 
    This function compares training and prediction time
    for both decision trees based on the size of the data
    """
    
    sample_portions = np.linspace(0.1, 1, 10)
    dt_tr_times, dt_pr_times, clf_tr_times, clf_pr_times = [], [], [], []
    
    for sample_portion in sample_portions:
        X_sample, y_sample = sampling_dataset(X, y, sample_portion)

        X_train, X_test, y_train, y_test = \
            train_test_split(X_sample, y_sample, test_size=0.1, random_state=0)
        
        """ MY DECISION TREE """
        dt = DecisionTree()
        t0 = time.time()
        dt.train(X_train, y_train)
        dt_tr_times.append(time.time() - t0)
        
        t0 = time.time()
        y_pred = dt.predict(X_test)
        dt_pr_times.append(time.time() - t0)
    
        """ SKLEARN DECISION TREE """
        clf = DecisionTreeClassifier(random_state=0)
        t0 = time.time()
        clf.fit(X_train, y_train)
        clf_tr_times.append(time.time() - t0)

        t0 = time.time()
        y_pred = clf.predict(X_test)
        clf_pr_times.append(time.time() - t0)
        
    # Plot time based on my model
    x = np.linspace(0.1, 1, 10)
    plt.figure()
    plt.plot(x, dt_tr_times, label='my decision tree training time')
    plt.plot(x, dt_pr_times, label='my decision tree prediction time')
    plt.xlabel('Proportion of the Wine dataset')
    plt.ylabel('Seconds')
    plt.legend()
    plt.savefig('my_time_vs_sample_size.png')
    plt.show()
    
    # Plot time based on sklearn model
    plt.figure()
    plt.plot(x, clf_tr_times, label='sklearn decision tree training time')
    plt.plot(x, clf_pr_times, label='sklearn decision tree prediction time')
    plt.xlabel('Proportion of the Wine dataset')
    plt.ylabel('Seconds')
    plt.legend()
    plt.savefig('skl_time_vs_sample_size.png')
    plt.show()


def test_models(datasets, test_sizes, hyperparameters):
    """
    This function tests both decision trees under different settings
    and evaluates computational and machine learning aspects    
    """
    
    # save the results of our model and sklearn's in two dictionaries
    # in each dictionary, name of the datasets and hyperparameters are keys
    # computational and ML aspects are the values
    my_result, skl_result = {}, {}
    
    for dataset in datasets:
        
        # extract X, y, label
        X, y, label, sample_size = dataset[0], dataset[1], dataset[2], dataset[3]
        # add a dictionary for each dataset
        my_result[label], skl_result[label] = dict(), dict()
        
        for test_size in test_sizes:
            
            # split the dataset
            X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_size, random_state=0)
            
            for max_depth in hyperparameters['max_depth']:
                for min_samples_split in hyperparameters['min_samples_split']:
                    
                    # append this setting as a key
                    my_result[label][(test_size, max_depth, min_samples_split)] = dict()
                    skl_result[label][(test_size, max_depth, min_samples_split)] = dict()
                    
                    """ MY DECISION TREE """
                    dt = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
                    t0 = time.time()
                    dt.train(X_train, y_train)
                    my_model_train_time = time.time() - t0
                    # print("Training time for our decision tree with \
                    # max_depth = {} is :".format(max_depth), time.time() - t0)
                    t0 = time.time()
                    my_y_pred = dt.predict(X_test)
                    my_model_pred_time = time.time() - t0
                    # print("Prediction time for our decision tree is :", time.time() - t0)
                    # print("Classification report for our classifier :\n", 
                           # classification_report(np.array(y_test).astype(int), my_y_pred))
                
                
                    """ SKLEARN DECISION TREE """
                    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=0)
                    t0 = time.time()
                    clf.fit(X_train, y_train)
                    skl_model_train_time = time.time() - t0
                    # print("Training time for sklearn's decision tree is :", time.time() - t0)
                    t0 = time.time()
                    skl_y_pred = clf.predict(X_test)
                    skl_model_pred_time = time.time() - t0
                    # print("Prediction time for sklearn's decision tree is :", time.time() - t0)
                    # print("Classification report for sklearn's classifier :\n", 
                          # classification_report(y_test, y_pred))
                    
                    # add all the results to our dictionary
                    y_test = np.array(y_test).astype(int)
                    my_result[label][(test_size, max_depth, min_samples_split)] = {
                        'sample_size' : sample_size,
                        'training_time' : my_model_train_time,
                        'prediction_time' : my_model_pred_time,
                        'accuracy' : accuracy_score(y_test, my_y_pred),
                        'precision' : precision_score(y_test, my_y_pred, average='weighted'),
                        'recall' : recall_score(y_test, my_y_pred, average='weighted'),
                        'f1' : f1_score(y_test, my_y_pred, average='weighted'),
                    }
                    skl_result[label][(test_size, max_depth, min_samples_split)] = {
                        'sample_size' : sample_size,
                        'training_time' : skl_model_train_time,
                        'prediction_time' : skl_model_pred_time,
                        'accuracy' : accuracy_score(y_test, skl_y_pred),
                        'precision' : precision_score(y_test, skl_y_pred, average='weighted'),
                        'recall' : recall_score(y_test, skl_y_pred, average='weighted'),
                        'f1' : f1_score(y_test, skl_y_pred, average='weighted'),
                    }
    return my_result, skl_result


def grid_search():
    """ 
    This function executes model testing with different data, 
    different train/test sizes and different hyperparameters 
    """
    
    # loading Iris dataset
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    # loading Adult dataset
    X_adult, y_adult = get_adult_dataset()
    # loading Wine Quality dataset
    X_wine, y_wine = get_wine_dataset()
    
    # aggregate datasets
    datasets = [(X_adult, y_adult, 'Adult', X_adult.shape[0]*X_adult.shape[1]), 
                (X_wine, y_wine, 'Wine Quality', X_wine.shape[0]*X_wine.shape[1]), 
                (X_iris, y_iris, 'Iris', X_iris.shape[0]*X_iris.shape[1])]
    
    # aggregate train/test sizes
    test_sizes = [0.1, 0.15, 0.2]
    
    # aggregate different hyperparameter settings
    # here we change max_depth and min_samples_split 

    hyperparameters = {
        'max_depth' : [2, 5, 10, 100, 1000], 
        'min_samples_split' : [2, 10, 100]
    }
    
    my_result, skl_result = test_models(datasets, test_sizes, hyperparameters)
    return my_result, skl_result

def create_df(results):
    """ 
    This function receives the results in a dictionary form 
    and returns dataframe
    """

    df = {'dataset' : [], 'test_size' : [], 'max_depth' : [], 'min_samples_split' : [], 
            'sample_size' : [], 'training_time' : [], 'prediction_time' : [], 'accuracy' : [], 'precision' : [],
            'recall' : [], 'f1' : []}
    
    # iterating through the results 
    for dataset, values in results.items():
        for hyperparameters, aspects in values.items():
            df['dataset'].append(dataset)
            df['test_size'].append(hyperparameters[0])
            df['max_depth'].append(hyperparameters[1])
            df['min_samples_split'].append(hyperparameters[2])
            df['sample_size'].append(aspects['sample_size'])
            df['training_time'].append(aspects['training_time'])
            df['prediction_time'].append(aspects['prediction_time'])
            df['accuracy'].append(aspects['accuracy'])
            df['precision'].append(aspects['precision'])
            df['recall'].append(aspects['recall'])
            df['f1'].append(aspects['f1'])
            
    return pd.DataFrame(df)
    
""" Comment the lines below when using pytest"""

X_adult, y_adult = get_adult_dataset()
X_wine, y_wine = get_wine_dataset()


my_result, skl_result = grid_search()

compare_data_size(X_wine, y_wine)

my_result_df = create_df(my_result)
skl_result_df = create_df(skl_result)

my_result_df.to_csv("my_result.csv")
skl_result_df.to_csv("skl_result.csv")

"""Worth noting that enncodinng makes our classifier faster"""