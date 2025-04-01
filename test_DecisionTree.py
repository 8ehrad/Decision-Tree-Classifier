#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 22:17:42 2023

@author: 13ehrad
"""

import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from DecisionTree import DecisionTree, gini, entropy

def test_DecisionTree():
    """ Compare the prediction of our moodel with the true labels """
    # loading the Iris dataset to test on
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=0)
    
    
    dt = DecisionTree()
    dt.train(X_train, y_train)

    y_pred = dt.predict(X_test)
    
    assert (y_test == y_pred).all()

def test_gini():
    """ Test gini index  for a given set of labels and check if the result 
        matches with what we know is the true answer
    """
    y = [0, 1, 1, 0]
    assert gini(y) == 1/2

def test_entropy():
    y = [0, 1, 1, 0]
    assert entropy(y) == 1
    
""" All other functions are involved and tested in the first test function """