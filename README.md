# Decision Tree Implementation and Comparison

## Overview

This project implements a decision tree classifier from scratch and compares its performance with the decision tree implementation in the popular Scikit-learn (Sklearn) library. The project aims to explore both the machine learning and computational aspects of decision trees through the use of multiple datasets with varying sizes. It evaluates the models’ performance in terms of accuracy, precision, recall, F1-score, training time, and prediction time, while experimenting with different hyperparameters. A statistical analysis is conducted to determine if there is a significant difference between the models' results.

## Objectives

- **Build a decision tree from scratch** in Python and compare it with the Sklearn implementation.
- **Compare the models' performance** across different datasets and under various hyperparameter settings.
- **Analyze computational aspects** including training and prediction time.
- **Use statistical techniques** such as regression analysis and t-tests to compare results.
  
## Datasets

Three datasets were used to test the models:

1. **Iris**: A small, simple dataset consisting of real-number features. Ideal for classification tasks.
2. **Adult**: A larger dataset featuring both categorical and numerical data, used to predict income levels (greater than or less than 50k).
3. **Wine Quality**: A dataset of wine quality ratings with numerical features, suffering from imbalanced classes.

## Methodology

### Decision Tree Algorithm

The decision tree algorithm splits the dataset based on the highest information gain at each step. The purity of each split is calculated using either the **Gini index** or **entropy**, and the tree continues to expand until one of the following conditions is met:
- The node is pure (contains only one class).
- The maximum depth of the tree is reached.
- There are fewer than a specified number of samples to split further.
- The information gain from splitting is below a specified threshold.

### Hyperparameters

The following hyperparameters control the decision tree's growth:
- **max_depth**: Limits the depth of the tree.
- **min_samples_split**: Minimum number of samples required to split a node.
- **min_information_gain**: Minimum information gain required for further splitting.

### Grid Search & Model Comparison

Grid search is employed to evaluate the models with different combinations of hyperparameters. The performance metrics (training time, prediction time, accuracy, precision, recall, and F1-score) are recorded and compared between the custom and Sklearn decision tree models.

## Results

The results include:
- **Training and prediction times** for both models under different data sizes.
- **Performance metrics** (accuracy, precision, recall, and F1-score) for each model.
- **Statistical comparison** using regression analysis to relate training times to various factors and t-tests to evaluate significant differences between the models.
