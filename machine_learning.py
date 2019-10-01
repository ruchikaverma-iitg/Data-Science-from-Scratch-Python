# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 04:02:12 2019

@author: Ruchika
"""

# Machine learning

"""
##############################################################################################################
##########################              Data split into train and test set          ##########################
##############################################################################################################
"""

import random
from typing import TypeVar, List, Tuple
X = TypeVar('X') #Generic type to represent a datapoint

def split_data(data: List[X], prob: float) -> Tuple [List[X], List[X]]:
        "Split data into fractions [prob, 1-prob]"
        data = data[:] #Make a shallow copy
        random.shuffle(data) #Shuffle modifies the list
        cut = int(len(data)*prob) #Use prob to find cutoff
        return data[:cut], data[cut:]
    
# Data creation and splitting into train (0.75) and test (0.25) variable
data = [n for n in range(1000)]
train,test = split_data(data, 0.75)   

print("Number of points in training =",len(train))
print("Number of points in testing =",len(test))

"""
##############################################################################################################
##########################             For paired input and output variables        ##########################
##############################################################################################################
"""
Y = TypeVar('Y') #Generic type to represent output variables

def train_test_split(xs: List[X], 
                     ys: List[Y],
                     test_pct: float) -> Tuple [List[X], List[X], List[Y], List[Y]]:
    # Generate the indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1-test_pct)
    
    return ([xs[i] for i in train_idxs], # x_train
            [xs[i] for i in test_idxs],  # x_test
            [ys[i] for i in train_idxs], # y_train
            [ys[i] for i in test_idxs])  # y_test

# Testing the code
xs = [x for x in range(1000)] #xs are 1.....1000
ys = [2*x for x in xs] # each y_i is twice of x_i

x_train,x_test,y_train,y_test = train_test_split(xs, ys, 0.25)

print("Number of points in training input =",len(x_train))
print("Number of points in training output =", len(y_train))
print("Number of points in testing input =",len(x_test))
print("Number of points in testing output =", len(y_test))


# Check data proportions are correct
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test)  == 250

# Check that the corresponding dataoints are paired correctly
assert all(y == 2*x for x,y in zip(x_train, y_train))
assert all(y == 2*x for x,y in zip(x_test, y_test))


"""
##############################################################################################################
##########################                   Model evaluation metric                ##########################
##############################################################################################################
"""

def accuracy (tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp+tn
    total = tp+fp+tn+fn
    return correct/total

print("Accuracy of (70, 4930, 13930, 981070) is ", accuracy(70, 4930, 13930, 981070))

# It seems high accuracy but it may not be a good metric to quantify testing performance of a model

def precision (tp: int, fp: int, fn: int, tn: int) -> float:
    return tp/(tp+fp)

def recall (tp: int, fp: int, fn: int, tn: int) -> float:
    return tp/(tp+fn)

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall (tp, fp, fn, tn)
    return 2*p*r/(p+r)
