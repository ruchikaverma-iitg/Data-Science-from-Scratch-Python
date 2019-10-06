# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:54:56 2019

@author: Ruchika
"""

"""
####################################################################################################
##########################            Supporting functions                ##########################
####################################################################################################
"""

from Vector_operations_on_data import dot, Vector;

# X: List of vectors [1,XX1,XX2,XX3]
# y_hat  = alpha*1+beta1*XX1+beta2*XX2+....
def predict(x:Vector, beta: Vector)-> float:
    "Assumes that first element of x is 1 (to add bias coefficient)"
    return dot(x,beta)

# Assumptions: all features are independent and uncorrelated

from typing import List
def error(x:Vector, y:float, beta:Vector) -> float:
    return predict(x,beta)-y

def squared_error(x:Vector,y:float,beta:Vector)-> float:
    return error(x,y,beta)**2

x=[1,2,3]
y = 30
beta = [4,4,4]
print(error(x,y,beta))
print(squared_error(x,y,beta)) 

def sqerror_gradient(x:Vector,y:float,beta:Vector) -> Vector:
    err = error(x,y,beta)
    return [2*err*x_i for x_i in x]

print(sqerror_gradient(x,y,beta))   

"""
####################################################################################################
##########################         Define least square fit function       ##########################
####################################################################################################
"""

import random
import tqdm
import numpy as np
from Vector_operations_on_data import vector_mean;
from gradient_descent import gradient_step;

def least_squares_fit(xs:List[Vector],
                      ys: List[Vector],
                      learning_rate: float = 0.001,
                      num_steps: int = 1000,
                      batch_size: int = 1) -> Vector:
    """Find beta that minimizes the sum of squared errors
    assuming the model y = dot(x,beta)"""
    
    #Start with a random guess
    guess = [random.random() for _ in xs[0]]
    
    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]
            
            gradient =vector_mean([sqerror_gradient(x,y,guess)
                                  for x,y in zip(batch_xs, batch_ys)])

            guess = gradient_step(guess, gradient,-learning_rate)
            
    return guess

# Data
from Statistics import daily_minutes_good;

inputs: List[List[float]] = [[1.,49,4,0],[1,41,9,0],[1,40,8,0],[1,25,6,0],[1,21,1,0],[1,21,0,0],[1,19,3,0],[1,19,0,0],[1,18,9,0],[1,18,8,0],[1,16,4,0],[1,15,3,0],[1,15,0,0],[1,15,2,0],[1,15,7,0],[1,14,0,0],[1,14,1,0],[1,13,1,0],[1,13,7,0],[1,13,4,0],[1,13,2,0],[1,12,5,0],[1,12,0,0],[1,11,9,0],[1,10,9,0],[1,10,1,0],[1,10,1,0],[1,10,7,0],[1,10,9,0],[1,10,1,0],[1,10,6,0],[1,10,6,0],[1,10,8,0],[1,10,10,0],[1,10,6,0],[1,10,0,0],[1,10,5,0],[1,10,3,0],[1,10,4,0],[1,9,9,0],[1,9,9,0],[1,9,0,0],[1,9,0,0],[1,9,6,0],[1,9,10,0],[1,9,8,0],[1,9,5,0],[1,9,2,0],[1,9,9,0],[1,9,10,0],[1,9,7,0],[1,9,2,0],[1,9,0,0],[1,9,4,0],[1,9,6,0],[1,9,4,0],[1,9,7,0],[1,8,3,0],[1,8,2,0],[1,8,4,0],[1,8,9,0],[1,8,2,0],[1,8,3,0],[1,8,5,0],[1,8,8,0],[1,8,0,0],[1,8,9,0],[1,8,10,0],[1,8,5,0],[1,8,5,0],[1,7,5,0],[1,7,5,0],[1,7,0,0],[1,7,2,0],[1,7,8,0],[1,7,10,0],[1,7,5,0],[1,7,3,0],[1,7,3,0],[1,7,6,0],[1,7,7,0],[1,7,7,0],[1,7,9,0],[1,7,3,0],[1,7,8,0],[1,6,4,0],[1,6,6,0],[1,6,4,0],[1,6,9,0],[1,6,0,0],[1,6,1,0],[1,6,4,0],[1,6,1,0],[1,6,0,0],[1,6,7,0],[1,6,0,0],[1,6,8,0],[1,6,4,0],[1,6,2,1],[1,6,1,1],[1,6,3,1],[1,6,6,1],[1,6,4,1],[1,6,4,1],[1,6,1,1],[1,6,3,1],[1,6,4,1],[1,5,1,1],[1,5,9,1],[1,5,4,1],[1,5,6,1],[1,5,4,1],[1,5,4,1],[1,5,10,1],[1,5,5,1],[1,5,2,1],[1,5,4,1],[1,5,4,1],[1,5,9,1],[1,5,3,1],[1,5,10,1],[1,5,2,1],[1,5,2,1],[1,5,9,1],[1,4,8,1],[1,4,6,1],[1,4,0,1],[1,4,10,1],[1,4,5,1],[1,4,10,1],[1,4,9,1],[1,4,1,1],[1,4,4,1],[1,4,4,1],[1,4,0,1],[1,4,3,1],[1,4,1,1],[1,4,3,1],[1,4,2,1],[1,4,4,1],[1,4,4,1],[1,4,8,1],[1,4,2,1],[1,4,4,1],[1,3,2,1],[1,3,6,1],[1,3,4,1],[1,3,7,1],[1,3,4,1],[1,3,1,1],[1,3,10,1],[1,3,3,1],[1,3,4,1],[1,3,7,1],[1,3,5,1],[1,3,6,1],[1,3,1,1],[1,3,6,1],[1,3,10,1],[1,3,2,1],[1,3,4,1],[1,3,2,1],[1,3,1,1],[1,3,5,1],[1,2,4,1],[1,2,2,1],[1,2,8,1],[1,2,3,1],[1,2,1,1],[1,2,9,1],[1,2,10,1],[1,2,9,1],[1,2,4,1],[1,2,5,1],[1,2,0,1],[1,2,9,1],[1,2,9,1],[1,2,0,1],[1,2,1,1],[1,2,1,1],[1,2,4,1],[1,1,0,1],[1,1,2,1],[1,1,2,1],[1,1,5,1],[1,1,3,1],[1,1,10,1],[1,1,6,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,4,1],[1,1,9,1],[1,1,9,1],[1,1,4,1],[1,1,2,1],[1,1,9,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,1,1],[1,1,1,1],[1,1,5,1]]

"""
####################################################################################################
#########################          Evaluate beta on the previously used data  ######################
####################################################################################################
"""

random.seed(0)
learning_rate =0.001

beta = least_squares_fit(inputs,daily_minutes_good, learning_rate, 5000,25)

"""
####################################################################################################
#########################                  Compute R squared                  ######################
####################################################################################################
"""

from linear_regression import total_sum_of_squares;
def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
    sum_of_squared_errors = sum(squared_error(x,y,beta)
                               for x,y in zip(xs,ys))
    return 1.0 - sum_of_squared_errors/total_sum_of_squares(ys)

print(multiple_r_squared(inputs,daily_minutes_good,beta))
 
"""
####################################################################################################
#########################    Compute a statistic on data using bootstrapping  ######################
####################################################################################################
"""       
# Compute a statistic on data using bootstrapping
# Size of the sample for bootstrapping is same as len(data) but with replacement

from typing import TypeVar, List, Callable

X = TypeVar('X') # Generic type for data
Stat = TypeVar('Stat') # Generic type for statistic

def bootstrap_sample(data: List[X]) -> List[X]:
    """Randomly samples len(data) elements with replacement"""
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data: List[X],
                       stats_fn: Callable[[List[X]], Stat],
                       num_samples: int) -> List[Stat]:
    """Evaluates stats_fn on num_samples bootstrap samples from data"""
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

import random

# Data1: Sample 101 points all very close to 100
close_to_100 = [99.5 + random.random() for _ in range(101)]

# Data2: Sample 101 points, 50 of them near 0 and 50 of them near 100
far_from_100 = ([99.5 + random.random()] +
                [random.random() for _ in range(50)] +
                [200 + random.random() for _ in range(50)])

# Compare median value of Data1 and Data2
from Statistics import median, standard_deviation, mean;

medians_close = bootstrap_statistic(close_to_100, median, 100)
medians_far = bootstrap_statistic(far_from_100, median, 100)
print(f"medians_close = {medians_close}\n")

print(f"medians_far = {medians_far}")

std_close = standard_deviation(medians_close)
std_far = standard_deviation(medians_far)

print(f"std_medians_close = {std_close}")
print(f"std_medians_far = {std_far}")

mean_close = mean(medians_close)
mean_far = mean(medians_far)

print(f"mean_medians_close = {mean_close}")
print(f"mean_medians_far = {mean_far}")

"""
####################################################################################################
#########################      Estimate sample beta using bootstrapping        ######################
####################################################################################################
"""     

from typing import Tuple
import datetime

def estimate_sample_beta(pairs: List[Tuple[Vector, float]]) -> Vector:
    x_sample = [x for x,_ in pairs]
    y_sample = [y for _, y in pairs]
    
    beta = least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
    print("Bootstrap sample", beta)
    return beta

bootstrap_betas = bootstrap_statistic(list(zip(inputs, daily_minutes_good)),
                                     estimate_sample_beta, 100)

"""
####################################################################################################
#########################   Compute statistics on bootstrapped coefficients(betas)   ###############
####################################################################################################
"""    

bootstrap_standard_errors = [
    standard_deviation([beta[i] for beta in bootstrap_betas]) 
    for i in range(4)]

print(bootstrap_standard_errors)

from Probability import normal_cdf;

def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
    if beta_hat_j > 0:
        #if the coefficient is positive, we need to compute
        #twice the probability of seeing an even larger value"""
        return 2*(1-normal_cdf(beta_hat_j/sigma_hat_j))
        #"""Otherwise twice the probability of a smaller value"""
    else:
        return 2*(normal_cdf(beta_hat_j/sigma_hat_j))
    
p_value(0.923,1.24)

"""
####################################################################################################
##########################################  Regularization   #######################################
####################################################################################################
"""
# alpha is a coefficient which decides how harsh the penalty is
#L2 norm
def ridge_penalty(beta: Vector,
                 alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x: Vector,
                       y: float,
                       alpha: float) -> float:
    """estimate error plus ridge penalty"""
    return error(x,y,beta)**2 + ridge_penalty(beta, alpha)

from Vector_operations_on_data import add
def ridge_penality_gradient(beta: Vector, alpha: float) -> float:
    """gradient of just ridge penality"""
    return [0.] + [2*alpha*beta_j for beta_j in beta[1:]]

def sqerror_ridge_gradient(x: Vector,
                          y: float, beta: Vector,
                          alpha: float) -> Vector:
    """gradient corresponding to the i-th squared error term
    including ridge penalty"""
    return add(sqerror_gradient(x,y,beta), 
               ridge_penality_gradient(beta, alpha))

def least_squares_fit(xs: List[Vector],
                     ys: Vector,
                     alpha: float,
                     learning_rate: float = 0.001,
                     num_steps: int = 1000,
                     batch_size: int = 1) -> Vector:
    """Finds beta that minimizes the sum of squared errors
    assuming the model dot(x, beta)"""
    # start with random guess
    guess = [random.random() for _ in xs[0]]
    
    for _ in tqdm.trange(num_steps, desc = "least squares fit"):
        for start in range(0,len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]
            
            gradient = vector_mean([sqerror_ridge_gradient(x,y,guess,alpha)
                                  for x,y in zip(batch_xs, batch_ys)])
            
            guess = gradient_step(guess, gradient, -learning_rate)
    return guess
random.seed(0)
beta_0 = least_squares_fit(inputs, daily_minutes_good, 0.0,
                          learning_rate, 5000,25)

#L1 regression
def lasso_penlty(beta,alpha):
    return alpha*sum(abs(beta_i) for beta_i in beta[1:])