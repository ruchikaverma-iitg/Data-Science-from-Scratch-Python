# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:51:31 2019

@author: Ruchika
"""


"""
####################################################################################################
##########################         Define least square fit function       ##########################
####################################################################################################
"""

def predict (alpha: float, beta: float, x_i: float) -> float:
    return beta*x_i+alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    # The error from predicting beta*x_i+alpha and y_i
    return predict(alpha, beta, x_i) - y_i

from Vector_operations_on_data import Vector
def sum_of_sqerrors (alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i)**2 
               for x_i, y_i in zip(x,y))

from typing import Tuple
from Statistics import correlation, standard_deviation, mean

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float,float]:
    """Given two vectors x and y,
    find the least-squares value of alpha and beta"""
    beta = correlation(x,y)*standard_deviation(y)/standard_deviation(x)
    alpha = mean(y) - beta*mean(x)
    #print(alpha, beta)
    return alpha, beta

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

least_squares_fit(x,y)

"""
####################################################################################################
######################### Evaluate alpha and beta on the previously used data ######################
####################################################################################################
"""

from Statistics import num_friends_good, daily_minutes_good
alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
alpha, beta

# Find coefficient of determination or R squared
from typing import List
def de_mean(xs: List[float]) -> List[float]:
    """Translate xs by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def total_sum_of_squares(y: Vector) -> float:
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """The fraction of variation in y captured by the model,
    which equals 1-the fraction not captured by the model"""
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y)
               /total_sum_of_squares(y))

print(r_squared(alpha, beta, num_friends_good, daily_minutes_good))

print("This tells how better we fit the model")
print("The higher the number, the better our model fits the data")

"""
####################################################################################################
########################   Solve the above problem using gradient descent     ######################
####################################################################################################
"""
import random 
import tqdm
from gradient_descent import gradient_step

num_epochs = 10000
random.seed(0)

guess = [random.random(), random.random()] # Choose random value to start

learning_rate = 0.00001

with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess # initial guess
        
        # partial derivative of loss wrt alpha
        grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                    for x_i, y_i in zip(num_friends_good, daily_minutes_good))
        
        # partial derivative of loss wrt beta
        grad_b = sum(2*error(alpha, beta, x_i, y_i)*x_i
                    for x_i, y_i in zip(num_friends_good, daily_minutes_good))
        
        # computes loss to stick tqdm description
        loss = sum_of_sqerrors(alpha, beta,
                              num_friends_good, daily_minutes_good)
        t.set_description(f"loss: {loss:3f}")
        
        # Finally update the guess
        guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
guess