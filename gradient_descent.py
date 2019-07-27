# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:28:10 2019

@author: Ruchika
"""

from typing import List
import random
from Vector_operations_on_data import distance,add,scalar_multiply,vector_mean #Importing functions from the previous code
from typing import Callable

Vector = List[float]

"""
##############################################################################################################
##########################    Calling function if f is a function of one variable   ##########################
##############################################################################################################
"""
# Callable function : square
def square(x:float) -> float:
    return x*x

# Derivative of a square function
def derivative(x:float) -> float:
    return 2*x

# Estimating solution of a derivative function
def difference_quotient(f: Callable[[float],float], x: float,h: float)-> float:
    return (f(x+h) -f(x))/h

xs = [x for x in range(-10,11)]
actuals = [derivative(x) for x in xs] # Get actual values in xs using derivative function
print(xs)
print(actuals)

# Get estimates of the derivative function using difference_quotient
estimates = [difference_quotient(square, x,h = 0.001) for x in xs]
print(estimates)

#Plotting to see if actual derivative values of a square function matches with the estimated values
import matplotlib.pyplot as plt
plt.title(" Actual derivatives vs. Estimates")
plt.plot(xs,actuals,'rx',label = 'Actual')
plt.plot(xs,estimates,'b+',label = 'Estimate')
plt.legend(loc = 9)
plt.show()

"""
##############################################################################################################
##########################    Calling function if f is a function of many variables ##########################
##############################################################################################################
"""
# Partial derivatives, f is a function of many variables
def partial_difference_quotient(f: Callable[[Vector],float],v:Vector, i:int,h:float)-> float:
    #Returns the ith partial difference quotient of f at v
    w = [v_j + (h if j ==i else 0) 
        for j,v_j in enumerate(v)]
    
    return (f(w)-f(v))/h

def estimate_gradient(f: Callable[[Vector],float],v:Vector,h:float = 0.0001)-> float:
    return [partial_difference_quotient(f,v,i,h)
           for i in range(len(v))]
    
def gradient_step (v:Vector, gradient:Vector,step_size:float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size,gradient)
    return add(v,step)

def sum_of_squares_gradient(v:Vector)-> Vector:
    return [2*v_i for v_i in v]

# Pick a random starting point to find minimum in a 3D vector
# Take tiny steps in the opposite of gradient until we reach a point where gradient is very small
v = [random.uniform(-10,10) for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v)
    v = gradient_step(v,grad,-0.01)
    print(epoch,v)
    
assert distance(v,[0,0,0])<0.001 # v should be close to 0

"""
##############################################################################################################
##########################            Use gradient descent to fit models            ##########################
##############################################################################################################
"""
# x ranges from -50 to 49, y is always 20*x+5
inputs = [(x,20*x+5) for x in range(-50,50)]
print(inputs)

def linear_gradient(x:float,y:float,theta:Vector)-> Vector:
    slope,intercept = theta
    predicted = slope*x+intercept # The prediction of the model
    error = (predicted - y) #error is predicted - actual
    squared_error = error**2 #Minime the squared error
    grad = [2*error*x,2*error]
    return grad

# Start with random values for slope and intercept
theta = [random.uniform(-1,1), random.uniform(-1,1)]

learning_rate = 0.001
for epoch in range(5000):
    #Compute the mean of the gradients
    grad = vector_mean([linear_gradient(x,y,theta) for x,y in inputs])
    # Take a step in that direction
    theta = gradient_step(theta, grad,-learning_rate)
    print(epoch, theta)
    
slope, intercept = theta
assert 19.9 < slope < 20.1 # Slope should be around 20
assert 4.9 < intercept < 5.1 # Intercept should be around 5

"""
##############################################################################################################
################# Split data into mini-batches and use gradient descent to fit models ########################
##############################################################################################################
"""

from typing import TypeVar, List, Iterator
T = TypeVar('T')

def minibatches(dataset:List[T],
               batch_size = int,
               shuffle:bool = True) -> Iterator[List[T]]:
    # Generate minibatches of batchsize from the datset
    #Index starts from 0
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    
    if shuffle:random.shuffle(batch_starts) #Shuffle the batches
    
    for start in batch_starts:
        end = start+batch_size
        yield dataset[start:end]
        
theta = [random.uniform(-1,1), random.uniform(-1,1)]

for epoch in range(1000):
    for batch in minibatches(inputs, batch_size = 20):
        grad = vector_mean([linear_gradient(x,y,theta) for x,y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print (epoch,theta)
    
slope, intercept = theta
assert 19.9 < slope < 20.1 # Slope should be around 20
assert 4.9 < intercept < 5.1 # Intercept should be around 5


"""
##############################################################################################################
####    Use Stochastic gradient descent in which you take gradient steps based on one training example     ###
##############################################################################################################
"""

theta = [random.uniform(-1,1), random.uniform(-1,1)]

for epoch in range(1000):
    for x,y in inputs:
        grad = linear_gradient(x,y,theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print (epoch,theta)
    
slope, intercept = theta
assert 19.9 < slope < 20.1 # Slope should be around 20
assert 4.9 < intercept < 5.1 # Intercept should be around 5

        


