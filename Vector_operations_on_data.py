# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:42:39 2019

@author: Ruchika
"""

from typing import List

#List of numbers as vectors
Vector = List[float]

#3-dimensional vector
height_weight_age = [70, #inches
                     170, #pounds,
                     40] #years

print(height_weight_age)

#4-dimensional vector
grades = [95, #exam1
          80, #exam2
          75, #exam3
          62] #exam4
print(grades)

######################################################################################
########################### Adding two vectors v & w #################################
######################################################################################

def add(v: Vector, w: Vector)-> Vector:
    assert len(v) == len(w) #vectors must be the same length
    return [v_i + w_i for v_i,w_i in zip(v,w)]
    
#Check if the function works correctly    
assert add([1,2,3],[4,5,6])==[5,7,9]

#Call function
print(add([1,2,3],[4,5,6]))

######################################################################################
#######################  Subtract two vectors: subtract w from v #####################
######################################################################################

def subtract(v: Vector, w: Vector)-> Vector:
    assert len(v) == len(w) #vectors must be the same length
    return [v_i - w_i for v_i,w_i in zip(v,w)]#Subtract only the corresponding elemments

#Call function    
print(subtract([5,7,9],[4,5,6]))

######################################################################################
########################## Adds all corresponding elements ###########################
######################################################################################


def vector_sum(vectors: List[Vector])-> Vector:
    #Check vectors are not empty
    assert vectors, "no vectors provided"
    
     #Check all vectors are of the same size
    num_elements = len(vectors[0]) #Number of columns
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    #The ith vector of sum is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]
    
 
assert vector_sum([[1,2],[3,4],[7,8],[5,6]])== [16,20]

#Call function    
print(vector_sum([[1,2],[3,4],[7,8],[5,6]]))

######################################################################################
######################### Multiplication of a scalar with vector #####################
######################################################################################

def scalar_multiply(c:float, v:Vector) -> Vector:
    #Multiply every element by c
    return [c*v_i for v_i in v]

assert scalar_multiply(2, [1,2,3])==[2,4,6]

print(scalar_multiply(2, [1,2,3]))

######################################################################################
########################### Componentwise mean of a vector ###########################
######################################################################################

def vector_mean(vectors:List[Vector])-> Vector:
    #Computes the element wise average
    n = len(vectors)
    return scalar_multiply(1/n,vector_sum(vectors))

#Call function  
print(vector_mean([[1,2],[3,4],[5,6]]))

######################################################################################
######################################## Dot product  ################################
######################################################################################

def dot(v:Vector, w:Vector)-> float:
    #Computes v_1*w_1+ ......+v_n*w_n
    assert len(v) == len(w), "vectors must be of same length"
    return sum(v_i * w_i for v_i, w_i in zip (v,w))

# Call function  
print(dot ([1,2,3],[4,5,6]))
    

######################################################################################
############################  Vectors sum of squares #################################
######################################################################################

def sum_of_squares(v:Vector)-> float:
    #Computes v_1*w_1+ ......+v_n*w_n
    return dot(v,v)

#Call function  
print(sum_of_squares([1,2,3]))

######################################################################################
############################ Magnitude of a vector  #################################
######################################################################################
    
import math
def magnitude(v:Vector) -> float:
    #Returns the magnitude (or length) of v
    return math.sqrt(sum_of_squares(v))

#Call function  
print(magnitude([1,2,3]))

######################################################################################
######################### Squared distance between two vectors  ######################
######################################################################################

def squared_distance(v:Vector, w:Vector) -> float:
    #Computes (v_1 - w_1)**2 +.....+(v_n - w_n)**2
    return sum_of_squares(subtract(v,w))

#Call function  
print(squared_distance([1,2,3],[4,5,6]))

######################################################################################
######################### L2 distance between two vectors  #############################
######################################################################################

def distance(v:Vector, w:Vector) -> float:
    #Computes the distance between v and w
    return math.sqrt(sum_of_squares(subtract(v,w)))

# Call function  
print(distance([1,2,3],[4,5,6]))
