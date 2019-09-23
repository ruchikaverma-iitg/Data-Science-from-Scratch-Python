# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:08:27 2019

@author: Ruchika
"""
#Represent data as Matrices

from typing import Tuple
import numpy as np
from typing import List
from Vector_operations_on_data import Vector 

Matrix = List[List[float]]
A=[[1,2,3],[4,5,6]]
B = [[1,2],[3,4],[5,6]]

from typing import Tuple
import numpy as np

######################################################################################
################# find shape of a matrix without using default function ##############
######################################################################################

def shape(A:Matrix) -> Tuple[int,int]:
    #returns # of rows of A, # of columns of A
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 #Number of elements in first row
    return num_rows, num_cols

#Call function
print(shape([[1,2,3],[4,5,6]])) # 2 rows, 3 columns

######################################################################################
############################## Get a particular row of a matrix ######################
######################################################################################

def get_row(A:Matrix,i:int)-> Vector:
    #Returns the ith row of A (as a Vector)
    return A[i]
 
#Call function
print(get_row(A,1))  

######################################################################################
####################### Extract a particular column of a matrix ######################
######################################################################################

def get_column(A:Matrix,j:int)-> Vector:
    #Returns the ith row of A (as a Vector)
    return [A_i[j]
           for A_i in A]#For each row A_i

#Call function
print(get_column(A,1))

######################################################################################
######### Make a matrix with user input num_rows and num_cols with ###################
#######################  condition defined in the entry_fn ###########################
######################################################################################

from typing import Callable
    
def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int,int], float]) -> Matrix:
    # Returns a num_rows * num_cols matrix
    # whose (i,j)th entry is entry_fn(i,j)
    return [[entry_fn(i,j) 
             for j in range(num_cols)]
           for i in range(num_rows)]

n = 5        
zero_matrix = np.array(make_matrix(n,n, lambda i, j: 0))
#Call function
print(zero_matrix)
  
######################################################################################
######### Create an identity matrix using make_matrix function ### ###################
######################################################################################
      
def identity_matrix(n:int)-> Matrix:
    "Returns n*n identity matri"
    return make_matrix(n,n, lambda i, j: 1 if i == j else 0)

#Call function
print(identity_matrix(5))

######################################################################################
###############        Create a friendship matrix when indices  ######################
###############    of friendship pairs are available in friendships ##################
######################################################################################
import numpy as np
friendships = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,8),(8,9)]
n = max(max(friendships))
frienship_matrix = np.array(make_matrix(n,n, lambda i, j: 0))
for i,j in friendships:
    frienship_matrix[i-1,j-1] = 1
    frienship_matrix[j-1,i-1] = 1

print(frienship_matrix)        
        