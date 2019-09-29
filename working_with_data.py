# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:15:17 2019

@author: Ruchika
"""

"""
##############################################################################################################
##########################        Create a histogram of 1D data                  ##########################
##############################################################################################################
"""
from typing import List, Dict
from collections import Counter
import math
import matplotlib.pyplot as plt


def bucketize(point: float, bucket_size: float) -> float:
    #Floor the point to the next lower multiple of bucket size
    return bucket_size*math.floor(point/bucket_size)

def make_histogram(points:List[float], bucket_size: float) -> Dict[float,int]:
    #Buckets the points and counts how many in each bucket
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram (points:List[float], bucket_size: float, title: str = ""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width = bucket_size)
    plt.title(title)
    
    
"""
##############################################################################################################
##########################  Histogram plotting with data of different distributions ##########################
##############################################################################################################
"""
    
# Data
import random
from Probability import inverse_normal_cdf   

random.seed(0)

# Data 1
#uniform between -100 and 100
uniform = [200*random.random() -100 for _ in range(10000)]

#Data 2
#normal distrinution with mean 0, standard deviation 57
normal = [57*inverse_normal_cdf(random.random())
         for _ in range(10000)]

# Plot Data 1
plot_histogram(uniform, 10, "Uniform Histogram")

# Plot Data 2
plot_histogram(normal, 10, "Normal Histogram") 