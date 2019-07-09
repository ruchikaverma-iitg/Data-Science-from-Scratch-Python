# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 07:07:00 2019

@author: Ruchika
"""
######################################################################################
########################### Binomial distribution   #################################
######################################################################################

from typing import Tuple
import math

def normal_approximation_to_binomial(n:int,p:float)-> Tuple[float,float]:
    #Returns mu and sigma corresponding to a Binomial (n,p)
    mu = n*p
    sigma = math.sqrt(p*(1-p)*n)
    return mu, sigma

######################################################################################
#Finding whether probability of a value lies within or outside a particular interval #
######################################################################################
    
from Probability import normal_cdf

# Normal cdf is the probability that the variable is below the threshold
normal_probability_below = normal_cdf

#It is above the threshold if it's not below the threshold
def normal_probability_above(lo:float,
                            mu: float=0,
                            sigma: float =1)-> float:
    # The probability that an N(mu,sigma) is greater than lo.
    return 1-normal_cdf(lo, mu, sigma)

#It is between if it's less than hi but above lo
def normal_probability_between(lo:float,
                               hi: float,
                               mu: float=0,
                               sigma: float =1)-> float:
    # The probability that an N(mu,sigma) is between lo and hi.
    return normal_cdf(hi, mu, sigma)-normal_cdf(lo, mu, sigma)

#It's outside if not between
def normal_probability_outside(lo:float,
                              hi:float,
                              mu:float = 0,
                              sigma: float = 1) -> float:
     # The probability that an N(mu,sigma) is not between lo and hi.
        return 1-normal_probability_between(lo,hi, mu, sigma)
 
######################################################################################
#####################   Find z when P(Z<=z) or P(Z>=z) or P(Z~=z)    #################
######################################################################################

from Probability import inverse_normal_cdf

def normal_upper_bound(probability:float,
                      mu:float = 0,
                      sigma: float = 1)-> float:
    # Returns the z for which P(Z<=z) = probability
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability:float,
                      mu:float = 0,
                      sigma: float = 1)-> float:
    # Returns the z for which P(Z>=z) = probability
    return inverse_normal_cdf(1-probability, mu, sigma)

def normal_two_sided_bounds(probability:float,
                      mu:float = 0,
                      sigma: float = 1)-> Tuple[float,float]:
    #Returns the symmetric (about) the means) bounds that contain the specified probability
    tail_probability = (1-probability)/2
    
    #Upper bound should have tail probability above it
    upper_bound = normal_lower_bound(tail_probability,mu,sigma)
    
    #Lower bound should have tail probability below it
    lower_bound = normal_upper_bound(tail_probability,mu,sigma)
    
    return lower_bound, upper_bound   
  
# Flipping a coin 1000 times. Let's suppose that the coin is fair and have probability = 0.5

mu_0,sigma_0 = normal_approximation_to_binomial(1000,0.5)
print(mu_0,sigma_0) 

# Make a decision about significance
# Let's assume that the there are 5% chances of Type I error (False positives) in which we reject H_0 hypothesis

# Consider the test that rejects H_0 if X falls outside the bounds
lower_bound, upper_bound = normal_two_sided_bounds(0.95,mu_0,sigma_0)
print(lower_bound, upper_bound) 

# We are often interested in the power of a test
# Prob of not making a Type II error (False negatives), in  which we fail to reject H_0 even though it's false

#95% bounds based on assumption p is 0.5
lo, hi = normal_two_sided_bounds(0.95,mu_0,sigma_0)

#actual mu and sigma based on p is 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000,0.55)

# a Type II error means we fail to reject null hypothesis
# which will happen when X is still in our original interval
type_2_probability = normal_probability_between(lo,hi,mu_1,sigma_1)

power = 1-type_2_probability
print(power)

# Imagine if coin is not fair and X (N of heads) is much larger than 500 but not X <=500
# 5% significance test using normal_probability_below to find the cut_off belowwhich 95% of the probability lies

hi = normal_upper_bound(0.95,mu_0,sigma_0)

type_2_probability = normal_probability_below(hi,mu_1,sigma_1)

power = 1-type_2_probability
print(power)

######################################################################################
####################################   Hypothesis testing   ##########################
######################################################################################
# p-values
def two_sided_p_value(x:float, mu:float=0, sigma:float = 1)-> float:
    # How likely are we to see a value at least as extreme as x (in either direction) if our values are from an N(mu,sigma)
    
    if x >= mu:
        # If x is greater than the mean, so the tail is everything greater than x
        return 2*normal_probability_above(x, mu, sigma)
    else:
        # If x is less than the mean, so the tail is everything less than x
        return 2*normal_probability_below(x, mu, sigma)

# If we see 530 heads    
print(two_sided_p_value(529.5, mu_0, sigma_0)) 

import random

extreme_value_count = 0
for _ in range(1000):
    # Count number of heads in 1000 flips
    num_heads = sum(1 if random.random() < 0.5 else 0   
                    for _ in range(1000))  
    # Count how often the # is extreme
    if num_heads >= 530 or num_heads <= 470:            
        extreme_value_count += 1                       

# p-value was 0.062 => ~62 extreme values out of 1000
assert 59 < extreme_value_count < 65, f"{extreme_value_count}"       

print(two_sided_p_value(531.5, mu_0, sigma_0))

tspv = two_sided_p_value(531.5, mu_0, sigma_0)
assert 0.0463 < tspv < 0.0464

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

print(upper_p_value(524.5, mu_0, sigma_0))

print(upper_p_value(526.5, mu_0, sigma_0))

######################################################################################
#################### Testing whether a coin is fair or not   #########################
######################################################################################
# Confidence intervals
#Example 1
p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)   
print(mu,sigma)

print(normal_two_sided_bounds(0.95, mu, sigma)) #As 0.5 lies in the CI, hence the coin is fair

#Example 2
p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) 
print(mu,sigma)

print(normal_two_sided_bounds(0.95, mu, sigma)) #As 0.5 doesnot lie in the CI, hence the coin is unfair

from typing import List

def run_experiment() -> List[bool]:
    # Flips a fair coin 1000 times, True = heads, False = tails
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]) -> bool:
    """Using the 5% significance levels"""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment
                      for experiment in experiments
                      if reject_fairness(experiment)])

assert num_rejections == 46


# n_A, n_B represents number of times an add got clicked out of N_A and N_B people respectively.
# where A & B are the two differentmodes of advertisement of a same ad.
# Let's see if there is a significant difference between the advertisement by A & B  

def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)


z = a_b_test_statistic(1000, 200, 1000, 180)
print(z)

# There is no significant difference if p_val>0.05
p_val = two_sided_p_value(z) 
print(p_val)

# Try with different n_B
z = a_b_test_statistic(1000, 200, 1000, 150)
print(z)

# Significant difference if p_val<0.05
p_val = two_sided_p_value(z) 
print(p_val)


def B(alpha: float, beta: float) -> float:
    """A normalizing constant so that the total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:          # no weight outside of [0, 1]
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
