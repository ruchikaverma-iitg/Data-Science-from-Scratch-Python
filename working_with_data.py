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
    plt.figure()
    plt.bar(histogram.keys(), histogram.values(), width = bucket_size)
    plt.title(title)
    plt.show()
    
    
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

"""
##############################################################################################################
##########################                     2-dimensional data                   ##########################
##############################################################################################################
"""

def random_normal() -> float:
    # Returns a random draw from a standard normal distribution
    return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [ x+random_normal()/2 for x in xs]
ys2 = [-x+random_normal()/2 for x in xs]

# Plot Data 1
plot_histogram(ys1, 10, "Normal Histogram")
# Plot Data 2
plot_histogram(ys2, 10, "Normal Histogram")

plt.figure()
plt.scatter(xs,ys1, marker = '.', color = 'black', label ='ys1')
plt.scatter(xs,ys2, marker = '.', color = 'red', label ='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc = 9)
plt.title('Very different joint distributions')
plt.show()


"""
##############################################################################################################
##########################                             Correlation                  ##########################
##############################################################################################################
"""

# Difference between ys1 and ys2 would be apparent through correlations
from Statistics import correlation
print(correlation(xs,ys1)) # about 0.9
print(correlation(xs,ys2)) # about -0.9

# corr_data is a list of four 100-d vectors
# corr_data = [[random.random() for _ in range(100)] for _ in range(4)]
num_points = 100

def random_row() -> List[float]:
    row = [0.0, 0, 0, 0]
    row[0] = random_normal()
    row[1] = -5 * row[0] + random_normal()
    row[2] = row[0] + row[1] + 5 * random_normal()
    row[3] = 6 if row[2] > -2 else 0
    return row

random.seed(0)
# each row has 4 points, but really we want the columns
corr_rows = [random_row() for _ in range(num_points)]

corr_data = [list(col) for col in zip(*corr_rows)]

num_vectors = len(corr_data)
plt.figure()
fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
    for j in range(num_vectors):
        # Scatter column_j on the x-axis vs. column on the y-axis
        if i != j: ax[i][j].scatter(corr_data[j], corr_data[i])
        # Unless i == j, in which case show the series name
        else: ax[i][j].annotate("series" + str(i), (0.5,0.5),
                               xycoords = 'axes fraction',
                               ha = "center", va = "center")
            
# Fix the bottom right and top left axis labels, which are wrong because
    # their charts only have text in them
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())
plt.show()

"""
##############################################################################################################
##########################             Dictionaries, NamedTuple and classes         ##########################
##############################################################################################################
"""
#Create dictionary
import datetime
stock_price = {'closing_price': 102.06,
               'date':datetime.date(2014,8,29),
               'symbol':'AAPL'}
print(stock_price)

# To avoid confusions and typos, use namedtuple
from collections import namedtuple

StockPrice = namedtuple('StockPrice', ['symbol','date','closing_price']) 
price = StockPrice('MSFT',datetime.date(2018,12,14), 106.03)
print(price)

# Immutable
assert price.symbol == 'MSFT'
assert price.closing_price == 10000

# Dataclasses are a mutable version of NamedTuple
from typing import NamedTuple
class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float
        
    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT','GOOG','FB','AMZN','AAPL']
    
price = StockPrice('MSFT', datetime.date(2018,12, 14), 106.3)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.3
assert price.is_high_tech()

from dateutil.parser import parse

def parse_row(row:List[str])-> StockPrice:
    symbol,date,closing_price = row
    return StockPrice(symbol = symbol,
                      date = parse(date).date(),
                      closing_price = float(closing_price))
    
#Now test the function
stock = parse_row(['MSFT',"2018-12-14", "106.03"])

# If there's a bad data. For example: a float value that doesn't actually represent a number.
# It may return None and crash the program

from typing import Optional
import re

def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date_, closing_price_ = row

    # Stock symbol should be all capital letters
    if not re.match(r"^[A-Z]+$", symbol):
        return None

    try:
        date = parse(date_).date()
    except ValueError:
        return None

    try:
        closing_price = float(closing_price_)
    except ValueError:
        return None

    return StockPrice(symbol, date, closing_price)

assert try_parse_row(['MSFT0',"2018-12-14", "106.03"]) is None
assert try_parse_row(['MSFT',"2018-12--14", "106.03"]) is None
assert try_parse_row(['MSFT',"2018-12-14", "x"]) is None

from dateutil.parser import parse
import csv

with open("stocks.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = [[row['symbol'], row['date'], row['price']]
            for row in reader]

# skip header
maybe_data = [try_parse_row(row) for row in rows]

# Make sure they all loaded successfully:
assert maybe_data
assert all(sp is not None for sp in maybe_data)

# This is just to make mypy happy
data = [sp for sp in maybe_data if sp is not None]


max_aapl_price = max(stock_price.closing_price 
                     for stock_price in data 
                     if stock_price.symbol == 'AAPL')
print(max_aapl_price)

from collections import defaultdict

max_prices: Dict[str, float] = defaultdict(lambda: float('-inf'))
    
for sp in data:
    symbol, closing_price = sp.symbol, sp.closing_price
    
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price
max_prices

from typing import List
prices: Dict[str, List[float]] = defaultdict(list)

for sp in data:
    prices[sp.symbol].append(sp)
# order (or sort) the prices by date
prices = {symbol: sorted(symbol_prices) 
          for symbol, symbol_prices in prices.items()}
print(prices)

def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return today.closing_price / yesterday.closing_price - 1

class DailyChange(NamedTuple):
    symbol: str
    date: datetime.date
    pct_change: float
    
def day_over_day_changes(prices: List[StockPrice]) -> List[DailyChange]:
    """Assumes prices are for once stock and are ordered"""
    return [DailyChange(symbol = today.symbol,
                           date = today.date,
                           pct_change = pct_change(yesterday, today))
            for yesterday, today in zip(prices, prices[1:])]

all_changes = [change for symbol_prices in prices.values()
              for change in day_over_day_changes(symbol_prices)]
max_change = max(all_changes, key = lambda change: change.pct_change)
max_change

changes_by_month: List[DailyChange] = {month: [] for month in range(1,13)}
changes_by_month

for change in all_changes:
    changes_by_month[change.date.month].append(change)
changes_by_month

avg_daily_change = {month: sum(change.pct_change for change in changes)/ len(changes)
                   for month, changes in changes_by_month.items()}

assert avg_daily_change[10] == max(avg_daily_change.values())

# When data is not of same scale then we may need to rescale the data to have mean 0 and std 1
from typing import Tuple
from Vector_operations_on_data import vector_mean
from Statistics import standard_deviation
Vector = List[float]

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """Returns mean and standard deviation of each feature"""
    dim = len(data[0])
    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data])
             for i in range(dim)]
    return means, stdevs

vectors = [[-3, -1, 1],[-1, 0, 1], [1, 1, 1], [2, 4, 5]]
means, stdevs = scale(vectors)
means, stdevs

def rescale(data:List[Vector]) -> List[Vector]:
    #Rescale the input data
    dim = len(data[0])
    
    means, stdevs = scale(data)
    
    # Make a copy of each vector
    rescaled = [v[:] for v in data]

    for v in rescaled:
        for i in range(dim):
            if stdevs[i]>0:
                v[i] = (v[i] - means[i])/stdevs[i]
                
    return rescaled

means,stdevs = scale(rescale(vectors))
means,stdevs

#Produce a progress bar
import tqdm

for i in tqdm.tqdm(range(100)):
    #do something slow
    _ = [random.random() for _ in range(100000)]
    
from typing import List

def primes_up_to (n: int) -> List[int]:
    primes = [2]
    
    with tqdm.trange(3,n) as t:
        for i in t:
            # i is prime if no smaller prime divides it
            i_is_prime = not any(i % p == 0 for p in primes)
            if i_is_prime:
                primes.append(i)
                
            t.set_description(f"{len(primes)} primes")
    return primes

my_primes = primes_up_to(1000)

import numpy as np
np.transpose(my_primes)

