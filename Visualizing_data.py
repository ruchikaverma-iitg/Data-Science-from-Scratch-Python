# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:47:10 2019

@author: Ruchika
"""

#Simple plots using matplotlib
"""
####################################################################
####################   A simple line chart  ########################
####################################################################
"""

from matplotlib import pyplot as plt

#Data to plot
years = [i for i in range(1950,2011,10)]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

#Plotting a line chart, years on x-axis and gdp on y-axis
plt.figure()
plt.plot(years, gdp, color='red', marker = 'o', linestyle = 'solid')
plt.title("Nominal GDP")#Title
plt.ylabel("Billions of $")#Adding label to the y-axis
plt.xlabel("Years")#Adding label to the x-axis
plt.show()

"""
####################################################################
############################   Bar chart  ##########################
####################################################################
"""
#Data to plot
movies = ["Annie Hall", "Ben-Hur", "Casablanca","Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

#Plotting a bar chart with x-coordinates [0, 1, 2, 3, 4, 5] and length of y-coordinates represent #oscars
plt.figure()
plt.bar(range(len(movies)), num_oscars, color='red')
plt.title("My favorite Movies")#Title
plt.ylabel("Number of Academy Awards")#Adding label to the y-axis
plt.xticks(range(len(movies)), movies)#Adding movie names as label to the x-axis at bar centers
plt.show()

"""
####################################################################
############ Plotting histogram of grades using bar charts #########
####################################################################
"""
from collections import Counter
grades = [83, 95, 91,87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

#Bucket grades by decile, but put 100 in with 90s
histogram = Counter(min(grade//10*10, 90) for grade in grades) #Counter counts the number of times grade counted in the same bucket

#Plotting
plt.figure()
plt.bar([x+5 for x in histogram.keys()],#Shift bars right by 5
       histogram.values(),              #Give each bar its correct height computed by Counter
       10,                              #Give each bar a width of 10
       edgecolor=(0,0,0))               #Give black color on bar edges to make all bars distinct

plt.axis([-5, 105, 0, 5]) # x axis range from -5 to 105 and y-axis range from 0 to 5

plt.xticks([10*i for i in range(11)]) #xaxis label from 0, 10, 20, ......100
plt.xlabel("Decile")#Adding label to the x-axis
plt.ylabel("# of Students")#Adding label to the y-axis
plt.title("Distribution of exam 1 Grades")#Title
plt.show()

"""
####################################################################
################# Multiple plots on the same chart #################
####################################################################
"""
variance = [2**i for i in range(9)]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i,_ in enumerate(variance)]

# Multiple line charts on the same plot
plt.figure()
plt.plot(xs, variance,     'g-', label = 'variance') #Green solid line
plt.plot(xs, bias_squared, 'r-.', label = 'bias^2')  #Red dot-dashed line
plt.plot(xs, total_error,  'b:', label = 'total error')  #Blue dotted line

plt.legend(loc = 9)
plt.xlabel("Model complexity")#Adding label to the x-axis
plt.xticks([])
plt.title("The Bias-Variance Tradeoff")#Title
plt.show()

"""
####################################################################
##########################  Scatter plots ##########################
####################################################################
"""

#Data to plot
friends = [70,65,72,63,71,64,60,64,67]
minutes=[175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = [i for i in 'abcdefghi']

plt.figure()
plt.scatter(friends,minutes,color='red')

#Label each point 
for label, friend_count, minute_count in zip(labels,friends,minutes):
    plt.annotate(label,
                xy=(friend_count, minute_count), #Put the label with its point
                xytext=(5,-5),
                textcoords = 'offset points')
    
plt.xlabel("# of friends")#Adding label to the x-axis
plt.ylabel("daily minute spent on the site")#Adding label to the y-axis
plt.title("Daily minutes vs. Number of friends")#Title
plt.show()