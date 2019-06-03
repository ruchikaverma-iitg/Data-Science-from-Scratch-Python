# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:30:00 2019

@author: Ruchika
"""

"""
######################################################################################################################
CountingClicker can be used to count or track how many people have shown up for a class. Everybody will press a 
button (click function) and at the end, read function can be used to see the total number of students/people attended
the class. 
There is a subclass NoResetClicker and it will be used by students who entered the class and reset button/function
doesn't work for it so that students can't reset the counter. However, Professor can reset the counter using the reset
function of the class CountingClicker.
######################################################################################################################
"""

# Count how many times a button was pressed
class CountingClicker:    
    
    def __init__ (self,count = 0):
        self.count = count
        
    def __repr__(self):
        return f"CountingClicker (count = {self.count})" 
    
    def click(self, num_times = 1):
        #Click the clicker for a few times
        self.count += num_times
        return self.count
        
    def read(self):
        return self.count
    
    def reset(self):
        self.count = 0
        return self.count
        
        
# Create an instance of the class CountingClicker
clicker1 = CountingClicker()
print(clicker1)

# Create another instance of the class CountingClicker starting with number of counts = 100
clicker2 = CountingClicker(100)
print(clicker2)

#Other way to assign value for clicker2
clicker2 = CountingClicker(count = 100)

print(clicker2)
print(clicker2.click()) #Click one more time and print number of counts
print(clicker2.read()) #Read current number of counts
print(clicker2.reset()) #Print number of counts after resetting the CountingClicker 
print(clicker2.click()) #Click and print number of counts


"""
##############################################################################################################
Subclass can be used for the students in the attendance system so that they can't reset the CountingClicker
##############################################################################################################
"""

# A subclass inherits the behavior of its parent class
class NoResetClicker(CountingClicker):
    #This class works same as CountingClicker except the reset function
    def reset(self):
        pass
    
    
# Create an instance of the class CountingClicker
clicker1 = NoResetClicker()
print(clicker1)

# Create another instance of the class CountingClicker starting with number of counts = 100
clicker2 = NoResetClicker(100)
print(clicker2)

#Other way to assign value for clicker2
clicker2 = NoResetClicker(count = 100)

print(clicker2)
print(clicker2.click()) #Click one more time and print number of counts
print(clicker2.read()) #Read current number of counts
print(clicker2.reset()) #Print number of counts after resetting the CountingClicker 
print(clicker2.click()) #Click and print number of counts    

