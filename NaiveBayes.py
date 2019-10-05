# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 09:17:52 2019

@author: Ruchika
"""


"""
####################################################################################################
##########################         Split text into distinct words         ##########################
####################################################################################################
"""
from typing import Set
import re

def tokenize(text: str) -> Set[str]:
    text = text.lower()  # Convert to lowercase
    all_words = re.findall("[a-z0-9']+",text) # re.findall is used to extract words consisting of letters, numbers and apostrophes
    return set(all_words) #Remove duplicates

tokenize("The science of today is the technology of tomorrow")
    

"""
####################################################################################################
##########################       Define type of our training data         ##########################
####################################################################################################
"""

from typing import NamedTuple
    
class Message(NamedTuple):
    text: str
    is_spam : bool
    

"""
####################################################################################################
################### Define all functions in a class named as NaiveBayesClassifier###################
####################################################################################################
"""
# Refer nonspam emails as ham emails

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:

    def __init__(self, k: float = 0.5) -> None:
        self.k = k # smoothing factor
        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0
 
      
    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            #Increment message counts
            if message.is_spam:
                 self.spam_messages += 1
            else:
                 self.ham_messages += 1
                    
            #Increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1  
    
                    
    def _probabilties (self, token: str) -> Tuple[float, float]:
        """ returns P(token/spam) and P(token/ham) """
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]
        
        p_token_spam = (spam + self.k)/ (self.spam_messages + 2*self.k)
        p_token_ham = (ham + self.k)/ (self.ham_messages + 2*self.k)
        
        return p_token_spam, p_token_ham    


    
    """ Find prob (spam/token)"""
    # Use exp(log(pi*p2*p3)) instead of using p1*p2*p3
    def predict(self, text:str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0
        
        #Iterate through each word of our vocabulary
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilties(token)
            
            # If *token* appears in the message,
            # add the log probability of seeing it            
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
                
            # Otherwise add the log probability of not seeing it
            # which is log(1-probability of seeing it)            
            else:
                log_prob_if_spam += math.log(1-prob_if_spam)
                log_prob_if_ham += math.log(1-prob_if_ham)
                
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam/(prob_if_spam+prob_if_ham) #prob (spam/token)  


"""
####################################################################################################
##########################                    Test the model              ##########################
####################################################################################################
"""

Messages = [Message("spam rules", is_spam = True),
            Message("ham rules", is_spam = False),
            Message("hello ham", is_spam = False)]

model = NaiveBayesClassifier(k=0.5)
model.train(Messages)

assert model.tokens  == {"spam","ham","rules","hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

print(model.tokens)
print(model.spam_messages)
print(model.ham_messages)
print(model.token_spam_counts)
print(model.token_ham_counts)

"""
####################################################################################################
##########################                 Download real data             ##########################
####################################################################################################
"""

from io import BytesIO #To treat bytes as a file
import requests #To download the files
import tarfile #For tar.bz files

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"

FILES = ["20021010_easy_ham.tar.bz2",
        "20021010_hard_ham.tar.bz2",
        "20021010_spam.tar.bz2"]

OUTPUT_DIR = "spam_data"

for filename in FILES:
    # Use requests to get the file contents at each URL
    content = requests.get(f"{BASE_URL}/{filename}").content
    
    # Wrap the in-memory bytes so we can use them as a file
    fin = BytesIO(content)
    
    # And extract all the files to the output directory
    with tarfile.open(fileobj= fin, mode = 'r:bz2') as tf:
        tf.extractall(OUTPUT_DIR)

"""
####################################################################################################
##########################          Arrange data in NamedTuple            ##########################
####################################################################################################
"""        

import glob, re

path = "spam_data/*/*"
data: List[Message] = []
    
# glob.glob returns every filename that matches the wildcarded path
for filename in glob.glob(path):
    is_spam = "ham" not in filename
    
    # There are some garbage characters in the emails; the errors = 'ignore'
    # skips them instead of raising an exception
    with open(filename, errors = 'ignore') as email_file:
        for line in email_file:
            if line.startswith("Subject:"):
                subject = line.lstrip("Subject: ")
                data.append(Message(subject, is_spam))
                break # done with this file
  
"""
####################################################################################################
##########################     Split data into training and testing set   ##########################
####################################################################################################
"""                 

import random
from machine_learning import split_data

random.seed(0)
train_messages, test_messages = split_data(data, 0.75)

  
"""
####################################################################################################
##########################          Train the model with training set     ##########################
####################################################################################################
"""      
model = NaiveBayesClassifier()
model.train(train_messages)

from collections import Counter

predictions = [(message, model.predict(message.text))
              for message in test_messages]

confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                          for message, spam_probability in predictions)
print(confusion_matrix)

"""
####################################################################################################
######## Inspect model to find words which are most and least indicative of spam ###################
####################################################################################################
"""
def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    prob_if_spam, prob_if_ham = model._probabilties(token)
    return prob_if_spam/ (prob_if_spam + prob_if_ham)

words = sorted(model.tokens, key = lambda t: p_spam_given_token(t,model))
print("spammiest_words", words[-10:])
print("hammiest_words", words[:10])                