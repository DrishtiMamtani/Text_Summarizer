#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[15]:


from attention import AttentionLayer


# In[16]:


import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


# # Data Read

# In[17]:


data=pd.read_csv("Dataset.csv")


# In[18]:


data.head(1)


# In[19]:


data.drop_duplicates(subset=['Text'],inplace=True)
#subset=['Text'] searches for duplicates only in the column with name Text(Last column)
#inplace=true will cause all the rows which have same text value to be dropped. 


# In[20]:


data.dropna(axis=0,inplace=True)
#this is the instruction to delete all rows with atleast one NaN values


# # Data Info

# In[21]:


data.info()


# # Preprocessing

# In[22]:


# To remove unnecessary symbols we will define a dictionary for expanding the contractions
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}


# In[23]:


#Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that 
#a search engine has been programmed to ignore, both when indexing entries for searching and 
#when retrieving them as the result of a search query.
#To check the list of stopwords we use the following instruction
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 


# In[24]:


stop_words


# Defining function for test cleaning

# In[135]:


def text_cleaner(text,num):
    String1 = text.lower() #converting to lower case. After this the complete review will be in lower case
    String1 = BeautifulSoup(String1, "lxml").text 
    #Beautiful Soup is a Python library for pulling data out of HTML and XML files. It is used for tasks like extracting the 
    #entire text from a page, extracting all URLs found in a page
    #We create a BeautifulSoup object by passing two arguments:newString(raw HTML content) and lxml(HTML parser we want to use)
    String1 = re.sub(r'\([^)]*\)', '', String1)
    #The re.sub() function in the re module can be used to replace substrings. 
    #The syntax for re.sub() is re.sub(pattern,repl,string). 
    #That will replace the matches in string with repl. 
    String1 = re.sub('"','', String1)
    String1 = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in String1.split(" ")]) 
    #The join() method is a string method and returns a string in which the elements of sequence have been joined by str separator.
    #Here we join with an empty string.
    #The above instruction removes contraction from the string.
    String1 = re.sub(r"'s\b","",String1)
    String1 = re.sub("[^a-zA-Z]", " ", String1) 
    String1 = re.sub('[m]{2,}', 'mm', String1)
    #removes the stopwords
    #tokens will be a list
    if(num==0):
        #for text remove the stop_words
        tokens = [w for w in String1.split() if not w in stop_words]
    else:
        #for summary stop words cannot be removed because the summary is already small. So just take all words in summary as tokens
        tokens=String1.split()
    long_words=[]
    for i in tokens:
        #for each token if length of the token is less than one then eliminate the token/word 
        if len(i)>1:                                                
            long_words.append(i)  
    #join will convert the list back to string and strip() will remove leading spaces if any.
    return (" ".join(long_words)).strip()


# Understanding the function text_cleaner

# In[76]:


#Sample string to understand the use of contraction mapping and join function
string="ABC ain't def ain't"


# In[77]:


#split the words of a sentence at the " " (string.split(" "))
#check each word if it is the key of the dictionary contraction_mapping then replace the key by the value
#if not then keep the word as it is
#string will be list of resultant words
string =[contraction_mapping[t] if t in contraction_mapping else t for t in string.split(" ")]
string


# In[78]:


#to get the string back from list of words
string=' '.join(string) 
string


# In[79]:


#Join the list of words so obtanined with an empty string to convert back to string
string = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in string.split(" ")]) 
string


# In[87]:


string


# In[88]:


string.split() #string bydefault splits at spaces


# In[91]:


#This gives words which are not stopwords
tokens = [w for w in string.split() if not w in stop_words]
tokens


# In[93]:


#redefining tokens to understand elimination of short words
tokens=['I','am','a','girl']


# In[100]:


long_words=[]
for i in tokens :
    #Initially long words is a empty list
    #Examine each token and if its length is greater than 1 then include it in the long_word list. 
    #In this way all the short words with length 0 or 1 are removed
    if len(i)>1:                                                
        long_words.append(i)  
long_words


# In[101]:


" ".join(long_words)


# In[102]:


" ".join(long_words).strip()


# Calling the function

# In[133]:


#call the function
#cleaned_text is an empty string intially. For each entry i.e. row the text column value is 
#taken and cleaned by the function text_cleaner defined above. The cleaned text is added in the cleaned_text list.
cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t,0))


# In[136]:


#The same function is called for the Summary column as well and in similar manner cleanned_summary list is generated.
cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(text_cleaner(t,1))


# In[137]:


cleaned_text[0:4]


# In[138]:


cleaned_summary[0:4]


# In[140]:


#Adding two new columns namely cleaned_text and cleaned_summary in the data
data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary


# Drop ' '(Empty) rows

# In[141]:


#first replace blank spaces with NaN and then drop rows with NaN.
#This can be called a trick to drop ' ' by using dropna
data.replace('', np.nan, inplace=True)
data.dropna(axis=0,inplace=True)


# In[143]:


data.head(1)


# In[ ]:




