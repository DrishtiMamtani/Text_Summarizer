#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[2]:


from attention import AttentionLayer


# In[3]:


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

# In[4]:


data=pd.read_csv("Dataset.csv")


# In[8]:


data.head(1)


# In[9]:


data.drop_duplicates(subset=['Text'],inplace=True)
#subset=['Text'] searches for duplicates only in the column with name Text(Last column)
#inplace=true will cause all the rows which have same text value to be dropped. 


# In[10]:


data.dropna(axis=0,inplace=True)
#this is the instruction to delete all rows with atleast one NaN values


# # Data Info

# In[11]:


data.info()


# # Preprocessing

# In[12]:


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


# In[13]:


#Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that 
#a search engine has been programmed to ignore, both when indexing entries for searching and 
#when retrieving them as the result of a search query.
#To check the list of stopwords we use the following instruction
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 


# In[14]:


stop_words


# Defining function for test cleaning

# In[15]:


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

# In[16]:


#Sample string to understand the use of contraction mapping and join function
string="ABC ain't def ain't"


# In[17]:


#split the words of a sentence at the " " (string.split(" "))
#check each word if it is the key of the dictionary contraction_mapping then replace the key by the value
#if not then keep the word as it is
#string will be list of resultant words
string =[contraction_mapping[t] if t in contraction_mapping else t for t in string.split(" ")]
string


# In[18]:


#to get string back from list of words
string=' '.join(string) 
string


# In[19]:


#Join the list of words so obtanined with an empty string to convert back to string
string = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in string.split(" ")]) 
string


# In[20]:


string


# In[21]:


string.split() #string bydefault splits at spaces


# In[22]:


#This gives words which are not stopwords
tokens = [w for w in string.split() if not w in stop_words]
tokens


# In[23]:


#redefining tokens to understand elimination of short words
tokens=['I','am','a','girl']


# In[24]:


long_words=[]
for i in tokens :
    #Initially long words is a empty list
    #Examine each token and if its length is greater than 1 then include it in the long_word list. 
    #In this way all the short words with length 0 or 1 are removed
    if len(i)>1:                                                
        long_words.append(i)  
long_words


# In[25]:


" ".join(long_words)


# In[26]:


" ".join(long_words).strip()


# Calling the function

# In[27]:


#call the function
#cleaned_text is an empty string intially. For each entry i.e. row the text column value is 
#taken and cleaned by the function text_cleaner defined above. The cleaned text is added in the cleaned_text list.
cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t,0))


# In[28]:


#The same function is called for the Summary column as well and in similar manner cleanned_summary list is generated.
cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(text_cleaner(t,1))


# In[29]:


cleaned_text[0:4]


# In[30]:


cleaned_summary[0:4]


# In[31]:


#Adding two new columns namely cleaned_text and cleaned_summary in the data
data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary


# Drop ' '(Empty) rows

# In[32]:


#first replace blank spaces with NaN and then drop rows with NaN.
#This can be called a trick to drop ' ' by using dropna
data.replace('', np.nan, inplace=True)
data.dropna(axis=0,inplace=True)


# 
# # Understanding the distribution

# In[51]:


#For plotting graphs
import matplotlib.pyplot as plt
text_word_count = []
summary_word_count = []
for i in data['cleaned_text']:
    #for each entry in cleaned_text the number of words are counted in the entry and the count is appended to text_word_count list 
    text_word_count.append(len(i.split()))
for i in data['cleaned_summary']:
    #for each entry in cleaned_summary the number of words are counted in the entry and the count is appended to summary_word_count list  
    summary_word_count.append(len(i.split()))
#A dataframe with two columns is made. 1st column has entries of text_word_count and 2nd has entries of summart_word_count 
df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})

        


# In[52]:


df.head() #by default 5 entries are displayed


# In[53]:


#Plotting the histogram for the dataframe
#Histogram plots the graph between values and frequencies
df.hist(bins = 30)
plt.show()


# Lets find out the percentage of summaries below length=8,length=9 and length=10

# Find the appropriate max summary length

# In[79]:


#initialize count with value 0
#for each entry in cleaned summary, split the cleaned summary at spaces to find number of words and if the no. of word
#are less than or equal to 8 then increement count
#In this way count will have the count of cleaned summaries with number of words less than or equal to 8.
count=0
for i in data['cleaned_summary']:
    if(len(i.split())<=8):
        count=count+1
print(count/len(data['cleaned_summary']))


# In[80]:


#initialize count with value 0
#for each entry in cleaned summary, split the cleaned summary at spaces to find number of words and if the no. of word
#are less than or equal to 9 then increement count
#In this way count will have the count of cleaned summaries with number of words less than or equal to 9.
count=0
for i in data['cleaned_summary']:
    if(len(i.split())<=9):
        count=count+1
print(count/len(data['cleaned_summary']))


# In[81]:


#initialize count with value 0
#for each entry in cleaned summary, split the cleaned summary at spaces to find number of words and if the no. of word
#are less than or equal to 10 then increement count
#In this way count will have the count of cleaned summaries with number of words less than or equal to 10.
count=0
for i in data['cleaned_summary']:
    if(len(i.split())<=10):
        count=count+1
print(count/len(data['cleaned_summary']))


# Lets fix the max cleaned summary length to 10

# Find the appropriate max text length

# In[83]:


#initialize count with value 0
#foe each entry in cleaned text, split the cleaned text at spaces to find number of words and if the no. of word
#are less than or equal to 20 then increement count
#In this way count will have the count of cleaned texts with number of words less than or equal to 20.
count=0
for i in data['cleaned_text']:
    if(len(i.split())<=20):
        count=count+1
print(count/len(data['cleaned_text']))


# In[85]:


#initialize count with value 0
#foe each entry in cleaned text, split the cleaned text at spaces to find number of words and if the no. of word
#are less than or equal to 25 then increement count
#In this way count will have the count of cleaned texts with number of words less than or equal to 25.
count=0
for i in data['cleaned_text']:
    if(len(i.split())<=25):
        count=count+1
print(count/len(data['cleaned_text']))


# In[86]:


#initialize count with value 0
#foe each entry in cleaned text, split the cleaned text at spaces to find number of words and if the no. of word
#are less than or equal to 35 then increement count
#In this way count will have the count of cleaned texts with number of words less than or equal to 35.
count=0
for i in data['cleaned_text']:
    if(len(i.split())<=35):
        count=count+1
print(count/len(data['cleaned_text']))


# In[87]:


#initialize count with value 0
#foe each entry in cleaned text, split the cleaned text at spaces to find number of words and if the no. of word
#are less than or equal to 45 then increement count
#In this way count will have the count of cleaned texts with number of words less than or equal to 45.
count=0
for i in data['cleaned_text']:
    if(len(i.split())<=45):
        count=count+1
print(count/len(data['cleaned_text']))


# Lets fix the max text length as 45

# Now selecting those entries in which cleaned text length is less than equal to 45 and cleaned summary length is less than equal to 10

# In[91]:


max_summary_len=10
max_text_len=45


# In[92]:


#making array of cleaned text entries and cleaned summary entries
cleaned_text =np.array(data['cleaned_text'])
cleaned_summary=np.array(data['cleaned_summary'])

#short_text and short_summary are initially empty but will contain all the text and summary entries which fall in the desired range
short_text=[]
short_summary=[]

for i in range(len(cleaned_text)):
    #For all entries if the cleaned_summary has no. of words <=max summary length which is equal to 10 
    #and cleaned_text has no. of words <=max text length which is equal to 45 add such entries to the lists short_text and short_summary
    if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])
#create a dataframe to store the results of short_text and short_summary    
df1=pd.DataFrame({'text':short_text,'summary':short_summary})


# In[93]:


df1.head()


# Now add the start and end token to each summary. This can be done using lambda function

# In[94]:


#This will replace each summary with 'starttoken' as start token concatenated with summary concatenated with 'endtoken' as end token 
#Be sure that the chosen special tokens never appear in the summary
df1['summary'] = df1['summary'].apply(lambda x : 'starttoken '+ x + ' endtoken')


# Now splitting data into training and testing sets. Take 90% of the dataset as the training data and evaluate the performance on the remaining 10%

# In[96]:


#Sklearn is used to perform the split. This is standard technique to split the dataset.Test size is set to 0.1 i.e. 10%.
#x variable is text
#y variable is summary
from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['text']),np.array(df['summary']),test_size=0.1,random_state=0,shuffle=True)


# In[ ]:




