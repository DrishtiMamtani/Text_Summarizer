#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


from attention import AttentionLayer


# In[2]:


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

# In[3]:


data=pd.read_csv("Dataset.csv")


# In[4]:


data.head(1)


# In[5]:


data.drop_duplicates(subset=['Text'],inplace=True)
#subset=['Text'] searches for duplicates only in the column with name Text(Last column)
#inplace=true will cause all the rows which have same text value to be dropped. 


# In[6]:


data.dropna(axis=0,inplace=True)
#this is the instruction to delete all rows with atleast one NaN values


# # Data Info

# In[7]:


data.info()


# # Preprocessing

# In[8]:


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


# In[9]:


#Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that 
#a search engine has been programmed to ignore, both when indexing entries for searching and 
#when retrieving them as the result of a search query.
#To check the list of stopwords we use the following instruction
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 


# In[10]:


stop_words


# Defining function for text cleaning

# In[11]:


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
    #removes all the stopwords
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

# In[12]:


#Sample string to understand the use of contraction mapping and join function
string="ABC ain't def ain't"


# In[13]:


#split the words of a sentence at the " " (string.split(" "))
#check each word if it is the key of the dictionary contraction_mapping then replace the key by the value
#if not then keep the word as it is
#string will be list of resultant words
string =[contraction_mapping[t] if t in contraction_mapping else t for t in string.split(" ")]
string


# In[14]:


#to get the string back from list of words
string=' '.join(string) 
string


# In[15]:


#Join the list of words so obtanined with an empty string to convert back to string
string = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in string.split(" ")]) 
string


# In[16]:


string


# In[17]:


string.split() #string bydefault splits at spaces


# In[18]:


#This gives words which are not stopwords
tokens = [w for w in string.split() if not w in stop_words]
tokens


# In[19]:


#redefining tokens to understand elimination of short words
tokens=['I','am','a','girl']


# In[20]:


long_words=[]
for i in tokens :
    #Initially long words is a empty list
    #Examine each token and if its length is greater than 1 then include it in the long_word list. 
    #In this way all the short words with length 0 or 1 are removed
    if len(i)>1:                                                
        long_words.append(i)  
long_words


# In[21]:


" ".join(long_words)


# In[22]:


" ".join(long_words).strip()


# Calling the function

# In[23]:


#call the function
#cleaned_text is an empty string intially. For each entry i.e. row the text column value is 
#taken and cleaned by the function text_cleaner defined above. The cleaned text is added in the cleaned_text list.
cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t,0))


# In[24]:


#The same function is called for the Summary column as well and in similar manner cleanned_summary list is generated.
cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(text_cleaner(t,1))


# In[25]:


cleaned_text[0:4]


# In[26]:


cleaned_summary[0:4]


# In[27]:


#Adding two new columns namely cleaned_text and cleaned_summary in the data
data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary


# Drop ' '(Empty) rows

# In[28]:


#first replace blank spaces with NaN and then drop rows with NaN.
#This can be called a trick to drop ' ' by using dropna
data.replace('', np.nan, inplace=True)
data.dropna(axis=0,inplace=True)


# 
# # Understanding the distribution

# In[29]:


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

        


# In[30]:


df.head() #by default 5 entries are displayed


# In[31]:


#Plotting the histogram for the dataframe
#Histogram plots the graph between values and frequencies
df.hist(bins = 30)
plt.show()


# Lets find out the percentage of summaries below length=8,length=9 and length=10

# Find the appropriate max summary length

# In[32]:


#initialize count with value 0
#for each entry in cleaned summary, split the cleaned summary at spaces to find number of words and if the no. of word
#are less than or equal to 8 then increement count
#In this way count will have the count of cleaned summaries with number of words less than or equal to 8.
count=0
for i in data['cleaned_summary']:
    if(len(i.split())<=8):
        count=count+1
print(count/len(data['cleaned_summary']))


# In[33]:


#initialize count with value 0
#for each entry in cleaned summary, split the cleaned summary at spaces to find number of words and if the no. of word
#are less than or equal to 9 then increement count
#In this way count will have the count of cleaned summaries with number of words less than or equal to 9.
count=0
for i in data['cleaned_summary']:
    if(len(i.split())<=9):
        count=count+1
print(count/len(data['cleaned_summary']))


# In[34]:


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

# In[35]:


#initialize count with value 0
#foe each entry in cleaned text, split the cleaned text at spaces to find number of words and if the no. of word
#are less than or equal to 20 then increement count
#In this way count will have the count of cleaned texts with number of words less than or equal to 20.
count=0
for i in data['cleaned_text']:
    if(len(i.split())<=20):
        count=count+1
print(count/len(data['cleaned_text']))


# In[36]:


#initialize count with value 0
#foe each entry in cleaned text, split the cleaned text at spaces to find number of words and if the no. of word
#are less than or equal to 25 then increement count
#In this way count will have the count of cleaned texts with number of words less than or equal to 25.
count=0
for i in data['cleaned_text']:
    if(len(i.split())<=25):
        count=count+1
print(count/len(data['cleaned_text']))


# In[37]:


#initialize count with value 0
#foe each entry in cleaned text, split the cleaned text at spaces to find number of words and if the no. of word
#are less than or equal to 35 then increement count
#In this way count will have the count of cleaned texts with number of words less than or equal to 35.
count=0
for i in data['cleaned_text']:
    if(len(i.split())<=35):
        count=count+1
print(count/len(data['cleaned_text']))


# In[38]:


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

# In[39]:


max_summary_len=10
max_text_len=45


# In[40]:


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


# In[41]:


df1.head()


# Now add the start and end token to each summary. This can be done using lambda function

# In[42]:


#This will replace each summary with 'starttoken' as start token concatenated with summary concatenated with 'endtoken' as end token 
#Be sure that the chosen special tokens never appear in the summary
df1['summary'] = df1['summary'].apply(lambda x : 'starttoken '+ x + ' endtoken')


# Now splitting data into training and testing sets. Take 90% of the dataset as the training data and evaluate the performance on the remaining 10%

# In[355]:


#Sklearn is used to perform the split. This is standard technique to split the dataset.Test size is set to 0.1 i.e. 10%.
#x variable is text
#y variable is summary
#df['text'] and df['summary'] contain respective reviews and summaries in form of array
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(np.array(df1['text']),np.array(df1['summary']),test_size=0.1,random_state=0,shuffle=True)
#xtrain,x_test,y_train,y_test all are numpy arrays containing reviews and summaries


# # Preparing the Tokenizer
# 

# # Text Tokenizer

# In[356]:


from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences


# In[357]:


#prepare a tokenizer for reviews on training data
t = Tokenizer() 


# In[358]:


t
#Output would be  <keras_preprocessing.text.Tokenizer at 0x217e980deb8>


# In[359]:


#tokenizing x_train. First x_train which is a numpy array is converted to list
t.fit_on_texts(list(x_train))
#fit_on_texts Updates internal vocabulary based on a list of texts. 
#This method creates the vocabulary index based on word frequency. 
#So if you give it something like, "The cat sat on the mat." 
#It will create a dictionary s.t. word_index["the"] = 1; 
#word_index["cat"] = 2 it is word -> index dictionary so every word gets a unique integer value. 
#0 is reserved for padding. 
#So lower integer means more frequent word (often the first few are stop words because they appear a lot).


# Rarewords

# In[360]:


#Rare words are those words which do not appear too often
#Defining the threshold as 3. If the words apear less than thrice then the word are rare words.
threshold=3
#count has count of rare words
count=0
#totalcount has the count of total number of words i.e. size of vocabulary
totalcount=0
#frequency has total frequency of all the rare words
frequency=0
#totalfrequency has the sum of all frequencies of all words
totalfrequency=0
#t.word_counts.items() will give items of ordered dictionary i.e.  'key' and 'value' pair. key being the word and value being the number of times it ocuured.
#odict_items([('love', 45148), ('raspberry', 1096), ('shortbread', 276), ('cookies', 6596), ('easy', 10588), ('find', 21556), ....
for key,value in t.word_counts.items():
    #accessing each key value pair
    #totalcount is increemented by 1 as the word encountered is a new word add totalcount by 1
    totalcount=totalcount+1
    #totalfrequency is increemneted by value 
    totalfrequency=totalfrequency+value
    #if value is less than threshold than it is rare word and count is incremented by 1 and frequency by value. 
    if(value<threshold):
        count=count+1
        frequency=frequency+value
print(count)
print(totalcount)
# %of rare words is (number of rare words divided by total number of words) multiplied by 100 i.e. (count divided by totalcount) multiplied by 100
print("% of rare words in vocabulary:",(count/totalcount)*100)
# coverage of rare words is (frequency divided by total frequency) multiplied by 100
print("Total Coverage of rare words:",(frequency/totalfrequency)*100)


# In[361]:


#prepare a tokenizer for reviews on training data
#totalcount-count is number of common words
#Only common words will be remembered
x_tokenizer = Tokenizer(num_words=totalcount-count)
x_tokenizer.fit_on_texts(list(x_train))
#x_tokenizer.word_index will give
#{'like': 1,'good': 2, 'great': 3,'taste': 4, 'product': 5,'love': 6,'one': 7,....
#This means like appers is the most common word followed by good then great and so on

#convert text sequences into integer sequences
x_train_sequence    =   x_tokenizer.texts_to_sequences(x_train)
#only common words will be remembered
x_test_sequence   =   x_tokenizer.texts_to_sequences(x_test)

#pad_sequences is used to ensure that all sequences in a list have the same length.
#By default this is done by padding 0 in the beginning of each sequence until each sequence has the same length as the longest sequence.
#Here zeros are padded at the end
x_train   =   pad_sequences(x_train_sequence,  maxlen=max_text_len,padding='post')
x_test   =   pad_sequences(x_test_sequence, maxlen=max_text_len, padding='post')

#size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1


# In[362]:


x_train


# In[363]:


x_voc


# For understanding Tokenizer

# In[364]:


from keras.preprocessing.text import Tokenizer
texts = ['a a a', 'b b b b b', 'c c c c c c c','ddd','aa a','aa aa']
#'a a a' is a string with 3 words


# In[365]:


tokenizer = Tokenizer(num_words=4)
#num_words: the maximum number of words to keep, based on word frequency. 
#Only the most common num_words-1 words will be kept.
#Tokenizer will use only three most common words and at the same time, it will keep the counter of all words - even when it's obvious that it will not use it later.


# In[366]:


tokenizer.fit_on_texts(texts)


# In[367]:


tokenizer.word_index
#c is the most common word so will get the value 1.
#b will get value 2
#a will get 3
#aa will get 4
#ddd will get 5
#More times a number appears lesser will be its key


# In[368]:


tokenizer.texts_to_sequences(texts)
#only c,b,and a will be remembered
#See "aa a" i.e. 4th index only a is remembered and aa is not so only 3 is the answer


# # Summary Tokenizer

# In[369]:


#prepare a tokenizer for reviews on training data
t1= Tokenizer()   
t1.fit_on_texts(list(y_train))


# In[370]:


#Doing the similar thing with summary
#Threshold is set to 5
threshold=5
count=0
totalcount=0
frequency=0
totalfrequency=0

for key,value in t.word_counts.items():
    totalcount=totalcount+1
    totalfrequency=totalfrequency+value
    if(value<threshold):
        count=count+1
        frequency=frequency+value
print(count)
print(totalcount)    
print("% of rare words in vocabulary:",(count/totalcount)*100)
print("Total Coverage of rare words:",(frequency/totalfrequency)*100)


# In[371]:


#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words=totalcount-count) 
y_tokenizer.fit_on_texts(list(y_train))

#convert text sequences into integer sequences
y_train_sequence=y_tokenizer.texts_to_sequences(y_train) 
y_test_sequence=y_tokenizer.texts_to_sequences(y_test) 

#padding zero upto maximum length
y_train=pad_sequences(y_train_sequence, maxlen=max_summary_len, padding='post')
y_test=pad_sequences(y_test_sequence, maxlen=max_summary_len, padding='post')

#size of vocabulary
y_voc=y_tokenizer.num_words +1


# In[372]:


y_voc


# In[374]:


#The number of times startoken appears should be equal to length of training data 
y_tokenizer.word_counts['starttoken'],len(y_train)


# In[377]:


#Deleting those rows which only contain start and end token
empty=[]
#Checking each element of y train. Each element of y train is a list in itself.
for i in range(len(y_train)):
    count=0
    for j in y_train[i]:
        #checking each element in one element of y_train
            count=count+1
    if(count==2):
        #if there are only 2 non zero elements that is start and end token then the list is actualy empty and append that index 
        #in empty list so that we can delete those rows
        empty.append(i)

#Deleting x and y  for those indices present in empty list that is those rows which only have start and end token
y_train=np.delete(y_train,empty, axis=0)
x_train=np.delete(x_train,empty, axis=0)
#Axis is 0 because rows have to be deleted


# In[378]:


#Deleting those rows which only contain start and end token
empty=[]
for i in range(len(y_test)):
    count=0
    for j in y_test[i]:
        if j!=0:
            count=count+1
    if(count==2):
        empty.append(i)

y_test=np.delete(y_test,empty, axis=0)
x_test=np.delete(x_test,empty, axis=0)

