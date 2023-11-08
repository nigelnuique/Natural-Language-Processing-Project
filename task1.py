#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Nigel Nuique
# #### Student ID: s3985410
# 
# Date: September 22, 2023
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * pandas
# * re
# * numpy
# * sklearn
# * nltk
# * itertools
# * collections
# 
# ## Introduction
# In this part of the assessment, the data is first loaded and examined. It is then pre-processed according to the assignment specification. the result is the vocabulary of the dataset, the tokenized job advertisements and other information that will be used in task 2 and 3.
# 

# ## Importing libraries 

# In[1]:


import pandas as pd
import re
import numpy as np
from sklearn.datasets import load_files 
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from collections import Counter


# ### 1.1 Examining and loading data
# The data is loaded using the load_files function from sklearn and then the data is passed on to the proper data structures.

# In[2]:


job_data = load_files(r"data") 


# The result is a data bunch:

# In[3]:


type(job_data)


# The data is organized according to the categories of jobs. The folder names are based on the categories. Inside the categories are the individual job advertisements in the txt file format.

# In[4]:


job_data['filenames']


# The target is written as a list of numbers from 0 to 3 indicating that there are 4 job categories.

# In[5]:


job_data['target'][0:10]


# There are 4 folders corresponding to the following job categories:

# In[6]:


job_data['target_names']


# In[7]:


# test whether the job category name and the target value matches:
example = 10 
job_data['filenames'][example], job_data['target'][example] 


# The job advertisements are loaded into a list and the categories are loaded into a numpy array.

# In[8]:


job_ads, categories = job_data.data, job_data.target  
print(type(categories))
print(type(job_ads))


# Loading an example job advertisement

# In[9]:


job_ads[example][0:1000]


# In[10]:


job_data["target_names"][categories[example]]


# ### 1.2 Pre-processing data
# In this step, the job descriptions are pre-processed according to the assignment specification.

# This function extracts the description from the job advertisement, converts all the words to lowercases, segments the raw job advertisement into sentences, tokenizes each sentence and converts them to a list of tokens.

# In[11]:


def tokenize_job_ad(raw_job_ad):
    """
        
    """        
    ad = raw_job_ad.decode('utf-8') # convert the bytes-like object to python string, need this before we apply any pattern search on it
    ad = ad.lower() # cover all words to lowercase
    
    # Find the start of the description part using the "Description:" keyword
    description_start = ad.find("description:")

    if description_start == -1:
        # Handle the case where "Description:" is not found
        return []

    # Extract the description part from "Description:" until the end of the text
    description = ad[description_start + len("Description:"):].strip()

    # segment into sentences
    sentences = sent_tokenize(description)
    
    # tokenize each sentence
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sentence) for sentence in sentences]
    
    # merge them into a list of tokens
    tokenized_job_ad = list(chain.from_iterable(token_lists))
    return tokenized_job_ad


# This function extracts the webindex of each job advertisement for use in Task 2:

# In[12]:


def extract_webindex(raw_job_ad):
    ad = raw_job_ad.decode('utf-8')  # Convert the bytes-like object to a Python string
    ad = ad.lower()  # Convert the string to lowercase to make the search case-insensitive
    
    # Find the start of the webindex part using the "webindex:" keyword
    webindex_start = ad.find("webindex:")
    
    if webindex_start == -1:
        # Handle the case where "webindex:" is not found
        return None
        
    # Find the end of the webindex part (assuming it's terminated by a newline character)
    webindex_end = ad.find("\n", webindex_start)
    
    if webindex_end == -1:
        # Handle the case where a newline character is not found after "webindex:"
        return None
        
    # Extract the webindex value
    webindex_value = ad[webindex_start + len("webindex:"):webindex_end].strip()
    
    # Ensure that the extracted value is not empty
    if not webindex_value:
        return None
    
    return webindex_value
    


# This function prints the relevant metrics of our tokenized job advertisements

# In[13]:


def stats_print(tk_job_ads):
    words = list(chain.from_iterable(tk_job_ads)) # we put all the tokens in the corpus in a single list
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of Job Advertisements:", len(tk_job_ads))
    lens = [len(job_ad) for job_ad in tk_job_ads]
    print("Average Job Advertisement length:", np.mean(lens))
    print("Maximun Job Advertisement length:", np.max(lens))
    print("Minimun Job Advertisement length:", np.min(lens))
    print("Standard deviation of Job Advertisement length:", np.std(lens))


# In[14]:


#tokenizing each job ad and extracting the web indexes:
tk_job_ads = [tokenize_job_ad(job_ad) for job_ad in job_ads]  
webindexes = [extract_webindex(job_ad) for job_ad in job_ads]


# In[15]:


#Checking the webindex of our example:
webindexes[example]


# In[16]:


#Checking the tokenized version of our example:
print("Raw Job Advertisement:\n",job_ads[example][0:1000],'\n')
print("Tokenized Job Advertisement:\n",tk_job_ads[example][0:100])


# In[17]:


stats_print(tk_job_ads)


# #### removing single character tokens:

# In[18]:


single_token_list = [[word for word in job_ad if len(word) <= 1 ]                       for job_ad in tk_job_ads] # create a list of single character tokens for each job advertisement
list(chain.from_iterable(single_token_list)) # merge them together in one list


# In[19]:


# Filter out words with length less than 2:
tk_job_ads = [[word for word in job_ad if len(word) >2]                       for job_ad in tk_job_ads]


# In[20]:


#Print the first 10 tokenized words from example:
print("Tokenized Job Advertisement:\n",tk_job_ads[example][0:10])


# In[21]:


# load the stopwords file:
file_path = 'stopwords_en.txt'
# Read the lines of the text file into a list
with open(file_path, 'r') as file:
    data_list = file.readlines()
# Strip newline characters from each line
stopwords = [line.strip() for line in data_list]
stopwords


# In[22]:


# filter out the stop words
tk_job_ads = [[word for word in job_ad if word not in stopwords] for job_ad in tk_job_ads]


# In[23]:


print("Tokenized Job Advertisement without stopwords:\n",tk_job_ads[example][0:100])


# The vocabulary size did not decrease by a lot.

# In[24]:


stats_print(tk_job_ads)


# In[25]:


# Remove words that appear only once in the document collection based on term frequency
words = list(chain.from_iterable(tk_job_ads))
word_counts = Counter(words)
words_to_remove = []
for word, count in word_counts.items():
    if count == 1:
        print(f"{word}: {count}")
        words_to_remove.append(word)


tk_job_ads = [[word for word in job_ad if word not in words_to_remove] for job_ad in tk_job_ads]   


# The operation decreased our vocabulary by almost half:

# In[26]:


stats_print(tk_job_ads)


# #### Remove the top 50 most common words based on document frequency:

# In[27]:


# Initialize a Counter to keep track of document frequency
document_frequency = Counter()
words_to_remove = []

# Count the document frequency of each word
for job_ad in tk_job_ads:
    unique_words = set(job_ad)  # Get unique words in the document
    document_frequency.update(unique_words)

top50 = document_frequency.most_common(50)
words_to_remove = [word for word , frequency in top50]
print(words_to_remove)


# In[28]:


tk_job_ads = [[word for word in job_ad if word not in words_to_remove] for job_ad in tk_job_ads]   
stats_print(tk_job_ads)
print(len(webindexes))


# ## Saving required outputs
# Saving the vocabulary and job advertisement texts as per assignment spectification and saving other files to be used in task 2 and 3.

# In[29]:


#join the tokenized ads into sentences and write them on the file, separating them using new lines.
def save_job_ads(job_ad_filename,tk_job_ads):
    out_file = open(job_ad_filename, 'w')
    string = "\n".join([" ".join(job_ad) for job_ad in tk_job_ads])
    out_file.write(string)
    out_file.close() # close the file
    
save_job_ads('job_ads.txt',tk_job_ads)


# In[30]:


#save the category for each job advertisement. This will be used as the target variable for classification later on.
def save_categories(category_filename,categories):
    out_file = open(category_filename, 'w') 
    string = "\n".join([str(category) for category in categories])
    out_file.write(string)
    out_file.close() # close the file   
    
save_categories('categories.txt',categories)


# In[31]:


# saving the vocabulary
def save_vocab(sorted_vocab):
    out_file = open("vocab.txt", 'w')
    string = "\n".join([word+":"+str(index) for index,word in enumerate(sorted_vocab)])
    out_file.write(string)
    out_file.close() # close the file
    
words = list(chain.from_iterable(tk_job_ads))
vocab = set(words)
sorted_vocab = sorted(vocab)
    
save_vocab(sorted_vocab)


# In[32]:


# saving the webindexes which will be used in task 2.
def save_webindexes(webindexes):
    out_file = open("webindexes.txt", 'w') 
    string = "\n".join([index for index in webindexes])
    out_file.write(string)
    out_file.close() 
    
save_webindexes(webindexes)


# ## Summary
# In this part of the code, I preprocessed the job advertisement data and saved all the relevant files for the next tasks.
