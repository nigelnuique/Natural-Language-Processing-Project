#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Nigel Nuique
# #### Student ID: s3985410
# 
# Date: September 25, 2023
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * pandas
# * numpy
# * itertools
# * gensim
# * sklearn
# * nltk
# * logging
# 
# ## Introduction
# 
# In this part of the assessment, the files created from the first task were loaded and used to create various feature representations to be used in classifying the job advertisments according to their categories. 
# The primary feature vectors are derived from word embeddings generated using the unweighted and TF-IDF weighted versions of the Google News 300 model. To assess their respective performance, a comparative analysis is conducted through K-folds cross-validation. Following this, a comprehensive evaluation of model performance was conducted, taking into account various levels of information: title-only, description-only, and a combination of both title and description.

# ## Importing libraries 

# In[1]:


import pandas as pd
import numpy as np
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_files 
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import logging


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# ## Bag of Words Model : Count Vector Representation

# #### Loading the files generated from task 1.

# In[2]:


webindexes_file = 'webindexes.txt'
with open(webindexes_file) as f: 
    webindexes = f.read().splitlines() 
    
job_ads_file = 'job_ads.txt'
with open(job_ads_file) as f: 
    joined_job_ad_descs = f.read().splitlines()
    
vocab_desc_file = 'vocab.txt'
with open(vocab_desc_file) as f:
    lines = f.read().splitlines()  # Read all lines from the file
vocab_desc = [line.split(":")[0] for line in lines]


# In[3]:


cVectorizer = CountVectorizer(analyzer = "word",vocabulary = vocab_desc) # initialised the CountVectorizer
count_features = cVectorizer.fit_transform(joined_job_ad_descs)
count_features.shape
count_matrix = count_features.toarray() #converting to dense representation for save operation


# ### Saving outputs
# Save the count vector representation:

# In[4]:


def save_count_vectors(vocab):
    out_file = open("count_vectors.txt", 'w')
    for doc_index, doc_vector in enumerate(count_matrix):
        out_file.write(f"#{webindexes[doc_index]}:,")
        for word_index, word_count in enumerate(doc_vector):
            if word_count > 0:
                word = vocab[word_index]
                out_file.write(f"{word_index}:{word_count},")
        out_file.write("\n")
    out_file.close() # close the file
    
save_count_vectors(vocab_desc)


# ## Models based on word embeddings (Google News 300)

# ### Word2vec embeddings: unweighted

# In[5]:


txt_fname = 'job_ads.txt'
with open(txt_fname) as txtf:
    job_ads = txtf.read().splitlines() # reading a list of strings, each for a job description
tk_job_ads = [job_ad.split(' ') for job_ad in job_ads]

categories_file = 'categories.txt'
with open(categories_file) as f: 
    categories = f.read().splitlines() 
    
example = 10
df = pd.DataFrame({'webindex':webindexes, 'categories': categories,'tk_job_ads':tk_job_ads})
df.iloc[example]


# In[6]:


# logging for event tracking
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
word2vec_googlenews = api.load('word2vec-google-news-300')
print(word2vec_googlenews)


# 
# This code generates vector representations for documents based on word embeddings from the Google News 300 model and stores these representations in a pandas DataFrame.

# In[7]:


def generate_docvecs(word2vec_googlenews,tk_job_ads): # generate vector representation for documents
    docvecs = pd.DataFrame() # creating empty final dataframe
    for i in range(0,len(tk_job_ads)):
        tokens = tk_job_ads[i]
        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = word2vec_googlenews[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                temp = temp.append(pd.Series(word_vec), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        docvec = temp.sum() # take the sum of each column
        docvecs = docvecs.append(docvec, ignore_index = True) # append each document value to the final dataframe
    return docvecs


# In[8]:


# generate the feature vectors
unweighted_desc = generate_docvecs(word2vec_googlenews,df['tk_job_ads'])
unweighted_desc.isna().any().sum() # check whether there is any null values in the document vectors dataframe.


# In[9]:


unweighted_desc.head(5)


# ### Word2vec embeddings: TFIDF weighted

# a TF-IDF vectorizer is initialized with the vocabulary and then used to generate TF-IDF vector representations for the collection of job advertisements' descriptions

# In[10]:


tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab_desc) # initialised the TfidfVectorizer
tfidf_features_desc = tVectorizer.fit_transform(joined_job_ad_descs) # generate the tfidf vector representation for all articles
tfidf_features_desc.shape


# In[11]:


#load vocab as a dictionary (index:word)
vocab_file_desc = 'vocab.txt'
with open(vocab_file_desc) as f:
    lines = f.read().splitlines()  # Read all lines from the file
 
word_indexes = [line.split(":") for line in lines]
vocab_dict_desc = {int(word_index[1]):word_index[0] for word_index in word_indexes}

# Print the first 10 items:
count = 0
for key, value in vocab_dict_desc.items():
    if count < 10:
        print(f"{key}: {value}")
        count += 1
    else:
        break


# This function calculates and stores the TF-IDF weights for words in each document. The function returns a list of dictionaries where each dictionary maps words from vocab_dict to their corresponding TF-IDF weights for a specific document.

# In[12]:


def doc_wordweights(tfidf_features, vocab_dict):
    tfidf_weights = []  # a list to store the word:weight dictionaries of documents

    for doc_index in range(tfidf_features.shape[0]):
        doc_weights = tfidf_features[doc_index].toarray()[0]  # Get TF-IDF weights for the current document
        wordweight_dict = {vocab_dict[word_index]: weight for word_index, weight in enumerate(doc_weights) if weight > 0}
        tfidf_weights.append(wordweight_dict)

    return tfidf_weights

# Call the function with tfidf_features and vocab_dict
tfidf_weights_desc = doc_wordweights(tfidf_features_desc, vocab_dict_desc)

# Print the first 10 items of example:
count = 0
for key, value in tfidf_weights_desc[example].items():
    if count < 10:
        print(f"{key}: {value}")
        count += 1
    else:
        break


# This function computes weighted document vectors by combining word embeddings from a given word embedding model with their corresponding TF-IDF weights for a collection of documents.

# In[13]:


def weighted_docvecs(embeddings, tfidf, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        tf_weights = [float(tfidf[i].get(term, 0.)) for term in valid_keys]
        assert len(valid_keys) == len(tf_weights)
        weighted = [embeddings[term] * w for term, w in zip(valid_keys, tf_weights)]
        docvec = np.vstack(weighted)
        """
        Note: using `sum` here, other 'pooling' options are possible too,
        e.g. mean, etc.
        """
        docvec = np.sum(docvec, axis=0)
        vecs[i,:] = docvec
    return vecs


# In[14]:


weighted_desc = weighted_docvecs(word2vec_googlenews, tfidf_weights_desc, df['tk_job_ads'])
weighted_desc


# ## Task 3. Job Advertisement Classification

# In this task, the performance difference between the unweighted and TF-IDF weighted vector representations made using the Google News 300 model is evaluated.

# ### Unweighted Word2vec vs TF-IDF weighted Word2vec

# In[15]:


seed = 0  # set a seed to make sure the experiment is reproducible
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
X_unweighted = unweighted_desc
X_weighted = weighted_desc
y = df['categories']
model = LogisticRegression(max_iter=2000, random_state=seed)

# Cross-validation for unweighted features
unweighted_scores = cross_val_score(model, X_unweighted, y, cv=kf)

# Cross-validation for weighted features
weighted_scores = cross_val_score(model, X_weighted, y, cv=kf)

# Print performance of each fold
for fold, (unweighted_score, weighted_score) in enumerate(zip(unweighted_scores, weighted_scores)):
    print(f"Fold {fold + 1} accuracy :")
    print(f"  Unweighted: {unweighted_score:.2f}")
    print(f"  Weighted: {weighted_score:.2f}")
    print()

# Calculate and print mean accuracy for each representation
mean_unweighted_accuracy = unweighted_scores.mean()
mean_weighted_accuracy = weighted_scores.mean()
print("Mean Unweighted Feature Representation Accuracy: {:.2f}".format(mean_unweighted_accuracy * 100))
print("Mean Weighted Feature Representation Accuracy: {:.2f}".format(mean_weighted_accuracy * 100))


# The performance of both vector representations where quite similar for each fold with the unweighted feature representation performing slightly better on average.

# ## Comparing Different Levels of information

# In this part of the activity, I compared word embeddings derived from three sources: only the title, only the description, and a combination of both title and description, considering both unweighted and TF-IDF weighted versions.

# #### Unweighted Title Embeddings

# This function tokenizes the title portion of a job advertisement, converting it to lowercase, segmenting it into sentences, and further tokenizing each sentence into individual words, returning a list of these tokens.

# In[16]:


def tokenize_job_ad_title(raw_job_ad):
    ad = raw_job_ad.decode('utf-8') # convert the bytes-like object to python string, need this before we apply any pattern search on it
    ad = ad.lower() # cover all words to lowercase
    
    # Find the start of the title part using the "title:" keyword
    title_start = ad.find("title:")
    if title_start == -1: # Handle the case where "title:" is not found
        return []
    
    # Find the end of the title part (assuming it's terminated by a newline character)
    title_end = ad.find("\n", title_start)
    
    if title_end == -1:
        # Handle the case where a newline character is not found after "title:"
        return []
        
    title = ad[title_start + len("title:"):title_end].strip() # Extract the title value
    sentences = sent_tokenize(title) # segment into sentences
    
    # tokenize each sentence
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sentence) for sentence in sentences]
    
    tokenised_job_ad = list(chain.from_iterable(token_lists)) # merge them into a list of tokens
    return tokenised_job_ad


# In[17]:


#generating unweighted vector representation for just the title
job_data = load_files(r"data") 
job_ads, categories = job_data.data, job_data.target 
tk_job_ad_titles = [tokenize_job_ad_title(job_ad) for job_ad in job_ads]  
print(tk_job_ad_titles)
df_titles = pd.DataFrame({'webindex':webindexes, 'categories': categories,'tk_job_ad_titles':tk_job_ad_titles})
unweighted_titles = generate_docvecs(word2vec_googlenews,df_titles['tk_job_ad_titles'])


# In[18]:


print(unweighted_titles.isna().any().sum()) # check whether there is any null values in the document vectors dataframe.
Nulls = unweighted_titles[unweighted_titles.isna().any(axis=1)]

print(df_titles.iloc[360])
print(df_titles.iloc[572])
print(df_titles.iloc[733])

#drop NAs
unweighted_titles.dropna(axis=0, inplace=True)
# Drop the corresponding rows from df_titles
df_titles.drop(Nulls.index, axis=0, inplace=True)

print(df_titles.shape)


# #### Weighted Title Embeddings

# In[19]:


#combine the tokenized job titles into a list of sentences
joined_job_ad_titles = []
for token_list in tk_job_ad_titles:
    sentence = " ".join(token_list)
    joined_job_ad_titles.append(sentence)
for i in range(10):
    print(joined_job_ad_titles[i])  


# In[20]:


#create the vocabulary and the dictionary
words = list(chain.from_iterable(tk_job_ad_titles))
vocab_title = set(words)
vocab_title = sorted(vocab_title)
vocab_dict_title = {index:word for index, word in enumerate(vocab_title)}
for index in range(10):
    print(vocab_title[index])
for index in range(10):
    print(f"{index}:{vocab_dict_title[index]}")


# In[21]:


#generate the weighted vectore representation
tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab_title) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform(joined_job_ad_titles) # generate the tfidf vector representation for all articles
print(tfidf_features.shape)
tfidf_weights = doc_wordweights(tfidf_features, vocab_dict_title)
print(tfidf_weights[example])
weighted_titles = weighted_docvecs(word2vec_googlenews, tfidf_weights, df_titles['tk_job_ad_titles'])
print(weighted_titles)


# #### Unweighted Title + Description Embeddings

# In[22]:


#concatenate the title and description from before and then generate the unweighted vector representation
df['categories'] = df['categories'].astype('int64')
df_td = pd.merge(df_titles, df, on=['webindex', 'categories'], how='left')
df_td['tk_td'] = df_td.apply(lambda row: row['tk_job_ad_titles'] + row['tk_job_ads'], axis=1)
print(df_td.head(5))
unweighted_td = generate_docvecs(word2vec_googlenews,df_td['tk_td'])


# #### Weighted Title + Desc Embeddings

# In[23]:


#join the tokenized title+description into a list of sentences
#create the vocabulary and dictionary
joined_job_ad_td = [sent1 + " " + sent2 for sent1, sent2 in zip(joined_job_ad_titles, joined_job_ad_descs)]
words = vocab_desc+vocab_title
vocab_td = set(words)
vocab_td = sorted(vocab_td)
vocab_dict_td = {index:word for index, word in enumerate(vocab_td)}
for index in range(10):
    print(vocab_dict_td[index])
for index in range(10):
    print(f"{index}:{vocab_td[index]}")


# In[24]:


tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab_td) # initialised the TfidfVectorizer
tfidf_features_td = tVectorizer.fit_transform(joined_job_ad_td) # generate the tfidf vector representation for all articles
tfidf_features_td.shape


# In[25]:


#generate the weighted vector representation
tfidf_weights_td = doc_wordweights(tfidf_features_td, vocab_dict_td)
print(tfidf_weights_td[example])


# In[26]:


weighted_td = weighted_docvecs(word2vec_googlenews, tfidf_weights_td, df_td['tk_td'])
weighted_td


# ### Comparison of Model Performance:

# In[27]:


# Drop the corresponding rows from weighted and unweighted to match the size
weighted_desc = np.delete(weighted_desc, Nulls.index, axis=0)
unweighted_desc.drop(Nulls.index, axis=0, inplace=True)


# In[28]:


seed = 0  # set a seed to make sure the experiment is reproducible
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

datasets = [
    ("Description - Unweighted", unweighted_desc),
    ("Description - Weighted", weighted_desc),
    ("Title - Unweighted", unweighted_titles),
    ("Title - Weighted", weighted_titles),
    ("Title + Description - Unweighted", unweighted_td),
    ("Title + Description - Weighted", weighted_td)
]

# Iterate through each dataset and feature representation, and perform cross-validation
for dataset_name, X in datasets:
    y = df_td['categories']
    model = LogisticRegression(max_iter=2000, random_state=seed)
    scores = cross_val_score(model, X, y, cv=kf)
    
    # Print performance of each fold
    print(f"{dataset_name}")
    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold} accuracy: {score:.2f}")
    print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
    print()


# Using the description alone yields a consistently high accuracy, with a mean accuracy of 82.28% (unweighted) and 81.63% (weighted).
# When considering only the title, unweighted embeddings perform well with a mean accuracy of 85.12%, but the accuracy drops significantly to 52.40% when using weighted embeddings.
# Combining both title and description information leads to a good accuracy, with unweighted embeddings achieving a mean accuracy of 83.32% but with weighted embeddings only achieving 60.29%.

# ## Summary
# 
# This activity provided valuable hands-on experience in text data classification and the utilization of pretrained models. It underscores the critical aspect of data selection in the training process, demonstrating that adding more data does not always yield better results. Additionally, it highlights that, in this specific context, TF-IDF embeddings may not be the optimal choice, as they led to a significant decrease in performance across most scenarios.

# In[ ]:




