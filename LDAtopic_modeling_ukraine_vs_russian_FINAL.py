# reference : https://www.kaggle.com/code/alifiyabatterywala/ukraine-vs-russian-topic-modeling

"""
The code is a Python script for data analysis on a CSV file with preprocessed tweets related to Ukraine (filenameYY_MM) containing no stop words. It selects a month to analyze.

After reading the file into a DataFrame, it plots the data using Matplotlib and Seaborn to show the frequency of the most common words used in the tweets.

It is important to note that the code does not include the complete analysis and is just a portion of it as the relevant details are not available.
"""

import numpy as np
import pandas as pd 
import os
import csv
import warnings
# data preprocessing / data cleaning
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter(action='ignore')

# load preprocessed data of Ukraine tweets for a specific month
filenameYY_MM = r"/kaggle/input/UkraineTweetsPreprocessed_NoStopWordsYY_MM.csv"
dfYY_MM = pd.read_csv(filenameYY_MM)
# select a specific month to analyze
# perform exploratory data analysis on the selected month's data
df = dfYY_MM

# take texts only from the dataframe
tweets_df = df.iloc[:,2]



###############################################################################
###############################################################################




# N-grams
# Ngrams are simply contiguous sequences of n words
# If the number of words is two, it is called bigram. For 3 words it is called a trigram and so on
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import  Counter

"""
plot_top_ngrams_barchart
inputs : text string and an integer n
task : tokenizes the text and creates a frequency distribution of the n-grams in the text. It then plots a horizontal bar chart showing the top 10 most frequent n-grams along with their frequencies. If there are fewer than 10 unique n-grams in the text, the function plots all of them. The function is useful for visualizing the most common n-grams in a given text, which can provide insights into the language and topics being discussed.
""""
def plot_top_ngrams_barchart(text, n=2):
    new=text.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus) #  count the number of n-grams in corpus
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10] # top 10 most frequent n-grams; tuple; (word, sum_words)

    top_n_bigrams=_get_top_ngram(text,n)[:10] # stores the top 10 in the top_n_bigrams variable.
    
    x,y=map(list,zip(*top_n_bigrams)) # (word, sum_words)
    sns.barplot(x=y,y=x)
import matplotlib as mpl




###############################################################################
###############################################################################



# Topic Modeling
# LDA : create topics along with the probability distribution for each word in our vocabulary for each topic
from sklearn.decomposition import LatentDirichletAllocation
vectorizer = CountVectorizer(
analyzer='word',       
min_df=3,# minimum required occurences of a word 
token_pattern='[a-zA-Z0-9]{3,}',# num chars > 3
max_features=5000,# max number of unique words
                            )
data_matrix = vectorizer.fit_transform(tweets_df.values.astype('U')) # disregard NAN value
data_matrix


lda_model = LatentDirichletAllocation(
                n_components=10, # Number of topics to extract
                learning_method='online', # Use an online variational Bayes algorithm to update the model
                random_state=20, # Set the random seed to ensure reproducibility
                n_jobs = -1 # Use all available CPUs to speed up the training process
                                     )
lda_output = lda_model.fit_transform(data_matrix)




###############################################################################
###############################################################################




# Ploting topics
import pyLDAvis
import pyLDAvis.sklearn
warnings.filterwarnings("ignore", category=DeprecationWarning)

#pyLDAvis extracts information from a fitted LDA topic model to inform an interactive web-based visualization
pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda_model, data_matrix, vectorizer, mds='tsne')

#  top 10 most frequent words from each topic that found by LDA
warnings.filterwarnings("ignore", category=DeprecationWarning)
for i,topic in enumerate(lda_model.components_):
    print('Top 10 words for topic:',i)
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


"""
each row represents a document &
each column represents a topic. 
topic_values = the probability of each topic for the corresponding document, 
where all the probabilities sum up to 1.0 for each "row"
In other words, lda_model.transform applies the trained LDA model to the input data_matrix and 
returns the probabilities of each topic for each document. 
This is useful for topic modeling and other natural language processing tasks where 
the goal is to identify the main themes or topics in a collection of documents.
"""
# adding a new topic column in the dataframe based on the probability value, the suitable topic
# topic_values = lda_model.transform(data_matrix)
# print(topic_values.shape)
# print(topic_values[:10])
# tweets['Topic'] = topic_values.argmax(axis=1)
# tweets.head(3)



###############################################################################
###############################################################################



# save a word cloud image for given topic

from wordcloud import WordCloud
def save_word_cloud(index):
    imp_words_topic=""
    comp = lda_model.components_[index]
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:50]
    for word in sorted_words:
        imp_words_topic=imp_words_topic+" "+word[0]
    wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
    pngname = f"23_02_{index}.png"
    wordcloud = wordcloud.to_file(pngname)
    #     plt.figure( figsize=(5,5))
    #     plt.imshow(wordcloud)
    #     plt.axis("off")
    #     plt.tight_layout()
    #     plt.show()

for i in range(10):
    save_word_cloud(i)
