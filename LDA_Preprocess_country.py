import json
import pyspark as spark
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import preprocessor as p
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

from datetime import datetime
import csv
import pandas as pd
import string
import os
import re

"""
PySpark Pipeline

This code preprocess raw tweets dataset to be used for LDA analysis. 
The code basically cleans up emojis, urls, mentions, stopwords and other texts that are not needed for LDA analysis.
Then, texts are lowercased, lemmatized to boost the quality of analysis.
The code also try to match user-defined location to actual location for later use of plotting sentiment statistics on world map.
"""

# initialize Spark confs
conf = SparkConf()
sc = SparkSession.builder.getOrCreate()

# download missing nltk files before cleaning text
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# load location csv
location_df = pd.read_csv('geolocation_final.csv')
location_rows = location_df.values.tolist()
location_dict = {}

# create location dictionary to use it for user-defined location matching
print('\r Creating Location Dictionary...')
for row in location_rows:
    row_state_code = str(row[3]).lower()
    if row_state_code in location_dict:
        location_dict[row_state_code].append(('',row[4],row[8],row[5],row[6]))
    else:
        location_dict[row_state_code] = [('',row[4],row[8],row[5],row[6])]
    row_country_code = str(row[7]).lower()
    if row_country_code in location_dict:
        location_dict[row_country_code].append(('','',row[8],row[9],row[10]))
    else:
        location_dict[row_country_code] = [('','',row[8],row[9],row[10])]
    row_city_name = row[0].lower()
    row_state_name = row[4].lower()
    row_country_name = row[8].lower()
    
    if row_city_name in location_dict:
        location_dict[row_city_name].append((row[0],row[4],row[8],row[1],row[2]))
    else:
        location_dict[row_city_name] = [(row[0],row[4],row[8],row[1],row[2])]
    if row_state_name in location_dict:
        location_dict[row_state_name].append(('',row[4],row[8],row[5],row[6]))
    else:
        location_dict[row_state_name] = [('',row[4],row[8],row[5],row[6])]
    if row_country_name in location_dict:
        location_dict[row_country_name].append(('','',row[8],row[9],row[10]))
    else:
        location_dict[row_country_name] = [('','',row[8],row[9],row[10])]
print('\r Location Dictionary Created!')
# get list of stop words
stop_words = set(stopwords.words('english'))
stop_words = [x.lower() for x in stop_words]

def preprocess_tweets(text):
    result = []
    # convert all text to lower case
    text = text.lower()

    # preprocess tweet texts. removes URLs, mentions, hashtags, digist, and emojis (and smileys)
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER, p.OPT.EMOJI, p.OPT.SMILEY)
    intmd_result = p.clean(text)
    # remove possible HTML tag leftovers
    intmd_result = intmd_result.replace("&amp", "").replace("\n", "")

    
    # remove all the punctuations from text
    remove_punctuations = str.maketrans('','', string.punctuation)
    intmd_result = intmd_result.translate(remove_punctuations)

    # Initailize Lemmatizer. reduce words to their base form
    lemmatizer = WordNetLemmatizer()
    # Remove Stop words ex) a, the, his, her, etc..
    for word, tag in pos_tag(word_tokenize(intmd_result)):

        # lemmatize first
        if word not in stop_words: # comment this line if want only lemmatize
            pos = ''
            if tag.startswith("NN"):
                pos = 'n'   
            elif tag.startswith('VB'):
                pos = 'v'
            elif tag.startswith('JJ'):
                pos = 'a'
            if pos != '':
                word = lemmatizer.lemmatize(word, pos)
            result.append(word)
    return result

# check if any token in user-defined location matches to real location
def match_location(location):
    location_info = ('','','','','')
    if location == None:
        return location_info
    location = [word.replace('.','').strip() for word in location.split(',')]
    for i in range(len(location)):
        if location[i] in location_dict:
            loc_temp = location_dict[location[i]]
            if len(loc_temp) == 1:
                loc_temp = loc_temp[0]
                if len([x for x in location_info if x == '']) > len([x for x in loc_temp if x == '']):
                    location_info = loc_temp
            else:
                for j in range(i+1, len(location)):
                    for loc_temp_cand in loc_temp:
                        if location[j] in loc_temp_cand:
                            if len([x for x in location_info if x == '']) > len([x for x in loc_temp_cand if x == '']):
                                location_info = loc_temp_cand
    return location_info
    
# Print iterations progress
# customized function referring to https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters/13685020
def printProgressBar (iteration, total, filename, output_filename, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', filename +' ===> '+ output_filename, end = printEnd)
    if iteration == total: 
        print()

# check if tweet text is json applicable (some tweet texts are not formatted correctly)
def json_format_check(text):
    try:
        text = text.replace("\'", "\"")
        json.loads(text)
        return True
    except:
        return False
    
# Compute the popularity score
# indicates the popularity of the tweet
def compute_popularity_score(retweets, favorites):
    try:
        retweets = int(retweets)
        favorites = int(favorites)
        return retweets + favorites
    except:
        return 0
    
# Compute the reach score
# indicates the number of potential viewers of the tweet
def compute_reach_score(followings, followers):
    try:
        followers = int(followers)
        followings = int(followings)
        reach_score = followers - followings
        if reach_score < 0:
            reach_score = 0
        return reach_score
    except:
        return 0
    
directory = './UkraineTweets/'
output_directory = './preprocessed/'
indexed_output_directory ='./preprocessed_indexed/'
indexed_output_filename = ''
filename_format = 'UkraineTweetsPreprocessedForLDA_'
# iterate over files in the directory
total = sum([len(files) for root, dirs, files in os.walk(directory)])
count = 0
total_country_tweets = 0
output_row_dict = {}
output_by_country_row_dict = {}
for root, dirs, files in os.walk(directory):

    # Check the date of the tweets created,
    # and create output filename based on year and month
    year_month_check = ''
    for filename in files:
        output_row_num = 1
        output_by_country_row_num = 1
        output_filename = ''
        output_by_country_filename = ''
        tweet_by_country = {}
        with open(directory + filename, newline='', encoding='UTF-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            column_index = header.index('tweetcreatedts')
            dt = ''
            while(dt == ''):
                try:
                    row = next(reader)
                    date = row[column_index].split('.')[0]
                    dt = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                except StopIteration:
                    break
                except:
                    pass
            if dt == '':
                print(root +'/'+filename, '<=== Failed')
            else:
                year_month = dt.strftime('%Y_%m')
                output_filename_date_format = year_month + '.csv'
                output_by_country_filename_date_format = 'by_country_' +year_month + '.csv'
                output_filename = output_directory + filename_format + output_filename_date_format
                output_by_country_filename = output_directory + filename_format + output_by_country_filename_date_format
                if year_month not in output_row_dict:
                    output_row_dict[year_month] = 1
                if year_month not in output_by_country_row_dict:
                    output_by_country_row_dict[year_month] = 1
                output_row_num = output_row_dict[year_month]
                output_by_country_row_num = output_by_country_row_dict[year_month]
        if output_filename != '':
            count += 1

            # progress bar to check the task progress.
            printProgressBar(count, total, filename=os.path.join(directory + filename), output_filename = output_filename)

            # preprocess files with PySpark
            df = sc.read.csv(directory + filename, header=True, inferSchema=True)
            filtered_df = df.select('location','tweetcreatedts','text','hashtags','following','followers','totaltweets','retweetcount','favorite_count').where((df.language=='en'))
            filtered_df = filtered_df.rdd.filter(lambda rec: json_format_check(rec[3]) == True).map(lambda rec: match_location(rec[0]) + (rec[1], preprocess_tweets(rec[2]), [hashtag_info['text'] for hashtag_info in json.loads(rec[3].replace("\'", "\""))], rec[4], rec[5], rec[6], rec[7], rec[8], compute_reach_score(rec[4], rec[5]),compute_popularity_score(rec[7], rec[8])))
            preprocessed_output = filtered_df.collect()
            with open(output_filename, 'a', newline='', encoding='UTF-8') as f:
                writer = csv.writer(f)
                if os.path.getsize(output_filename) == 0:
                    header = ['id','city', 'state', 'country', 'lat','lon','date', 'text','hashtags','following','followers','total_tweets','retweet_count','favorite_count', 'reach_score', 'popularity_score']
                    writer.writerow(header)
                for row in preprocessed_output:
                        new_row = (output_row_num,) + row
                        writer.writerow(new_row)
                        output_row_num += 1
            output_row_dict[year_month] = output_row_num
            #ANCHOR This is to create preprocessed tweets by country. (Tweets with undefined location will be eliminated)
            preprocessed_output = filtered_df.filter(lambda rec: rec[4] != '' and rec[5] != '').collect()
            with open(output_by_country_filename, 'a', newline='', encoding='UTF-8') as f:
                writer = csv.writer(f)
                if os.path.getsize(output_by_country_filename) == 0:
                    header = ['id','city', 'state', 'country', 'lat','lon','date', 'text','hashtags','following','followers','total_tweets','retweet_count','favorite_count', 'reach_score', 'popularity_score']
                    writer.writerow(header)
                for row in preprocessed_output:
                        new_row = (output_by_country_row_num,) + row
                        writer.writerow(new_row)
                        output_by_country_row_num += 1
            output_by_country_row_dict[year_month] = output_by_country_row_num