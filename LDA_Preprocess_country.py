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

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter


from datetime import datetime
import csv
import pandas as pd
import string
import os
import re

# initialize Spark confs
conf = SparkConf()
sc = SparkSession.builder.getOrCreate()

# download missing nltk files
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

location_df = pd.read_csv('geolocation_final.csv')
location_rows = location_df.values.tolist()
location_dict = {}
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
# def word_match(word, text):
#     pattern = r'([A-Za-z]'+word+'|'+word+'[A-Za-z]'+'|[A-Za-z]'+word+'[A-Za-z])'
#     word_match = False
#     if re.search(pattern, text) == False:
#         word_match = True
#     if word_match == True and word in text:
#         return True
#     else:
#         return False
# def match_location(location):
#     city = state = country = lat = lon = ''
#     # [0 ~ 2] "city_name", "city_latitude","city_longitude", 
#     # [3 ~ 6] "state_code", "state_name","state_latitude","state_longitude",
#     # [7 ~ 10] "country_code", "country_name","country_latitude","country_longitude"
#     if location == None:
#         return (city, state, country, lat, lon)
#     location = location.lower().replace(".", "")
#     for row in location_rows:
#         row_city_name = row[0].lower()
#         row_state_code = str(row[3]).lower()
#         row_state_name = row[4].lower()
#         row_country_code = str(row[7]).lower()
#         row_country_name = row[8].lower()
#         if row_city_name in location or row_state_code in location or row_state_name in location or row_country_code in location or row_country_name in location:
#             if city == '' and word_match(row_city_name, location):
#                 city = row[0]
#                 lat = row[1] # city lat
#                 lon = row[2] # city lon
#                 state = row[4]
#                 country = row[8]
#                 break
#             elif state == '' and (word_match(row_state_code, location) or word_match(row_state_name, location)):
#                 state = row[4]
#                 lat = row[5] # state lat
#                 lon = row[6] # state lon
#                 country = row[8]
#             elif country == '' and (word_match(row_country_code, location) or word_match(row_country_name, location)):
#                 country = row[8]
#                 lat = row[9] # country lat
#                 lon = row[10] # country lon
#     return (city, state, country, lat, lon)
# def get_country_state(location):
#     # Geocoding tweet location data
#     try:
#         geolocator = Nominatim(user_agent="Tweet_Preprocessor")
#         geo_data = geolocator.geocode(location, timeout=10)
#         if geo_data:
#             lat = geo_data.latitude
#             lon = geo_data.longitude
#             reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
#             address = reverse((geo_data.latitude, geo_data.longitude), timeout=10).raw['address']
#             state = address.get('state', '')
#             country = address.get('country', '')
#             return (state, country,  lat, lon)
#     except GeocoderTimedOut:
#         return get_country_state(location)
    
# Print iterations progress
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
    
directory = './UkraineTweets/'
output_directory = './preprocessed_by_country/'
indexed_output_directory ='./preprocessed_by_country_indexed/'
output_filename = ''
indexed_output_filename = ''
filename_format = 'UkraineTweetsPreprocessedByCountry_NoStopWords'
# iterate over files in the directory
total = sum([len(files) for r, d, files in os.walk(directory)])
count = 0
total_country_tweets = 0
for root, dirs, files in os.walk(directory):
    # Check the date of the tweets created,
    # and create output filename based on year and month
    for filename in files:
        tweet_by_country = {}
        with open(directory + filename, newline='', encoding='UTF-8') as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
            date = row[10][:19]
            dt = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            year_month = dt.strftime('%Y-%m')
            output_filename_date_format = year_month + '.csv'
            indexed_output_filename_date_format = year_month+'_indexed' + '.csv'
        count += 1
        # progress bar to check the task progress.
        printProgressBar(count, total, filename=os.path.join(directory + filename), output_filename = 'by Country')

        # preprocess files
        df = sc.read.csv(directory + filename, header=True, inferSchema=True)
        filtered_df = df.select('location','tweetcreatedts','text','hashtags').where((df.language=='en'))
        filtered_df = filtered_df.rdd.filter(lambda rec: json_format_check(rec[3]) == True).map(lambda rec: (match_location(rec[0]), rec[1], preprocess_tweets(rec[2]), [hashtag_info['text'] for hashtag_info in json.loads(rec[3].replace("\'", "\""))]))
        preprocessed_output = filtered_df.filter(lambda rec: len([x for x in rec[0] if x == '']) < 5).collect()
        total_country_tweets += len(preprocessed_output)
        for tweet in preprocessed_output:
            country = tweet[0][2]
            if country not in tweet_by_country:
                tweet_by_country[country] = [tweet]
            else:
                tweet_by_country[country].append(tweet)
        for country in tweet_by_country.keys():
            country_directory = output_directory + country + '/'
            output_filename = country_directory + filename_format + country + output_filename_date_format
            # indexed_output_filename = indexed_output_directory + filename_format + country + indexed_output_filename_date_format
            if not os.path.exists(country_directory):
                os.makedirs(country_directory)
            with open(output_filename, 'a', newline='', encoding='UTF-8') as f:
                writer = csv.writer(f)
                writer.writerows(tweet_by_country[country])
        # # preprocessed file appended to the matching year and month output file.
print('total country tweets:', total_country_tweets)
# below code adds row number for the preprocessed file
# total = sum([len(files) for r, d, files in os.walk(output_directory)])
# count = 0
# for root, dirs, files in os.walk(output_directory):
#     for filename in files:
#         count += 1
#         printProgressBar(count, total, filename=os.path.join(filename), output_filename = output_filename)
#         with open(output_directory + filename, 'r', newline='', encoding='UTF-8') as f_in:
#             reader = csv.reader(f_in)
#             with open(indexed_output_directory + filename, 'w', newline='', encoding='UTF-8') as f_out:
#                 writer = csv.writer(f_out)
#                 for i, row in enumerate(reader, start=1):
#                     new_row = [i] + row
#                     writer.writerow(new_row)

# total = sum([len(files) for r, d, files in os.walk(output_directory)])
# count = 0
# for root, dirs, files in os.walk(output_directory):
#     with open(indexed_output_directory + 'UkraineTweetsPreprocessed2022-2023.csv', 'w', newline='', encoding='UTF-8') as f_out:
#         writer = csv.writer(f_out)
#         for filename in files:
#             count += 1
#             printProgressBar(count, total, filename=os.path.join(filename), output_filename = indexed_output_directory + 'UkraineTweetsPreprocessed2022-2023.csv')
#             with open(output_directory + filename, 'r', newline='', encoding='UTF-8') as f_in:
#                 reader = csv.reader(f_in)
#                 for i, row in enumerate(reader, start=1):
#                     writer.writerow(row)
# df = sc.read.csv('./UkraineTweets/0829_UkraineCombinedTweetsDeduped.csv', header=True, inferSchema=True)
# filtered_df = df.where((df.language=='en'))
# rows = filtered_df.collect()
# for row in rows:
#     un = row["username"]
#     ht = row['hashtags']
#     print(row)
#     print("\n")
#     json_load(ht.replace("\'", "\""))