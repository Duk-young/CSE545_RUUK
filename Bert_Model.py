from transformers import pipeline
import pandas as pd
import warnings
from time import time
import random
import os

"""
BERT Pipeline

This code performs sentiment analysis on the tweets in the specified directory's 
CSV files using the DistilBERT model, updates the DataFrame with the sentiment 
classification and score, and saves the modified DataFrame to new CSV files.
"""

# Print iterations progress
# customized function referring to https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters/13685020
def printProgressBar(start, iteration, total, filename, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
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
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))  # Calculate the percentage
    # Calculate the filled length of the progress bar
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * \
        (length - filledLength)  # Create the progress bar

    # Print the progress bar with additional information
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', 'Runtime:{:0.2f}'.format(
        time()-start), ' | ', iteration, '/', total, '|', filename, end=printEnd)
    if iteration == total:
        print()


# Get the current time
start = time()

# Define the classifer using the TextClassificationPipeline and DistilBERT model that is finetuned on the SST-2 dataset
# Hyperparameters of the model:
# model: "distilbert-base-uncased-finetuned-sst-2-english"
# num_workers: 8 - When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.
# batch_size: 32
# learning_rate: 5e-5 (0.00005)
# warmup: 600 -> steps
# max_seq_length: 128
# num_train_epochs: 3.0
# activation_function: softmax
classifier = pipeline("text-classification",
                      model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
warnings.filterwarnings('ignore')

# Specify the directory of the dataset
directory = './preprocessed_by_country_bert_indexed'

# Iterate through files in the specified directory
for root, dirs, files in os.walk(directory):
    for filename in files:
        # Read CSV File into a DataFrame
        df = pd.read_csv(root+'/'+filename)

        # Initialize counters and variables
        count = 0
        total = len(df)
        sample_size = 500000

        # If the total number of rows in the DataFrame is larger than the sample size,
        # randomly select a subset of rows
        if total > sample_size:
            random_indices = random.sample(range(total), sample_size)
            df = df.iloc[random_indices]
            total = sample_size

        # Extract the 'text' column from the DataFrame
        tweets = df['text']

        # Add new empty columns for sentiment and sentiment score
        df['sentiment'] = ''
        df['sentiment_score'] = ''

        # Get the index of the last column
        classification_column = len(df.columns)-1

        # Iterate over each row in the 'tweets' column
        for i, tweet in tweets.iteritems():
            # Print progress bar
            printProgressBar(start, count, total, filename)
            try:
                # Perform sentiment analysis/classification on the tweet
                result = classifier(str(tweet))
                count += 1
                # print(result)
                # Extract the classification label and score from the result
                classification = result[0]['label']
                score = result[0]['score']

                # Update the corresponding columns in the DataFrame with the classification
                # Update the 5th column explicitly
                df.loc[i, df.columns[-2]] = classification
                df.loc[i, df.columns[-1]] = score
            except:
                print('Classification failed for ', i, 'th row ')
                pass

        # Save the updated DataFrame to a new CSV file
        df.to_csv('tweets_by_country_with_classification_' +
                  filename[len(filename)-11:len(filename)-4]+'.csv', index=False)
# # Save the updated DataFrame to a new CSV file
print('Runtime:', time()-start)
