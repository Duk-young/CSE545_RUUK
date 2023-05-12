from transformers import pipeline
import pandas as pd
import warnings 
from time import time
import torch
from torch.utils.data import Dataset, DataLoader
# # df = pd.read_csv('tweets.csv')
# # tweets = df.iloc[:, 2]

# classifier = pipeline("zero-shot-classification", model="bert-base-multilingual-cased", device=0)

# df = pd.read_csv('./preprocessed_indexed/UkraineTweetsPreprocessed2023-04.csv')

# # Extract the text content of each tweet
# tweets = df.iloc[:, 2]

# # Perform classification and update the 4th column
# for i, tweet in enumerate(tweets):
#     result = classifier(tweet, candidate_labels=["positive", "negative", "neutral"])
#     classification = result['labels'][0]
#     df.iloc[i, 3] = classification  # Update the 4th column explicitly

# # Save the updated DataFrame to a new CSV file
# df.to_csv('tweets_with_classification.csv', index=False)
# Specify the device as "cuda:0" to use the first available GPU
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

start = time()
classifier = pipeline("zero-shot-classification", model="bert-base-multilingual-cased", device=0)
warnings.filterwarnings('ignore')
# df = pd.read_csv('./preprocessed_indexed/UkraineTweetsPreprocessed2023-04.csv')

# # Extract the text content of each tweet
# tweets = df.iloc[:, 2]
# print('checkpoint')
# # Perform classification in batches and update the 4th column
# batch_size = 1024  # Adjust as needed based on your GPU memory
# total = len(tweets) / batch_size
# count = 0
# for i in range(0, len(tweets), batch_size):
#     printProgressBar(count, total)
#     batch = tweets[i:i+batch_size]
#     results = classifier(list(batch), candidate_labels=["positive", "negative", "neutral"])
#     count += 1
#     for j, result in enumerate(results):
#         classification = result['labels'][0]
#         df.iloc[i+j, 3] = classification  # Update the 4th column explicitly

# # Save the updated DataFrame to a new CSV file
# df.to_csv('tweets_with_classification.csv', index=False)
# print('Runtime:', time()-start)


class TweetDataset(Dataset):
    def __init__(self, dataframe, text_column):
        self.dataframe = dataframe
        self.text_column = text_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx, self.text_column]

# Specify the device as "cuda:0" to use the first available GPU
classifier = pipeline("zero-shot-classification", model="bert-base-multilingual-cased", device=0)

df = pd.read_csv('./preprocessed_indexed/UkraineTweetsPreprocessed2023-04.csv')

# Create a Dataset from the DataFrame
dataset = TweetDataset(df, 2)
data_loader = DataLoader(dataset, batch_size=1024)  # Adjust batch size as needed
total = len(df) / 1024
count = 0
# Iterate over the DataLoader, classifying batches of tweets
for i, batch in enumerate(data_loader):
    printProgressBar(count, total)
    results = classifier(batch, candidate_labels=["positive", "negative", "neutral"])
    count += 1
    for j, result in enumerate(results):
        classification = result['labels'][0]
        df.iloc[i*len(batch)+j, 3] = classification  # Update the 4th column explicitly

# Save the updated DataFrame to a new CSV file
df.to_csv('tweets_with_classification.csv', index=False)