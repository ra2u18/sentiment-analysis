import os
import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from constants import DATA
from utils import guess_language, preprocess_tweet

from tqdm import tqdm

# Training data path
train_path = Path(DATA['train_path'])
# Collect the training data as pandas dataframe
train_df = pd.read_csv(train_path, error_bad_lines=False, delimiter='\t')

def preprocess_train() -> None: 

    if not (os.path.isfile(DATA['unbalanced_train']) and os.path.isfile(DATA['balanced_train'])):
        print('Preprocessing the data, please wait...')
        preprocess_data()
        print('Finished preprocessing!')

    print('Preprocessed train dataset statistics.\n')
    unbalanced_train = pd.read_pickle(DATA['unbalanced_train'])
    balanced_train = pd.read_pickle(DATA['balanced_train'])
    print(f'Unbalanced train \ndataset shape {unbalanced_train.shape} \n unique classes {unbalanced_train.label.value_counts()} \n')
    print(f'Balanced train \ndataset shape {balanced_train.shape} \n unique classes {balanced_train.label.value_counts()}')
    print('\n End of preprocessed train dataset statistics')    
        
# Preprocess data and save it in another file
def preprocess_data():
    # Set humor labels to fake
    train_df.loc[train_df['label']=='humor', 'label'] = 'fake'

    # Find tweets in english only  
    train_lang = [guess_language(x) for x in tqdm(train_df['tweetText'])]
    train_df['language'] = train_lang
    
    english_mask = train_df['language'] == 'en'
    train_en_df = train_df[english_mask].copy()

    prep_tweets = []

    # Preprocess tweets
    for tweet in tqdm(train_en_df['tweetText']):
        prep_tweets.append(preprocess_tweet(train_en_df, tweet))
    
    train_en_df['preprocessed_tweets'] = prep_tweets

    # Save both unbalanced and balanced datasets.
    train_en_df.to_pickle(DATA['unbalanced_train'])

    # Balance the data
    balanced_en_df = train_en_df.copy()
    # Shuffle dataset
    balanced_en_df = balanced_en_df.sample(frac=1, random_state=42)
    # Put all real twitter posts in a separate datasets
    real_en_df = balanced_en_df.loc[balanced_en_df['label'] == 'real']
    false_en_df = balanced_en_df.loc[balanced_en_df['label']=='fake'].sample(n=len(real_en_df), random_state=42)

    normalized_en_df = pd.concat([real_en_df, false_en_df])

    # save model
    normalized_en_df.to_pickle(DATA['balanced_train'])

