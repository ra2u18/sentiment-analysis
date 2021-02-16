import os
import pickle
import pandas as pd

from pathlib import Path
from constants import DATA
from utils import guess_language, preprocess_tweet

from tqdm import tqdm

# Training data path
test_path = Path(DATA['test_path'])
# Collect the training data as pandas dataframe
test_df = pd.read_csv(test_path, error_bad_lines=False, delimiter='\t')

def preprocess_test() -> None:
    if not os.path.isfile(DATA['test']):
        print('Preprocessing the data, please wait...')
        preprocess_data()
        print('Finished preprocessing!')
    
    print('Preprocessed test dataset statistics.\n')
    test = pd.read_pickle(DATA['test'])
    print(f'Test \ndataset shape {test.shape} \n unique classes {test.label.value_counts()} \n')
    print('\n End of preprocessed train dataset statistics')   

# Preprocess data and save it in another file
def preprocess_data():
    # Set humor labels to fake
    test_df.loc[test_df['label']=='humor', 'label'] = 'fake'

    # Find tweets in english only  
    test_lang = [guess_language(x) for x in tqdm(test_df['tweetText'])]
    test_df['language'] = test_lang
    
    english_mask = test_df['language'] == 'en'
    test_en_df = test_df[english_mask].copy()

    prep_tweets = []

    # Preprocess tweets
    for tweet in tqdm(test_en_df['tweetText']):
        prep_tweets.append(preprocess_tweet(test_en_df, tweet))
    
    test_en_df['preprocessed_tweets'] = prep_tweets

    # save model
    test_en_df.to_pickle(DATA['test'])

