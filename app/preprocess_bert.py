from pathlib import Path
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import guess_language, preprocess_tweet_bert

# BERT preprocess
from pytorch_pretrained_bert import BertTokenizer

test_path = Path('data/test.txt')
train_path = Path('data/train.txt')

# preprocess bert
def preprocess_train_valid_bert():
    # load test data
    train_df = pd.read_csv(train_path, error_bad_lines=False, delimiter='\t', encoding='utf-8')

    train_df.loc[train_df['label']=='humor', 'label'] = 'fake'

    languages = [guess_language(x) for x in tqdm(train_df['tweetText'])]
    train_df['language'] = languages
    train_en_df = train_df[train_df['language'] == 'en'].copy()

    train_en_df.drop(['imageId(s)', 'username', 'timestamp', 'userId'], axis=1, inplace=True)

    prep_tweets = [preprocess_tweet_bert(x) for x in tqdm(train_en_df['tweetText'])]
    
    train_en_df['preprocessed_text'] = prep_tweets

    train_en_df['preprocessed_text_length'] = [len(text.split(' ')) for text in train_en_df['preprocessed_text']]

    mask_less = train_en_df['preprocessed_text_length'] < 85
    mask_more = train_en_df['preprocessed_text_length'] > 3

    train_en_df = train_en_df[(mask_less) & (mask_more)]

    # drop duplicates
    train_en_df = train_en_df.drop_duplicates(subset=['preprocessed_text'])

    train_en_df['preprocessed_text_bert'] = '[CLS] ' + train_en_df['preprocessed_text']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_en_df['preprocessed_text_bertbase_length'] = [len(tokenizer.tokenize(sent)) for sent in train_en_df['preprocessed_text_bert']]

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    train_en_df['preprocessed_text_bertlarge_length'] = [len(tokenizer.tokenize(sent)) for sent in train_en_df['preprocessed_text_bert']]

    label_dict = dict()
    for i, l in enumerate(list(train_en_df['label'].value_counts().keys())):
        label_dict.update({l : i})

    train_en_df['information_label'] = [label_dict[label] for label in train_en_df['label']]

    train, valid = train_test_split(train_en_df, test_size=0.25, random_state=42, stratify=train_en_df.information_label, shuffle=True)

    train.to_csv('bert/train_bert.csv')
    valid.to_csv('bert/valid_bert.csv')

def preprocess_test_bert():
    # load test data
    test_df = pd.read_csv(test_path, error_bad_lines=False, delimiter='\t', encoding='utf-8')

    languages = [guess_language(x) for x in tqdm(test_df['tweetText'])]
    test_df['language'] = languages
    test_en_df = test_df[test_df['language'] == 'en'].copy()

    test_en_df.drop(['imageId(s)', 'username', 'timestamp', 'userId'], axis=1, inplace=True)

    prep_tweets = [preprocess_tweet_bert(x) for x in tqdm(test_en_df['tweetText'])]
    test_en_df['preprocessed_text'] = prep_tweets

    test_en_df['preprocessed_text_length'] = [len(text.split(' ')) for text in test_en_df['preprocessed_text']]

    mask_less = test_en_df['preprocessed_text_length'] < 85
    mask_more = test_en_df['preprocessed_text_length'] > 3

    test_en_df = test_en_df[(mask_less) & (mask_more)]

    # drop duplicates
    test_en_df = test_en_df.drop_duplicates(subset=['preprocessed_text'])

    test_en_df['preprocessed_text_bert'] = '[CLS] ' + test_en_df['preprocessed_text']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_en_df['preprocessed_text_bertbase_length'] = [len(tokenizer.tokenize(sent)) for sent in test_en_df['preprocessed_text_bert']]

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    test_en_df['preprocessed_text_bertlarge_length'] = [len(tokenizer.tokenize(sent)) for sent in test_en_df['preprocessed_text_bert']]

    label_dict = dict()
    for i, l in enumerate(list(test_en_df['label'].value_counts().keys())):
        label_dict.update({l : i})

    test_en_df['information_label'] = [label_dict[label] for label in test_en_df['label']]

    test_en_df.to_csv('bert/test_bert.csv')

if __name__ == '__main__':
    print('Preprocessing train and valid')
    preprocess_train_valid_bert()
    print('Preprocessing test')
    preprocess_test_bert()