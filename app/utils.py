import re
import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

# Check language library
from constants import EMOTICONS, UNICODE_EMO

from langdetect import detect
from spellchecker import SpellChecker
from collections import Counter

# CONSTANTS
language_punctuation = (string.punctuation + '¿“”»«•').replace("'", "")
preprocess_tweet_punc = string.punctuation.replace('#', '')

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

rare_words = set()
spell = SpellChecker()
STOPWORDS = set(stopwords.words('english'))


lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
# END CONSTANTS

''' Convert emoticons to words '''
def convert_emoticons(tweet):
    for emot in EMOTICONS:
        tweet = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), tweet)
    return tweet

''' Convert emojis to words '''
def convert_emojis(tweet):
    for emot in UNICODE_EMO:
        tweet = re.sub(u'('+re.escape(emot)+')',' ' + UNICODE_EMO[emot].replace(":","") + ' ', tweet)
    return tweet

''' Convert more than 2 letter repetitions to 2 letter; funnnnny --> funny '''
def preprocess_word(word):
    word = re.sub(r'(.)\1+', r'\1\1', word)
    return word

''' Bring words to their simplest form '''
def lemmatize_words(tweet):
    pos_tagged_text = nltk.pos_tag(tweet.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN))
                    for word, pos in pos_tagged_text])

''' Custom function to remove the rare words '''
def remove_rarewords(tweet, rare_words):
    words = str(tweet).split()
    result = ""
    
    for word in words: 
        if word not in rare_words:
            result = result + word + " "
    return result.strip()
    #return " ".join([word for word in str(tweet).split() if word not in RAREWORDS])

''' Remove rare words '''
def find_rare_words(data, n_rare_words):
    cnt = most_common_words_c(data)
    return set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])

''' Custom function to remove the stopwords '''
def remove_stopwords(tweet):
    return " ".join([word for word in str(tweet).split() if word not in STOPWORDS])

''' Spelling corrector '''
def correct_spellings(tweet):
    corrected_text = []
    misspelled_words = spell.unknown(tweet.split())
    for word in tweet.split():
        if is_valid_word(word) and word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

''' Search for most common words '''
def most_common_words_c(data):
    cnt = Counter()
    for text in data['tweetText'].values:
        for word in text.split():
            cnt[word] += 1
    return cnt

''' Custom function to check for valid words '''
def is_valid_word(word: str) -> bool:
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)

''' Custom function to remove the punctuation '''
def remove_punctuation(tweet: str, punctuation: str) -> str:
    return tweet.translate(str.maketrans('', '', punctuation))

''' Remove emojis '''
def de_emojify(tweet: str):
    return emoji_pattern.sub(r'', tweet)

def guess_language(tweet: str) -> str:
    # Lower case
    tweet = tweet.lower()
    # Remove URLs, User mentions, RT
    tweet = re.sub(r'https?://\S+', '', tweet)
    tweet = re.sub(r'@[\S]+', '', tweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Remove emojis and strip end and beginning white spaces
    tweet = de_emojify(tweet).strip()
    # Remove punctuation
    tweet = remove_punctuation(tweet, language_punctuation)
    # Filter out non-valid words
    valid_words = []

    try:
        words = tweet.split()
        for word in words:
            if is_valid_word(word):
                valid_words.append(word)
        normalized_tweet = ' '.join(valid_words)
        language = detect(normalized_tweet)
    except:
        language = 'unknown'
        #print(f'This row throws an error\n {normalized_tweet}')
    
    return language

def preprocess_tweet_bert(tweet):
    preprocessed_tweet = []

    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'https?://\S+', '', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', '', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)

    tweet = re.sub(r'&amp;?',r'and', tweet)
    tweet = re.sub(r'&lt;',r'<', tweet)
    tweet = re.sub(r'&gt;',r'>', tweet)

    # Convert emoticons to words
    tweet = convert_emoticons(tweet)
    # Convert emoticons to words
    tweet = convert_emojis(tweet)
    
    for word in tweet.split():
        word = preprocess_word(word)
        preprocessed_tweet.append(word)
    
    tweet = ' '.join(preprocessed_tweet)
    
    tweet = re.sub(r'([\w\d]+)([^\w\d ]+)', r'\1 \2', tweet)
    tweet = re.sub(r'([^\w\d ]+)([\w\d]+)', r'\1 \2', tweet)

    return tweet.strip()


def preprocess_tweet(data, tweet):
    preprocessed_tweet = []

    global rare_words
    
    if len(rare_words) == 0:
        rare_words = find_rare_words(data, 10)

    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'https?://\S+', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Remove punctuation
    tweet = remove_punctuation(tweet, preprocess_tweet_punc)
    # Spell check
    tweet = correct_spellings(tweet)
    # Remove hashes
    tweet = tweet.replace('#', '')
    # Remove stopwords
    tweet = remove_stopwords(tweet)
    # Remove rare words // expensive operation, compute it once
    tweet = remove_rarewords(tweet, rare_words)
    # Lemmatize words
    tweet = lemmatize_words(tweet)
    # Convert emoticons to words
    tweet = convert_emoticons(tweet)
    # Convert emoticons to words
    tweet = convert_emojis(tweet)
    
    # Check if the words are valid, remove malformed words
    words = tweet.split()
    
    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            preprocessed_tweet.append(word)
            
    output = ' '.join(preprocessed_tweet)
    
    # Lower the output again
    return output.lower()