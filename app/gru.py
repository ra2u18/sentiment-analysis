import os
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

from tqdm import tqdm
from constants import DATA

from tensorflow import keras

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score



from metrics import f1

MAX_LEN = 70
NB_FEATURES = 50
embedding_dict = {}
glove_path = DATA['glove_encoding_50']

def create_model(word_index, embedding_matrix):
    # GRU with glove embeddings and two dense layers
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        NB_FEATURES,
                        weights=[embedding_matrix],
                        input_length=MAX_LEN,
                        trainable=False))
                        
    model.add(SpatialDropout1D(0.5))

    model.add(GRU(NB_FEATURES, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
    model.add(GRU(NB_FEATURES, dropout=0.25, recurrent_dropout=0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1])

    return model

def create_embedding_dict():
    num_lines = sum(1 for line in open(glove_path, 'r'))
    with open(glove_path, 'r') as f:
        for line in tqdm(f, total=num_lines):
            values=line.split()
            word=values[0]
            vectors=np.asarray(values[1:],'float32')
            embedding_dict[word]=vectors
    f.close()

# Deep learning neural net
def deep_l() -> None:
    # Read glove vectors
    create_embedding_dict()

    # Read train data and model
    train_dataset = pd.read_pickle(DATA['unbalanced_train'])
    train_lb_encoder = preprocessing.LabelEncoder()
    train_y = train_lb_encoder.fit_transform(train_dataset.label.values)

    xtrain, xvalid, ytrain, yvalid = train_test_split(train_dataset.preprocessed_tweets.values,
                        train_y, stratify=train_y, random_state=42, test_size=.1, shuffle=True)

    # we need to binarize the labels for the neural net
    ytrain_enc = np_utils.to_categorical(ytrain)
    yvalid_enc = np_utils.to_categorical(yvalid)

    # Create tokenizer
    token = text.Tokenizer(num_words=None)
    token.fit_on_texts(list(xtrain) + list(xvalid))

    xtrain_seq = token.texts_to_sequences(xtrain)
    xvalid_seq = token.texts_to_sequences(xvalid)

    # zero pad the sequences
    xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=MAX_LEN)
    xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=MAX_LEN)

    word_index = token.word_index

    # create an embedding matrix for the words we have in the dataset
    embedding_matrix = np.zeros((len(word_index) + 1, NB_FEATURES))
    for word, i in word_index.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    #if os.path.isfile('app/model/glove_50.h5'):
        # load the model
        # model = keras.models.load_model('app/model/glove_50.h5', custom_objects={'f1': f1})

    model = create_model(word_index, embedding_matrix)
    # Fit the model with early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, 
        verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])
    model.save('app/model/glove_50.h5')

    # Read test data and evaluate
    test_dataset = pd.read_pickle(DATA['test'])
    test_lb_encoder = preprocessing.LabelEncoder()
    test_y = test_lb_encoder.fit_transform(test_dataset.label.values)

    ytest_enc = np_utils.to_categorical(test_y)

    xtest = test_dataset.preprocessed_tweets.values
    token = text.Tokenizer(num_words=None)
    token.fit_on_texts(list(xtest))

    xtest_seq = token.texts_to_sequences(xtest)
    xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=MAX_LEN)

    model.evaluate(xtest_pad, y=ytest_enc)

if __name__ == '__main__':
    deep_l()