import os 
import pickle
import numpy as np
from tqdm import tqdm

from constants import DATA

with open(DATA['glove_encoding_200']) as f:
    words = f.readlines()

word2id = dict()
embeddings = []

for i, w in tqdm(enumerate(words)):
    w_list = w.split(' ')
    word2id.update({w_list[0]: i})
    embeddings.append([float(j) for j in w_list[1:]])

embeddings = np.array(embeddings)
pickle.dump(word2id, open('glove_word2id', 'wb'))
np.save('glove_embeddings', embeddings)
