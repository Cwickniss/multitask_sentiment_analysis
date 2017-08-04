import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

dataset_file = './data/Reviews.csv'
max_sentence_size = 200
boc_size = 10000

def get_dataset(batch_size):
    """ Gets the dataset iterator
    """
    dataset = pd.read_csv(dataset_file,
                          iterator=True,
                          chunksize=batch_size)
    return dataset

boc = get_dataset(boc_size).get_chunk()
boc = boc['Text'].values
boc = "".join(boc).lower()
boc = set(boc)
boc = sorted(boc)

# Character Classes Dictionaries
c2k = dict([(v,k) for k, v in enumerate(boc)])
k2c = dict([(k,v) for k, v in enumerate(boc)])

nb_classes = len(boc)

del boc

def word2vec(word):
    vec = map(c2k.get, word.lower())
    vec = list(vec)
    vec = np.array(vec, dtype=np.int32)
    return vec

def vec2word(vec):
    word = map(k2c.get, vec)
    word = "".join(word)
    return word

def sent2vec(sentence):
    words = word_tokenize(sentence + " <EOS>")
    vecs = map(word2vec, words)
    wv = list(vecs)
    return wv

def vec2sent(vec):
    words = map(vec2word, vec)
    sent = " ".join(words)
    return sent

def batch_generator(batch_size, nb_batches):
    """ Batch generator for the many task joint model.
    """
    batch_count = 0
    dataset = get_dataset(batch_size)

    while True:
        chunk = dataset.get_chunk()

        text = chunk['Text'].apply(sent2vec).values
        
        # The sentiment of the review where 1 is positive and 0 is negative
        sent = (chunk['Score'] >= 4).values
        sent = np.int32(sent)

        yield text, sent

        if batch_count >= nb_batches:
            dataset = get_dataset(batch_size)
            batch_count = 0
