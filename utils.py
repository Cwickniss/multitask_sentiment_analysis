import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder

dataset_file = './data/Reviews.csv'
max_sentence_size = 200
boc_size = 10000

postags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",
           "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP",
           "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG",
           "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]

nb_postags = len(postags)

t2k = dict([(v,k) for k, v in enumerate(postags)])
k2t = dict([(k,v) for k, v in enumerate(postags)])

tag_encoder = OneHotEncoder()
tag_encoder.fit(np.arange(nb_postags).reshape(-1, 1))

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

def sent2tags(sentence):
    tags = word_tokenize(sentence + " <EOS>")
    tags = nltk.pos_tag(tags)
    tags = [t2k.get(t[1]) for t in tags if t[1] in postags]
    tags = np.array(tags).reshape(-1, 1)
    tags = tag_encoder.transform(tags).toarray()
    return tags

def tags2sent(tags):
    sent = map(k2t.get, tags)
    sent = " ".join(tags)
    return sent

def batch_generator(batch_size, nb_batches):
    """ Batch generator for the many task joint model.
    """
    batch_count = 0
    dataset = get_dataset(batch_size)

    while True:
        chunk = dataset.get_chunk()

        text = chunk['Text'].apply(sent2vec).values
        tags = chunk['Text'].apply(sent2tags).values
        
        # The sentiment of the review where 1 is positive and 0 is negative
        sent = (chunk['Score'] >= 4).values
        sent = np.int32(sent)

        yield text, tags, sent

        if batch_count >= nb_batches:
            dataset = get_dataset(batch_size)
            batch_count = 0
