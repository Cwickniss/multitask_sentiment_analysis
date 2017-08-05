import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder

dataset_file = './data/Reviews.csv'
max_sentence_size = 200
max_word_size = 30
boc_size = 10000

postags = ["EMPTY", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",
           "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP",
           "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG",
           "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "UKN"]

nb_postags = len(postags)

t2k = dict([(v,k) for k, v in enumerate(postags)])
k2t = dict([(k,v) for k, v in enumerate(postags)])

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
boc = ['empty'] + boc

# Character Classes Dictionaries
c2k = dict([(v,k) for k, v in enumerate(boc)])
k2c = dict([(k,v) for k, v in enumerate(boc)])

nb_classes = len(boc) + 1

del boc

def word2vec(word):
    vec = map(c2k.get, word.lower())
    vec = list(vec)
    return vec

def vec2word(vec):
    word = map(k2c.get, vec)
    word = "".join([c for c in word if c != 'empty'])
    return word

def sent2vec(sentence):
    words = word_tokenize(sentence)
    vecs = map(word2vec, words)
    
    sent = np.zeros((len(sentence), max_word_size))
    
    for i, vec in enumerate(vecs):
        for j, char in enumerate(vec[:max_word_size]):
            sent[i][j] = char
    
    return sent

def vec2sent(vec):
    words = map(vec2word, vec)
    sent = " ".join(words)
    return sent

def sent2tags(sentence):
    tags = word_tokenize(sentence[:max_sentence_size])
    tags = nltk.pos_tag(tags)
    
    out = np.zeros((max_sentence_size))
    
    for k, tag in enumerate(tags):
        if tag in postags:
            out[k] = t2k.get(tag)
        else:
            out[k] = t2k.get("UKN")
    
    return out

def tags2sent(tags):
    sent = map(k2t.get, tags)
    sent = " ".join(tags)
    return sent

def batch_generator(batch_size, nb_batches):
    """ Batch generator for the many task joint model.
    """
    batch_count = 0
    dataset = get_dataset(batch_size)
    eos_vec = sent2vec(" <EOS>")
    
    while True:
        chunk = dataset.get_chunk()
        
        text = np.zeros((batch_size, max_sentence_size, max_word_size), dtype=np.int32)
        tags = np.zeros((batch_size, max_sentence_size), dtype=np.int32)

        for i, sent in enumerate(chunk['Text'].values):
            tags[i] = sent2tags(sent)
            vecs = sent2vec(sent)
            
            for j, vec in enumerate(vecs[:max_sentence_size - 6]):
                text[i][j] = vec
            
            for k in range(6):
                j += 1
                text[i][j] = eos_vec[k]

        # The sentiment of the review where 1 is positive and 0 is negative
        sent = (chunk['Score'] >= 4).values
        sent = np.int32(sent)

        yield text, tags, sent

        if batch_count >= nb_batches:
            dataset = get_dataset(batch_size)
            batch_count = 0
