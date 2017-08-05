import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder

dataset_file = './data/Reviews.csv'
max_sentence_size = 300
max_word_size = 15
boc_size = 10000

def avg_cross_entropy_loss(predicted, targets):
    losses = []
    length = len(predicted)
    
    for i in range(length):
        target = np.array(targets[i], dtype=np.float32)
        target = torch.from_numpy(target)
        target = Variable(target).long()
        
        loss = F.cross_entropy(predicted[i], target)
        
        losses.append(loss)

    loss = losses[0]
    
    for i in range(1, length):
        loss += losses[i]
    
    loss = loss / length

    return loss

chunk_gram = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
"""

chunk_parser = nltk.RegexpParser(chunk_gram)            

postags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",
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
    words = word_tokenize(sentence)
    vecs = map(word2vec, words)
    vecs = list(vecs)
    vecs = vecs[:max_sentence_size]
    
    return vecs

def vec2sent(vec):
    words = [vec2word(v) for v in vec if type(v) == list]
    sent = " ".join(words)
    
    return sent

def sent2tags(sentence):
    tags = word_tokenize(sentence)
    tags = nltk.pos_tag(tags)
    out = []
    
    for _, tag in tags[:max_sentence_size]:
        if tag in postags:
            out.append(t2k.get(tag))
        else:
            out.append(t2k.get("UKN"))

    return out

def tags2sent(tags):
    sent = map(k2t.get, tags)
    sent = " ".join(tags)
    
    return sent

def sent2chunk(sentence):
    tags = word_tokenize(sentence)
    tags = nltk.pos_tag(tags)
    chunked = chunk_parser.parse(tags)
    out = list()
    
    for chunk in chunked:
        if type(chunk) == nltk.tree.Tree:
            floor = False
            lvl = 0
            
            while not floor:
                sub_tree = []
                
                for chk in chunk:
                    if type(chk) == nltk.tree.Tree:
                        for e in chk:
                            sub_tree.append(e)
                    else:
                        out.append(lvl)
                
                if len(sub_tree) > 0:
                    chunk = sub_tree
                    lvl += 1
                else:
                    floor = True
        else:
            out.append(0)

    return out
    
def batch_generator(batch_size, nb_batches):
    """ Batch generator for the many task joint model.
    """
    batch_count = 0
    dataset = get_dataset(batch_size)
    
    while True:
        chunk = dataset.get_chunk()
        
        text, tags, chunks = [], [], []

        for sent in chunk['Text'].values:
            tags.append(sent2tags(sent))
            text.append(sent2vec(sent))
            chunks.append(sent2chunk(sent))
            
        # The sentiment of the review where 1 is positive and 0 is negative
        sent = (chunk['Score'] >= 4).values
        sent = np.int32(sent)

        yield text, tags, chunks, sent

        batch_count += 1
        
        if batch_count >= nb_batches:
            dataset = get_dataset(batch_size)
            batch_count = 0

gen = batch_generator(10, 100)

text, tags, chunks, sent = next(gen)
