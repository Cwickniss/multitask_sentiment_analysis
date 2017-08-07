import nltk
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.autograd import Variable
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder

dataset_file = './data/Reviews.csv'

max_sentence_size = 300
max_word_size = 15

# The number of lines to read from the dataset to build
# key value dicts
boc_size = 10000

def avg_cross_entropy_loss(predicted, targets):
    """ Helper function for computing the simple mean
        cross entropy loss between the predicted one-hot
        and the target class.
    """
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

# The chunk gram regex for the chunking parser
chunk_gram = r"""
NP: {<DT|JJ|NN.*>+}             # Chunk sequences of DT, JJ, NN
PP: {<IN><NP>}                  # Chunk prepositions followed by NP
VP: {<VB.*><NP|PP|CLAUSE>+$}    # Chunk verbs and their arguments
ACTION: {<PRP><VBP><VBN>}
CLAUSE: {<NP><VP>}              # Chunk NP, VP
SENT: {<.*>+}
      }<.>{
"""

chunk_tags = ['NP', 'PP', 'VP', 'ACTION', 'CLAUSE', 'SENT', 'S']

nb_chunktags = len(chunk_tags)

chk2k = dict([(v,k) for k, v in enumerate(chunk_tags)])
k2chk = dict([(k,v) for k, v in enumerate(chunk_tags)])

chunk_parser = nltk.RegexpParser(chunk_gram)            

# Postags to consider
postags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",
           "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP",
           "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG",
           "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "UKN"]

nb_postags = len(postags)

t2k = dict([(v,k) for k, v in enumerate(postags)])
k2t = dict([(k,v) for k, v in enumerate(postags)])

def get_dataset(batch_size, skip=0):
    """ Gets the dataset iterator
    """
    dataset = pd.read_csv(dataset_file,
                          iterator=True,
                          skiprows=skip * batch_size,
                          chunksize=batch_size)
    return dataset

# Bag of Chars
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
    """ Converts a word to its char-vector.
    """
    vec = map(c2k.get, word.lower())
    vec = list(vec)
    
    return vec

def vec2word(vec):
    """ Converts a char-vector to its respective
        string.
    """
    word = map(k2c.get, vec)
    word = "".join(word)
    
    return word

def sent2vec(sentence):
    """ Returns the char-vector word representation
        of a given sentence.
    """
    words = word_tokenize(sentence)
    vecs = map(word2vec, words)
    vecs = list(vecs)
    vecs = vecs[:max_sentence_size]
    
    return vecs

def vec2sent(vec):
    """ Converts a char-vector word representation
        sentence to its respective string.
    """
    words = [vec2word(v) for v in vec if type(v) == list]
    sent = " ".join(words)
    
    return sent

def sent2tags(sentence):
    """ Returns a vector of tag-classes from a given
        sentence.
    """
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
    """ Returns the string representation of a given
        vector of tag-classes.
    """
    sent = map(k2t.get, tags)
    sent = " ".join(tags)
    
    return sent

def sent2chunk(sentence):
    """ Returns the chunking classes of a given 
        sentence.
        
        Given: 
        I like this product, but I wouldn't recommend it.
        
        The chunk tree is formed such as:
        S                   S                             S
                            ,                             .
        SENT SENT             SENT SENT SENT SENT SENT
        I like
               NP NP
               this product
                              but I wouldn't recomment it
        
        Then, the resulted vector would be:
        [SENT, SENT, NP, NP, S, SENT, SENT, SENT, SENT, SENT, S]
    """
    # Get pos tags
    tags = word_tokenize(sentence)
    tags = nltk.pos_tag(tags)
    
    # Chunks it
    chunked = chunk_parser.parse(tags)
    
    flatten = [chunk for chunk in chunked]
    out = ['S'] * len(tags)
    pointer = 0
    
    # Parses all sublevel tree in order to flatten it
    # to its respective chunk-tags.
    while pointer < len(out):
        item = flatten[pointer]
        
        if type(item) == tuple:
            if type(item[1]) == nltk.tree.Tree:
                item = item[1]
            else:
                if item[0] in chunk_tags:
                    out[pointer] = item[0]
                pointer += 1
                continue
        
        flatten = flatten[:pointer] + flatten[pointer+1:]
        for i, leaf in enumerate(item):
            flatten = flatten[:pointer+i] + [(item._label, leaf)] + flatten[pointer+i:]
    
    out = [chk2k.get(chk) for chk in out]

    return out

def batch_generator(batch_size, nb_batches, skip_batches=0):
    """ Batch generator for the many task joint model.
    """
    batch_count = 0
    dataset = get_dataset(batch_size, skip_batches)
    
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
