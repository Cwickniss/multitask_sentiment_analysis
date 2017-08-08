import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
from utils import batch_generator
from utils import nb_classes
from utils import nb_postags
from utils import nb_chunktags
from utils import max_sentence_size
from utils import vec2word
from lang_model import CharacterLanguageModel
from lang_model import embedding_size
from pos_tag import POSTag
from chunking import Chunking

class Dependency(nn.Module):
    def __init__(self, hidden_state_size, 
                       nb_rnn_layers, 
                       tags_hidden_state_size,
                       chk_hidden_state_size):
        super(Dependency, self).__init__()
        
        self.input_size = embedding_size \
                        + nb_postags \
                        + nb_chunktags \
                        + tags_hidden_state_size * 2 \
                        + chk_hidden_state_size * 2

        self.w = nn.Parameter(torch.randn(nb_rnn_layers * 2, 
                                          max_sentence_size, 
                                          hidden_state_size))
        self.h = nn.Parameter(torch.randn(nb_rnn_layers * 2, 
                                          max_sentence_size,
                                          hidden_state_size))
        
        self.bi_lstm = nn.LSTM(self.input_size, 
                               hidden_state_size,
                               nb_rnn_layers,
                               bidirectional=True)
        
        self.wd = nn.Parameter(torch.randn(hidden_state_size * 2))
        self.fc = nn.Linear(hidden_state_size * 2, 1)
        
    def matching(self, t, j):
        m = t * (self.wd * j)
        return self.fc(m.view(1, -1))
    
    def forward(self, x, tags, hn_tags, chunks, hn_chunks):
        tags = tags.view(1, -1, nb_postags)
        chunks = chunks.view(1, -1, nb_chunktags)
        
        gt = torch.cat([hn_chunks, hn_tags, x, tags, chunks], dim=2)

        out, hn = self.bi_lstm(gt, (self.h[:,:x.size(1),:], 
                                    self.w[:,:x.size(1),:]))
        
        scores = torch.zeros(x.size(1), x.size(1))
        scores = Variable(scores)
        
        for i in range(x.size(1)):
            for j in range(i, x.size(1)):
                scores[i,j] = self.matching(out[0,i], out[0,j])
        
        return scores, out

chunking_hidden_state_size = 200
chunking_nb_rnn_layers = 2

postag_hidden_state_size = 100
postag_nb_rnn_layers = 2

dependency_hidden_state_size = 100
dependency_nb_rnn_layers = 2

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

        self.lang_model = CharacterLanguageModel()

        self.pos_tag = POSTag(postag_hidden_state_size, 
                              postag_nb_rnn_layers)

        self.chunking = Chunking(chunking_hidden_state_size, 
                                 chunking_nb_rnn_layers, 
                                 postag_hidden_state_size)
        
        self.dependency = Dependency(dependency_hidden_state_size, 
                                     dependency_nb_rnn_layers, 
                                     postag_hidden_state_size,
                                     chunking_hidden_state_size)

        self.pos_tag.load_state_dict(torch.load('./weights/postag_h100_l2_r0.001.2.th'))

    def forward(self, x):
        embedded = self.lang_model.forward(x)
        tags_out = list()
        chunk_out = list()
        tree_out = list()

        for batch in embedded:
            sent = np.zeros((1, len(batch), embedding_size), dtype=np.float32)

            for i, word in enumerate(batch):
                sent[0,i] = word.data.numpy()

            sent = torch.from_numpy(sent)
            sent = Variable(sent)

            tags, hn_tags = self.pos_tag.forward(sent)            
            chunks, hn_chunks = self.chunking.forward(sent, tags, hn_tags)
            dep_tree, hn_deep_tree = self.dependency.forward(sent, tags, hn_tags, chunks, hn_chunks)

            _, argmaxs = torch.max(dep_tree, dim=0)

            tree = list()
        
            for argmax in enumerate(argmaxs[0].data.numpy()):
                tree.append(argmax)
            
            tags_out.append(tags)
            chunk_out.append(chunks)
            tree_out.append(dep_tree)

        return tags_out, chunk_out, tree_out

from anytree import Node
from anytree import RenderTree

def build_tree(flat_tree, text):
    nodes = [None] * len(flat_tree)
    
    for child, parent in flat_tree:
        if child != parent:
            if nodes[parent] is None:
                nodes[parent] = Node(vec2word(text[parent]))
            nodes[child] = Node(vec2word(text[child]), parent=nodes[parent])

    return nodes



