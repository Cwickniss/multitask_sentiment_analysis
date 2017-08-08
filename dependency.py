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
from lang_model import CharacterLanguageModel
from lang_model import embedding_size
from pos_tag import POSTag
from pos_tag import postag_hn_size
from chunking import Chunking
from chunking import chunking_hn_size

# Hyperparams
dependency_hn_size = 100
dependency_nb_layers = 2

class Dependency(nn.Module):
    def __init__(self):
        super(Dependency, self).__init__()
        
        self.input_size = embedding_size \
                        + nb_postags \
                        + nb_chunktags \
                        + postag_hn_size * 2 \
                        + chunking_hn_size * 2

        self.w = nn.Parameter(torch.randn(dependency_nb_layers * 2, 
                                          max_sentence_size, 
                                          dependency_hn_size))
        self.h = nn.Parameter(torch.randn(dependency_nb_layers * 2, 
                                          max_sentence_size,
                                          dependency_hn_size))
        
        self.bi_lstm = nn.LSTM(self.input_size, 
                               dependency_hn_size,
                               dependency_nb_layers,
                               bidirectional=True)
        
        self.wd = nn.Parameter(torch.randn(dependency_hn_size * 2))
        self.fc = nn.Linear(dependency_hn_size * 2, 1)
        
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
