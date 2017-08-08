import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
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
from dependency import Dependency
from dependency import dependency_hn_size

sentiment_hn_size = 100
sentiment_nb_layers = 2

class SentimentClassification(nn.Module):
    def __init__(self):
        super(SentimentClassification, self).__init__()
        
        self.input_size = embedding_size \
                        + nb_postags \
                        + nb_chunktags \
                        + max_sentence_size \
                        + postag_hn_size * 2 \
                        + chunking_hn_size * 2 \
                        + dependency_hn_size * 2

        self.w = nn.Parameter(torch.randn(sentiment_nb_layers * 2, 
                                          max_sentence_size, 
                                          sentiment_hn_size))
        self.h = nn.Parameter(torch.randn(sentiment_nb_layers * 2, 
                                          max_sentence_size,
                                          sentiment_hn_size))
        
        self.bi_lstm = nn.LSTM(self.input_size, 
                               sentiment_hn_size,
                               sentiment_nb_layers,
                               bidirectional=True)
        
        self.fc = nn.Linear(sentiment_hn_size * 2, 1)
    
    def forward(self, x, tags, hn_tags, chunks, hn_chunks, deps, hn_deps):
        tags = tags.view(1, -1, nb_postags)
        chunks = chunks.view(1, -1, nb_chunktags)
        deps = deps.view(1, deps.size(0), deps.size(1))
        
        gt = torch.cat([hn_chunks, hn_tags, hn_deps, x, tags, chunks, deps], dim=2)

        pad = torch.zeros(1, x.size(1), self.input_size - gt.size(2))
        pad = Variable(pad)
        
        gt = torch.cat([gt, pad], dim=2)
        
        out, hn = self.bi_lstm(gt, (self.h[:,:x.size(1),:], 
                                    self.w[:,:x.size(1),:]))
                
        sentiment = self.fc(out[0,-1].view(1,-1))
        sentiment = F.sigmoid(sentiment)
                
        return sentiment, out
