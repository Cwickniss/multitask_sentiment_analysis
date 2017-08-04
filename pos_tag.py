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
from utils import max_sentence_size
from lang_model import CharacterLanguageModel
from lang_model import embedding_size

class POSTag(nn.Module):
    def __init__(self, hidden_state_size = 50, nb_rnn_layers = 1):
        super(POSTag, self).__init__()

        self.w = nn.Parameter(torch.randn(nb_rnn_layers * 2, 
                                          max_sentence_size, 
                                          hidden_state_size))
        self.h = nn.Parameter(torch.randn(nb_rnn_layers * 2, 
                                          max_sentence_size,
                                          hidden_state_size))
        
        self.bi_lstm = nn.LSTM(embedding_size, 
                               hidden_state_size,
                               nb_rnn_layers,
                               bidirectional=True)
        
        self.fc = nn.Linear(hidden_state_size * 2, nb_postags)

    def forward(self, x):
        out, hn = self.bi_lstm(x, (self.h, self.w))
        
        tags = [self.fc(xx) for xx in out[:]]
        
        return out, tags

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.lang_model = CharacterLanguageModel()
        self.pos_tag = POSTag()
    
    def forward(self, x):
        emb = self.lang_model.forward(x)
        emb = Variable(emb)

        hn_pos = self.pos_tag(emb)
        
        return hn_pos

gen = batch_generator(16, 100)

text, tags, sent = next(gen)

model = TestModel()

ys, tags = model.forward(text)
print(len(tags))




