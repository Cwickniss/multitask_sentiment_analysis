import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
from utils import batch_generator
from utils import nb_classes
from utils import max_sentence_size
from lang_model import CharacterLanguageModel
from lang_model import embedding_size

class POSTag(nn.Module):
    def __init__(self, hidden_state_size = 100, nb_rnn_layers = 2):
        super(POSTag, self).__init__()

        self.w = nn.Parameter(torch.randn(nb_rnn_layers * 2, 1, hidden_state_size // 2))
        self.h = nn.Parameter(torch.randn(nb_rnn_layers * 2, 1, hidden_state_size // 2))
        
        self.bi_lstm = nn.LSTM(embedding_size, 
                               hidden_state_size // 2,
                               nb_rnn_layers,
                               batch_first=True,
                               bidirectional=True)

    def forward(self, x):
        print(self.h.size(), self.w.size())
        out, hn = self.bi_lstm(x, (self.h, self.w))
        
        return out, hn

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.lang_model = CharacterLanguageModel()
        self.pos_tag = POSTag()
    
    def forward(self, x):
        emb = self.lang_model.forward(x)
        emb = Variable(emb)

        tags, hn_pos = self.pos_tag(emb)
        
        return tags

    
gen = batch_generator(16, 100)    

text, sent = next(gen)

model = TestModel()

ys = model.forward(text)
print(ys)





