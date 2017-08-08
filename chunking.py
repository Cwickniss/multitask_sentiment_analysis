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

# Hyperparams
chunking_hn_size = 200
chunking_nb_layers = 2

chunking_postag_emb_size = 20

class Chunking(nn.Module):
    def __init__(self):
        super(Chunking, self).__init__()
        
        self.input_size = embedding_size \
                        + nb_postags \
                        + postag_hn_size * 2

        self.w = nn.Parameter(torch.randn(chunking_nb_layers * 2, 
                                          max_sentence_size, 
                                          chunking_hn_size))
        self.h = nn.Parameter(torch.randn(chunking_nb_layers * 2, 
                                          max_sentence_size,
                                          chunking_hn_size))

        self.embedding = nn.Embedding(nb_postags, chunking_postag_emb_size)
        
        self.aux_emb = torch.arange(0, nb_postags)
        self.aux_emb = Variable(self.aux_emb).long()
        
        self.bi_lstm = nn.LSTM(self.input_size, 
                               chunking_hn_size,
                               chunking_nb_layers,
                               bidirectional=True)
        
        self.fc = nn.Linear(chunking_hn_size * 2, nb_chunktags)

    def mult_pos_emb(self, tags, emb):
        # TODO: Implement in Torch
        tags = tags[:,np.newaxis].data.numpy()
        emb = emb.data.numpy()
        
        y_pos = tags * emb.T
        y_pos = np.sum(y_pos.T, axis=1).T

        y_pos = torch.from_numpy(y_pos)
        y_pos = Variable(y_pos)
        y_pos = y_pos.view(1, -1, nb_postags)

        return y_pos
        
    def forward(self, x, tags, hn_tags):
        l_emb = self.embedding(self.aux_emb)
        
        y_pos = self.mult_pos_emb(tags, l_emb)
        
        gt = torch.cat([hn_tags, x, y_pos], dim=2)
        
        out, hn = self.bi_lstm(gt, (self.h[:,:x.size(1),:], 
                                    self.w[:,:x.size(1),:]))
        
        chunk = self.fc(out[0])
        
        return chunk, out
