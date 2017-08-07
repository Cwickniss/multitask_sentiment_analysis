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
from utils import avg_cross_entropy_loss
from lang_model import CharacterLanguageModel
from lang_model import embedding_size
from pos_tag import POSTag

class Chunking(nn.Module):
    def __init__(self, hidden_state_size, nb_rnn_layers, tags_hidden_state_size):
        super(Chunking, self).__init__()
        
        self.input_size = embedding_size \
                        + nb_postags \
                        + tags_hidden_state_size * 2

        self.w = nn.Parameter(torch.randn(nb_rnn_layers * 2, 
                                          max_sentence_size, 
                                          hidden_state_size))
        self.h = nn.Parameter(torch.randn(nb_rnn_layers * 2, 
                                          max_sentence_size,
                                          hidden_state_size))

        self.embedding = nn.Embedding(nb_postags, 20)
        
        self.aux_emb = torch.arange(0, nb_postags)
        self.aux_emb = Variable(self.aux_emb).long()
        
        self.bi_lstm = nn.LSTM(self.input_size, 
                               hidden_state_size,
                               nb_rnn_layers,
                               bidirectional=True)
        
        self.fc = nn.Linear(hidden_state_size * 2, nb_chunktags)

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

hidden_state_size = 100
nb_rnn_layers = 2

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.lang_model = CharacterLanguageModel()
        self.pos_tag = POSTag(hidden_state_size, nb_rnn_layers)
        self.chunking = Chunking(hidden_state_size, nb_rnn_layers, hidden_state_size)
        
    def forward(self, x):
        embedded = self.lang_model.forward(x)
        tags_out = list()
        chunk_out = list()

        for batch in embedded:
            sent = np.zeros((1, len(batch), embedding_size), dtype=np.float32)

            for i, word in enumerate(batch):
                sent[0,i] = word.data.numpy()

            sent = torch.from_numpy(sent)
            sent = Variable(sent)

            tags, hn_tags = self.pos_tag.forward(sent)            
            chunked, hn_chunk = self.chunking.forward(sent, tags, hn_tags)
                        
            tags_out.append(tags)
            chunk_out.append(chunked)

        return tags_out, chunk_out
    
gen = batch_generator(8, 100)

model = TestModel()
adam = optim.Adam(model.parameters(), lr=1e-2)

text, tags, chunks, sent = next(gen)

out_tags, out_chunks = model.forward(text)

loss_tags = avg_cross_entropy_loss(out_tags, tags)
loss_chunks = avg_cross_entropy_loss(out_chunks, chunks)
print(loss_tags, loss_chunks)

