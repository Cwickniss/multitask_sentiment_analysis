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
from dependency import Dependency

class SentimentClassification(nn.Module):
    def __init__(self, hidden_state_size, 
                       nb_rnn_layers, 
                       tags_hidden_state_size,
                       chk_hidden_state_size,
                       dep_hidden_state_size):
        super(SentimentClassification, self).__init__()
        
        self.input_size = embedding_size \
                        + nb_postags \
                        + nb_chunktags \
                        + max_sentence_size \
                        + tags_hidden_state_size * 2 \
                        + chk_hidden_state_size * 2 \
                        + dep_hidden_state_size * 2

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
        
        self.fc = nn.Linear(hidden_state_size * 2, 1)
    
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

chunking_hidden_state_size = 200
chunking_nb_rnn_layers = 2

postag_hidden_state_size = 100
postag_nb_rnn_layers = 2

dependency_hidden_state_size = 100
dependency_nb_rnn_layers = 2

sentiment_hidden_state_size = 100
sentiment_nb_rnn_layers = 2

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
        
        self.sentiment = SentimentClassification(sentiment_hidden_state_size,
                                                 sentiment_nb_rnn_layers,
                                                 postag_hidden_state_size,
                                                 chunking_hidden_state_size,
                                                 dependency_hidden_state_size)

        self.pos_tag.load_state_dict(torch.load('./weights/postag_h100_l2_r0.001.2.th'))

    def forward(self, x):
        embedded = self.lang_model.forward(x)
        tags_out = list()
        chunk_out = list()
        tree_out = list()
        sentiment_out = list()

        for batch in embedded:
            sent = np.zeros((1, len(batch), embedding_size), dtype=np.float32)

            for i, word in enumerate(batch):
                sent[0,i] = word.data.numpy()

            sent = torch.from_numpy(sent)
            sent = Variable(sent)

            tags, hn_tags = self.pos_tag.forward(sent)            
            chunks, hn_chunks = self.chunking.forward(sent, tags, hn_tags)
            dep_tree, hn_deep_tree = self.dependency.forward(sent, tags, hn_tags, chunks, hn_chunks)

            sentiment, hn_sentiment = self.sentiment.forward(sent, tags, hn_tags, 
                                                             chunks, hn_chunks,
                                                             dep_tree, hn_deep_tree)
            
            tags_out.append(tags)
            chunk_out.append(chunks)
            tree_out.append(dep_tree)
            sentiment_out.append(sentiment)

        return tags_out, chunk_out, tree_out, sentiment_out

gen = batch_generator(2, 1000)

text, tags, chunks, sent = next(gen)

model = TestModel()

tags_out, chunk_out, tree_out, sentiment_out = model.forward(text)
