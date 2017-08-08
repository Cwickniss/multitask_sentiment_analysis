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
from utils import vec2word
from utils import np2autograd
from lang_model import CharacterLanguageModel
from lang_model import embedding_size
from pos_tag import POSTag
from chunking import Chunking
from dependency import Dependency
from sentiment import SentimentClassification

# Hyperparams
learning_rate = 1e-3

postag_reg = 1e-3
chunking_reg = 1e-3
sentiment_reg = 1e-3

class JointMultiTaskModel(nn.Module):
    def __init__(self, mode='all'):
        super(JointMultiTaskModel, self).__init__()
        
        # models
        self.lang_model = CharacterLanguageModel()
        self.postag = POSTag()
        self.chunking = Chunking()
        self.dependency = Dependency()
        self.sentiment = SentimentClassification()
        
        # Mode - all or module_name
        self.mode = mode
    
    def embedded_batch(self, batch_x):
        embedded_batch = self.lang_model(batch_x)
    
        for batch in embedded_batch:
            sent = np.zeros((1, len(batch), embedding_size), dtype=np.float32)

            for i, word in enumerate(batch):
                sent[0,i] = word.data.numpy()
            
            sent = torch.from_numpy(sent)
            sent = Variable(sent)
            
            yield sent

    def get_postag(self, sentence):
        return self.postag(sentence)

    def get_chunking(self, sentence, tags, h_tags):
        return self.chunking(sentence, tags, h_tags)
    
    def get_dependency(self, sentence, tags, h_tags, chunks, h_chunks):
        return self.dependency(sentence, tags, h_tags, chunks, h_chunks)

    def get_sentiment(self, sentence, tags, h_tags, chunks, h_chunks, deps, h_deps):
        return self.sentiment(sentence, tags, h_tags, chunks, h_chunks, deps, h_deps)

    def run_all(self, x):
        sentences = self.embedded_batch(x)
        
        for s in sentences:
            y_pos, h_pos = self.get_postag(s)
            y_chk, h_chk = self.get_chunking(s, y_pos, h_pos)
            y_dep, h_dep = self.get_dependency(s, y_pos, h_pos, y_chk, h_chk)
            y_sent, h_sent = self.get_sentiment(s, y_pos, h_pos, y_chk, h_chk, y_dep, h_dep)
        
            yield y_pos, y_chk, y_dep, y_sent
    
    def forward(self, x):
        out = self.run_all(x)
        out = list(out)
        
        return out

    def postag_loss(self, y, yt):
        loss = F.cross_entropy(y, yt) \
             + (self.postag.w.norm() ** 2) * postag_reg

        return loss

    def chunking_loss(self, y, yt):
        loss = F.cross_entropy(y, yt) \
             + (self.chunking.w.norm() ** 2) * chunking_reg

        return loss

    def sentiment_loss(self, y, yt):
        loss = (yt.float() - y) ** 2 \
             + (self.sentiment.w.norm() ** 2) * sentiment_reg
        
        return loss
    
    def loss(self, y, tags, chunks, sentiment):
        losses = []
        length = len(y)

        for i in range(length):
            p_tag,   r_tag   = y[i][0], np2autograd(tags[i])
            p_chunk, r_chunk = y[i][1], np2autograd(chunks[i])
            p_sent,  r_sent  = y[i][3], np2autograd(sentiment[i])

            loss_tag  = self.postag_loss(p_tag, r_tag)
            loss_chk  = self.chunking_loss(p_chunk, r_chunk)
            loss_sent = self.sentiment_loss(p_sent, r_sent)
            
            loss = loss_tag * 0.3 + loss_chk * 0.2 + loss_sent * 0.5
            loss = loss / 3

            losses.append(loss)

        loss = losses[0]

        for i in range(1, length):
            loss += losses[i]

        loss = loss / length

        return loss

nb_epochs = 10
batch_size = 8
nb_batches = 30
    
gen = batch_generator(batch_size, nb_batches)

model = JointMultiTaskModel()
adam = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(nb_epochs):
    for batch in range(nb_batches):
        text, tags, chunks, sent = next(gen)
        
        out = model.forward(text)

        loss = model.loss(out, tags, chunks, sent)

        print("Epoch:", epoch,
              "Bacth:", batch,
              "Loss:", loss.data[0])

        adam.zero_grad()
        loss.backward()
        adam.step()

