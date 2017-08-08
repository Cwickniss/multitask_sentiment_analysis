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

def np2autograd(var):
    var = np.array(var, dtype=np.float32)
    var = torch.from_numpy(var)
    var = Variable(var).long()

    return var
    
def chunking_objective_function(y_tag_pred, y_tag_target, W_tag,
                                y_chk_pred, y_chk_target, W_chk,
                                Lambda_tag = 0.001, Lambda_chk = 0.01):
    losses = []
    length = len(y_tag_pred)

    L2_norm_tag = Lambda_tag * (W_tag.norm() ** 2)
    L2_norm_chk = Lambda_chk * (W_chk.norm() ** 2)
    
    for i in range(length):
        y_tag = np2autograd(y_tag_target[i])
        y_chk = np2autograd(y_chk_target[i])
        
        loss = F.cross_entropy(y_tag_pred[i], y_tag) + L2_norm_tag \
             + F.cross_entropy(y_chk_pred[i], y_chk) + L2_norm_chk

        losses.append(loss)

        loss = losses[0]

    for i in range(1, length):
        loss += losses[i]

    loss = loss / length

    return loss
    
if __name__ == '__main__':

    chunking_hidden_state_size = 200
    chunking_nb_rnn_layers = 2

    postag_hidden_state_size = 100
    postag_nb_rnn_layers = 2

    learning_rate = 1e-2

    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

            self.lang_model = CharacterLanguageModel()

            self.pos_tag = POSTag(postag_hidden_state_size, 
                                  postag_nb_rnn_layers)

            self.chunking = Chunking(chunking_hidden_state_size, 
                                     chunking_nb_rnn_layers, 
                                     postag_hidden_state_size)
            
            self.pos_tag.load_state_dict(torch.load('./weights/postag_h100_l2_r0.001.2.th'))

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

    batch_size = 12

    nb_train_batches = 1000
    nb_val_batches = 200

    epochs = 10

    train_gen = batch_generator(batch_size, nb_train_batches)

    model = TestModel()
    adam = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
        correct, wrong = 0, 0
        
        for batch in range(nb_train_batches):            
            text, tags, chunks, sent = next(train_gen)

            out_tags, out_chunks = model.forward(text)
            
            loss = chunking_objective_function(out_tags, tags, model.pos_tag.w,
                                               out_chunks, chunks, model.chunking.w)
            
            for predictions, targets in zip(out_chunks, chunks):
                for pred, target in zip(predictions, targets):
                    pred = pred.data.numpy().argmax()
                    if pred == target:
                        correct += 1
                    else:
                        wrong += 1

            accuracy = correct / (correct + wrong)
            
            print("Epoch:", epoch,
                  "Batch:", batch,
                  "Loss:", loss.data[0],
                  "Accuracy:", accuracy)

            adam.zero_grad()
            loss.backward()
            adam.step()
