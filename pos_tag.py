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
from utils import avg_cross_entropy_loss
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
        
        tags = np.zeros((x.size(0), max_sentence_size, nb_postags))
        
        for i, xx in enumerate(out[:]):
            tags[i] = self.fc(xx).data.numpy()
        
        tags = torch.from_numpy(tags)
        tags = Variable(tags)
        
        return tags, out

if __name__ == '__main__':

    class TestModel(nn.Module):
        def __init__(self, hidden_state_size, nb_rnn_layers):
            super(TestModel, self).__init__()

            self.lang_model = CharacterLanguageModel()
            self.pos_tag = POSTag(hidden_state_size, nb_rnn_layers)

        def forward(self, x):
            emb = self.lang_model.forward(x)
            emb = Variable(emb)

            pos_tags, hs_pos_tags = self.pos_tag(emb)

            return pos_tags, hs_pos_tags
    
    nb_batches = 10
    batch_size = 8
    epochs = 2
    hidden_state_size = 50
    nb_rnn_layers = 1

    gen = batch_generator(batch_size, nb_batches)

    model = TestModel(hidden_state_size, nb_rnn_layers)
    adam = optim.Adam(model.parameters(), lr=1e-2)

    fname = 'pos_tag_h{}_l{}'.format(hidden_state_size, nb_rnn_layers)

    losses = []
    
    try:
        for epoch in range(epochs):
            for batch in range(nb_batches):
                input_text, target_tags, sentiment = next(gen)

                target_tags = torch.from_numpy(target_tags).long()
                target_tags = Variable(target_tags, requires_grad=True)

                out_tags, _ = model.forward(input_text)

                loss = avg_cross_entropy_loss(out_tags, target_tags)

                print("Epoch:", epoch,
                      "Batch:", batch,
                      "Loss:", loss.data[0])

                adam.zero_grad()
                loss.backward()
                adam.step()

                if batch % 2 == 0:
                    losses.append(loss.data[0])
            
            torch.save(model.state_dict(), './weights/{}.th'.format(fname))
    finally:
        print("Generating plot.")
        fig = plt.figure()
        fig.suptitle('Part of Speech Bi-LSTM Tagging Task', fontsize=14, fontweight='bold')
        
        ax = fig.add_subplot(111)
        
        ax.plot(losses, color="red")
        
        ax.set_title('Hidden State Size: {}, Layers: {}'.format(hidden_state_size, 
                                                                nb_rnn_layers))

        ax.set_xlabel('Time')
        ax.set_ylabel('Loss')
        
        plt.savefig("./results/" + fname)
