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
    """ Returns the Part of Speech Tag for each word
        embedding in a given sentence.
    """
    def __init__(self, hidden_state_size, nb_rnn_layers):
        super(POSTag, self).__init__()

        self.w = nn.Parameter(torch.randn(nb_rnn_layers * 2, 
                                          max_sentence_size, 
                                          hidden_state_size))
        self.h = nn.Parameter(torch.randn(nb_rnn_layers * 2, 
                                          max_sentence_size,
                                          hidden_state_size))

        # Bidirectional LSTM
        self.bi_lstm = nn.LSTM(embedding_size, 
                               hidden_state_size,
                               nb_rnn_layers,
                               bidirectional=True)
        
        self.fc = nn.Linear(hidden_state_size * 2, nb_postags)

    def forward(self, x):
        # Runs the LSTM for each word-vector in the sentence x
        out, hn = self.bi_lstm(x, (self.h[:,:x.size(1),:],
                                   self.w[:,:x.size(1),:]))

        # Runs a linear classifier on the outputed state vector
        tags = self.fc(out[0])
        
        return tags, out

def postag_objective_function(predicted, targets, W, Lambda = 0.1):
    """ The objective function used to calculate the loss during the 
        training of the Part of Speech Module.
        
        J(y_t, y', W) = -Sum_s(Sum_t( log(p(y_t = y' | h_t)) ))
                      + Lamba * || W_pos ||^2
        
        where Lambda is the L2 regularization term.
    """
    losses = []
    length = len(predicted)
    L2_norm = Lambda * (W.norm() ** 2)

    for i in range(length):
        target = np.array(targets[i], dtype=np.float32)
        target = torch.from_numpy(target)
        target = Variable(target).long()

        loss = F.nll_loss(predicted[i], target) + L2_norm

        losses.append(loss)

        loss = losses[0]

    for i in range(1, length):
        loss += losses[i]

    loss = loss / length

    return loss
    
if __name__ == '__main__':

    class TestModel(nn.Module):
        def __init__(self, hidden_state_size, nb_rnn_layers):
            super(TestModel, self).__init__()

            self.lang_model = CharacterLanguageModel()
            self.postag = POSTag(hidden_state_size, nb_rnn_layers)

        def forward(self, x):
            embedded = self.lang_model.forward(x)
            out_tags, out_hn_tags = list(), list()
            
            for batch in embedded:
                sent = np.zeros((1, len(batch), embedding_size), dtype=np.float32)
                
                for i, word in enumerate(batch):
                    sent[0,i] = word.data.numpy()
                
                sent = torch.from_numpy(sent)
                sent = Variable(sent)
                
                tags, hn_tags = self.postag.forward(sent)
                
                out_tags.append(tags)
                out_hn_tags.append(hn_tags)
            
            return out_tags, out_hn_tags
    
    # training params
    batch_size = 12
    
    nb_train_batches = 2000
    nb_val_batches = 40
    nb_test_batches = 8300
    
    epochs = 6
    
    # hyperparams
    hidden_state_size = 200
    nb_rnn_layers = 2
    postag_regularization = 0.001
    
    train_gen = batch_generator(batch_size, nb_train_batches)
    val_gen = batch_generator(batch_size, nb_val_batches, 
                              skip_batches=nb_train_batches)
    
    model = TestModel(hidden_state_size, nb_rnn_layers)
    adam = optim.Adam(model.parameters(), lr=1e-2)

    fname = 'postag_h{}_l{}_r{}'.format(hidden_state_size, 
                                        nb_rnn_layers, 
                                        postag_regularization)

    train_acc = []
    val_acc = []

    try:
        for epoch in range(epochs):
            
            correct, wrong = 0, 0
            
            for batch in range(nb_train_batches):
                input_text, target_tags, _, _ = next(train_gen)

                out_tags, _ = model.forward(input_text)
                
                loss = postag_objective_function(out_tags, 
                                                 target_tags, 
                                                 model.postag.w,
                                                 Lambda=postag_regularization)

                for predictions, targets in zip(out_tags, target_tags):
                    for pred, target in zip(predictions, targets):
                        pred = pred.data.numpy().argmax()
                        if pred == target:
                            correct += 1
                        else:
                            wrong += 1

                accuracy = (correct) / (correct + wrong)
                
                print("Epoch:", epoch,
                      "Batch:", batch,
                      "Loss:", loss.data[0],
                      "Accuracy:", accuracy)

                adam.zero_grad()
                loss.backward()
                adam.step()

            train_acc.append(accuracy)
            
            correct, wrong = 0, 0

            for batch in range(nb_val_batches):
                input_text, target_tags, _, _ = next(val_gen)

                out_tags, _ = model.forward(input_text)
                               
                for predictions, targets in zip(out_tags, target_tags):
                    for pred, target in zip(predictions, targets):
                        pred = pred.data.numpy().argmax()
                        if pred == target:
                            correct += 1
                        else:
                            wrong += 1
                
                accuracy = (correct) / (correct + wrong)
                
                print("Validation",
                      "Epoch:", epoch,
                      "Batch:", batch,
                      "Accuracy:", accuracy)
            
            val_acc.append(accuracy)

            torch.save(model.postag.state_dict(), './weights/{}.{}.th'.format(fname, epoch))
    finally:
        #print("Generating plot.")
        #fig = plt.figure()
        #fig.suptitle('Part of Speech Bi-LSTM Tagging Task', fontsize=14, fontweight='bold')
        #ax = fig.add_subplot(111)
        #ax.plot(losses, color="red")
        #ax.set_title('Hidden State Size: {}, Layers: {}'.format(hidden_state_size, 
        #                                                        nb_rnn_layers))
        #ax.set_xlabel('Batch Count')
        #ax.set_ylabel('Loss')
        #plt.savefig("./results/" + fname)
        
        torch.save(model.postag.state_dict(), './weights/{}.final.th'.format(fname))

        with open('./results/' + fname + '.txt', 'w') as file:
            file.write(' '.join(map(str, train_acc)))
            file.write(' '.join(map(str, val_acc)))
